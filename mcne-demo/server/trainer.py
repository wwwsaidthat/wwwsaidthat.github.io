"""Cora training backend for the MCNE web demo.

This is a compact, demo-oriented implementation. It trains:
1) a conventional 768-D graph contrastive encoder whose prefixes are truncated;
2) one MCNE-style encoder jointly optimized at seven nested prefix dimensions.

Replace the loss implementation with the research repository's exact experiment
code when strict paper reproduction is required. The HTTP contract can remain
unchanged, so the front end does not need to be rewritten.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv


ProgressCallback = Callable[[dict], None]
CancelCallback = Callable[[], bool]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GCNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.prelu(x, torch.tensor(0.25, device=x.device))
        return self.conv2(x, edge_index)


def augment_graph(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    feature_drop: float,
    edge_drop: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_mask = torch.rand(x.shape[1], device=x.device) >= feature_drop
    x_aug = x * feature_mask.to(x.dtype)
    edge_mask = torch.rand(edge_index.shape[1], device=edge_index.device) >= edge_drop
    return x_aug, edge_index[:, edge_mask]


def contrastive_per_sample(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / tau
    targets = torch.arange(z1.shape[0], device=z1.device)
    loss_12 = F.cross_entropy(logits, targets, reduction="none")
    loss_21 = F.cross_entropy(logits.T, targets, reduction="none")
    return (loss_12 + loss_21) * 0.5, logits


def relational_distribution(logits: torch.Tensor) -> torch.Tensor:
    affinities = F.relu(logits) + 1e-8
    return affinities / affinities.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def symmetric_kl_student_teacher(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    s = relational_distribution(student)
    t = relational_distribution(teacher)
    forward = F.kl_div(s.log(), t.detach(), reduction="batchmean")
    reverse = F.kl_div(t.log(), s.detach(), reduction="batchmean")
    return 0.5 * (forward + reverse)


def mcne_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    dimensions: list[int],
    base_tau: float = 0.2,
    cdmd_weight: float = 0.1,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    logits_per_dim: list[torch.Tensor] = []
    previous_hardness: torch.Tensor | None = None
    max_dim = dimensions[-1]

    for dim in dimensions:
        tau = base_tau * math.sqrt(max_dim / dim)
        per_sample, logits = contrastive_per_sample(z1[:, :dim], z2[:, :dim], tau)
        if previous_hardness is not None:
            hard_weights = 1.0 + previous_hardness / previous_hardness.mean().clamp_min(1e-8)
            per_sample = per_sample * hard_weights.detach()
        dimension_weight = math.sqrt(dim / max_dim)
        losses.append(dimension_weight * per_sample.mean())
        logits_per_dim.append(logits)
        with torch.no_grad():
            negative = logits.detach().clone()
            negative.fill_diagonal_(-torch.inf)
            previous_hardness = negative.max(dim=1).values.softmax(dim=0) * negative.shape[0]

    base = torch.stack(losses).sum() / sum(math.sqrt(d / max_dim) for d in dimensions)
    teacher = logits_per_dim[-1]
    distillation = torch.stack([
        symmetric_kl_student_teacher(logits, teacher)
        for logits in logits_per_dim[:-1]
    ]).mean()
    return base + cdmd_weight * distillation


@torch.no_grad()
def encode(model: nn.Module, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(x, edge_index)


@torch.no_grad()
def nearest_centroid_predictions(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    embeddings = F.normalize(embeddings, dim=-1)
    classes = torch.unique(labels)
    centroids = torch.stack([
        embeddings[train_mask & (labels == cls)].mean(dim=0)
        for cls in classes
    ])
    centroids = F.normalize(centroids, dim=-1)
    predictions = (embeddings @ centroids.T).argmax(dim=-1)
    accuracy = (predictions[test_mask] == labels[test_mask]).float().mean().item() * 100
    return accuracy, predictions


def evaluate_all_dimensions(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    dimensions: Iterable[int],
) -> tuple[dict[str, float], dict[str, int], dict[int, torch.Tensor], torch.Tensor]:
    z = encode(model, x, edge_index)
    accuracies: dict[str, float] = {}
    errors: dict[str, int] = {}
    predictions: dict[int, torch.Tensor] = {}
    test_count = int(test_mask.sum().item())
    for dim in dimensions:
        accuracy, pred = nearest_centroid_predictions(z[:, :dim], labels, train_mask, test_mask)
        accuracies[str(dim)] = round(accuracy, 4)
        errors[str(dim)] = int(round(test_count * (1 - accuracy / 100)))
        predictions[dim] = pred.detach().cpu()
    return accuracies, errors, predictions, z


@torch.no_grad()
def build_plot_series(
    embeddings: torch.Tensor,
    predictions: dict[int, torch.Tensor],
    labels: torch.Tensor,
    dimensions: Iterable[int],
    max_points: int = 560,
) -> dict[str, list[list[float | int | bool]]]:
    node_count = embeddings.shape[0]
    sample = torch.linspace(0, node_count - 1, steps=min(max_points, node_count), device=embeddings.device).long()
    sampled_labels = labels[sample].detach().cpu()
    series: dict[str, list[list[float | int | bool]]] = {}
    for dim in dimensions:
        values = embeddings[:, :dim].float()
        values = values - values.mean(dim=0, keepdim=True)
        _, _, basis = torch.pca_lowrank(values, q=2, center=False)
        coords = values[sample] @ basis[:, :2]
        coords = coords / coords.abs().amax(dim=0, keepdim=True).clamp_min(1e-8)
        coords = coords.detach().cpu()
        sampled_predictions = predictions[dim][sample.cpu()]
        series[str(dim)] = [
            [
                round(float(coords[i, 0]), 5),
                round(float(coords[i, 1]), 5),
                int(sampled_labels[i]),
                bool(sampled_predictions[i] != sampled_labels[i]),
            ]
            for i in range(coords.shape[0])
        ]
    return series


def choose_device(mode: str) -> torch.device:
    normalized = mode.lower().strip()
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE_MODE=cuda, but CUDA is unavailable")
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(
    config: dict,
    update: ProgressCallback,
    should_cancel: CancelCallback,
    *,
    data_root: Path,
    checkpoint_root: Path,
    device_mode: str = "auto",
) -> dict:
    started = time.perf_counter()
    seed = int(config.get("seed", 42))
    epochs = int(config.get("epochs", 100))
    dimensions = sorted({int(value) for value in config.get("dimensions", [32, 64, 128, 256, 384, 512, 768])})
    max_dimension = int(config.get("max_dimension", max(dimensions)))
    if dimensions[-1] != max_dimension:
        dimensions.append(max_dimension)
    batch_size = min(int(config.get("batch_size", 512)), 1024)
    seed_everything(seed)
    device = choose_device(device_mode)

    dataset = Planetoid(root=str(data_root), name="Cora")
    data = dataset[0].to(device)
    encoder_width = int(config.get("encoder_width", 512))
    baseline = GCNEncoder(dataset.num_features, encoder_width, max_dimension).to(device)
    mcne = GCNEncoder(dataset.num_features, encoder_width, max_dimension).to(device)
    baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=1e-3, weight_decay=1e-5)
    mcne_optimizer = torch.optim.Adam(mcne.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch in range(1, epochs + 1):
        if should_cancel():
            raise RuntimeError("training cancelled")
        baseline.train(); mcne.train()
        x1, edge1 = augment_graph(data.x, data.edge_index, 0.2, 0.2)
        x2, edge2 = augment_graph(data.x, data.edge_index, 0.2, 0.2)
        node_ids = torch.randperm(data.num_nodes, device=device)[:batch_size]

        baseline_optimizer.zero_grad(set_to_none=True)
        baseline_z1 = baseline(x1, edge1)[node_ids]
        baseline_z2 = baseline(x2, edge2)[node_ids]
        baseline_loss = contrastive_per_sample(baseline_z1, baseline_z2, 0.2)[0].mean()
        baseline_loss.backward()
        baseline_optimizer.step()

        mcne_optimizer.zero_grad(set_to_none=True)
        mcne_z1 = mcne(x1, edge1)[node_ids]
        mcne_z2 = mcne(x2, edge2)[node_ids]
        nested_loss = mcne_loss(mcne_z1, mcne_z2, dimensions)
        nested_loss.backward()
        mcne_optimizer.step()

        combined_loss = float((baseline_loss + nested_loss).detach().cpu().item())
        payload = {
            "epoch": epoch,
            "total_epochs": epochs,
            "progress": epoch / epochs,
            "loss": combined_loss,
            "elapsed_seconds": time.perf_counter() - started,
            "device": str(device),
        }
        if epoch == 1 or epoch == epochs or epoch % max(5, epochs // 10) == 0:
            mcne_acc, _, _, _ = evaluate_all_dimensions(
                mcne, data.x, data.edge_index, data.y, data.train_mask, data.test_mask, dimensions
            )
            payload["metrics"] = {"mcne_accuracy": mcne_acc}
        update(payload)

    base_acc, base_errors, base_predictions, baseline_embeddings = evaluate_all_dimensions(
        baseline, data.x, data.edge_index, data.y, data.train_mask, data.test_mask, dimensions
    )
    mcne_acc, mcne_errors, mcne_predictions, mcne_embeddings = evaluate_all_dimensions(
        mcne, data.x, data.edge_index, data.y, data.train_mask, data.test_mask, dimensions
    )

    test_nodes = data.test_mask.nonzero(as_tuple=False).flatten()
    selected_node = int(test_nodes[min(71, len(test_nodes) - 1)].item())
    selected_vector = mcne_embeddings[selected_node].detach().cpu().float().tolist()
    label_names = dataset[0].y.unique().numel()
    prediction_summary = {
        str(dim): int(mcne_predictions[dim][selected_node].item())
        for dim in dimensions
    }
    baseline_plot = build_plot_series(baseline_embeddings, base_predictions, data.y, dimensions)
    mcne_plot = build_plot_series(mcne_embeddings, mcne_predictions, data.y, dimensions)

    checkpoint_dir = checkpoint_root
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": mcne.state_dict(),
        "input_dim": dataset.num_features,
        "hidden_dim": encoder_width,
        "output_dim": max_dimension,
        "dimensions": dimensions,
        "seed": seed,
    }, checkpoint_dir / "mcne-cora-demo.pt")

    return {
        "metrics": {
            "baseline_accuracy": base_acc,
            "mcne_accuracy": mcne_acc,
            "baseline_errors": base_errors,
            "mcne_errors": mcne_errors,
        },
        "result": {
            "selected_node": selected_node,
            "selected_label": int(data.y[selected_node].item()),
            "selected_vector": selected_vector,
            "selected_predictions": prediction_summary,
            "baseline_plot": baseline_plot,
            "mcne_plot": mcne_plot,
            "num_classes": int(label_names),
            "dimensions": dimensions,
        },
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint_dir / "mcne-cora-demo.pt"),
    }

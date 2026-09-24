(() => {
  "use strict";

  const DIMS = [32, 64, 128, 256, 384, 512, 768];
  const COLORS = ["#ff5d70", "#ff934b", "#f8ce46", "#33d17a", "#2bd2ca", "#448cff", "#9a6dff"];
  const NAMES = ["红维·轻量核心", "橙维·快速感知", "黄维·稳健表达", "绿维·平衡部署", "青维·关系增强", "蓝维·精细区分", "紫维·完整语义"];
  const DEVICES = ["IoT", "移动端", "边缘设备", "实时服务", "边缘服务器", "工作站", "云端"];
  const ART_SCALES = [2.55, 2.18, 1.82, 1.52, 1.3, 1.14, 1];
  const MOCK = {
    baseline: [63.8, 68.7, 72.4, 77.1, 79.2, 80.3, 81.1],
    mcne: [73.5, 78.8, 81.6, 83.4, 84.1, 84.5, 84.7],
    baseErrors: [402, 344, 286, 232, 211, 199, 192],
    mcneErrors: [275, 224, 184, 158, 149, 144, 141]
  };

  const cfg = window.MCNE_DEMO_CONFIG || {};

  const state = {
    active: 2,
    running: false,
    timer: null,
    jobId: null,
    epoch: 0,
    loss: null,
    elapsed: 0,
    startedAt: 0,
    metrics: structuredClone(MOCK),
    vectors: new Map(),
    remoteResult: null,
    connection: loadConnection()
  };

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];

  function loadConnection() {
    let saved = {};
    try { saved = JSON.parse(sessionStorage.getItem("mcne-connection") || "{}"); } catch (_) {}
    return {
      useGpu: saved.useGpu ?? Boolean(cfg.USE_GPU_BACKEND),
      apiUrl: saved.apiUrl || cfg.API_BASE_URL || "",
      apiKey: saved.apiKey || ""
    };
  }

  function apiUrl(path) {
    return `${state.connection.apiUrl.replace(/\/$/, "")}${path}`;
  }

  function apiHeaders(withJson = false) {
    const headers = {};
    if (withJson) headers["Content-Type"] = "application/json";
    if (state.connection.apiKey) headers["X-API-Key"] = state.connection.apiKey;
    return headers;
  }

  function seededRandom(seed) {
    let value = seed % 2147483647;
    if (value <= 0) value += 2147483646;
    return () => (value = value * 16807 % 2147483647) / 2147483647;
  }

  function selectedNodeId() {
    return Number($("#selected-node")?.textContent.match(/\d+/)?.[0] || 871);
  }

  function getFullVector(nodeId) {
    if (!state.vectors.has(nodeId)) {
      const random = seededRandom(nodeId * 97 + 2027);
      state.vectors.set(nodeId, Array.from({length: 768}, () => random() * 2 - 1));
    }
    return state.vectors.get(nodeId);
  }

  function initDimensions() {
    const orbit = $("#dimension-orbit");
    const deploy = $("#deployment-dolls");
    DIMS.forEach((dim, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "dimension-button";
      button.textContent = dim;
      button.style.setProperty("--dim-color", COLORS[index]);
      button.setAttribute("aria-label", `选择 ${dim} 维向量前缀`);
      button.addEventListener("click", () => setDimension(index));
      orbit.appendChild(button);

      const doll = document.createElement("button");
      doll.type = "button";
      doll.className = "deploy-doll";
      doll.textContent = `${dim}D`;
      doll.style.setProperty("--doll-color", COLORS[index]);
      doll.style.setProperty("--size", `${34 + index * 4}px`);
      doll.setAttribute("aria-label", `派遣 ${dim} 维套娃`);
      doll.addEventListener("click", () => selectDeployment(index));
      deploy.appendChild(doll);
    });
  }

  function initVector() {
    const map = $("#vector-heatmap");
    for (let i = 0; i < 96; i += 1) {
      const cell = document.createElement("span");
      cell.className = "vector-cell";
      map.appendChild(cell);
    }
  }

  function setDimension(index) {
    state.active = Math.max(0, Math.min(DIMS.length - 1, index));
    const dim = DIMS[state.active];
    $$(".dimension-button").forEach((button, i) => {
      button.classList.toggle("included", i <= state.active);
      button.classList.toggle("active", i === state.active);
      button.setAttribute("aria-pressed", String(i === state.active));
    });
    $$(".deploy-doll").forEach((button, i) => button.classList.toggle("active", i === state.active));
    $("#current-doll-title").textContent = `${dim}D · ${NAMES[state.active]}`;
    $("#device-chip").textContent = DEVICES[state.active];
    $("#metric-memory").textContent = formatBytes(dim * 4);
    $("#metric-compute").textContent = `${Math.round(dim / 768 * 100)}%`;
    $("#vector-prefix-label").textContent = `仅截取前 ${dim} 维`;
    const artFrame = $("#hero-art-frame");
    artFrame.style.setProperty("--layer-color", COLORS[state.active]);
    $("#hero-art").style.transform = `scale(${ART_SCALES[state.active]})`;
    $("#hero-layer-dim").textContent = `${dim}D`;
    $("#hero-layer-name").textContent = NAMES[state.active];
    updateScores();
    updateVector();
    updatePredictions();
    drawPlots();
  }

  function updateScores() {
    const i = state.active;
    const baseline = Number(state.metrics.baseline[i]);
    const mcne = Number(state.metrics.mcne[i]);
    const delta = mcne - baseline;
    $("#metric-accuracy").textContent = `${mcne.toFixed(1)}%`;
    $("#baseline-score").textContent = `${baseline.toFixed(1)}%`;
    $("#mcne-score").textContent = `${mcne.toFixed(1)}%`;
    $("#score-delta").textContent = `${delta >= 0 ? "+" : ""}${delta.toFixed(1)}`;
    $("#baseline-error-count").textContent = `错误节点 ${state.metrics.baseErrors[i]}`;
    $("#mcne-error-count").textContent = `错误节点 ${state.metrics.mcneErrors[i]}`;
  }

  function updateVector() {
    const dim = DIMS[state.active];
    const enabledCount = Math.max(1, Math.round(dim / 768 * 96));
    const vector = state.remoteResult?.selected_vector || getFullVector(selectedNodeId());
    $$(".vector-cell").forEach((cell, index) => {
      const sourceIndex = Math.round(index / 95 * Math.max(0, vector.length - 1));
      const value = Number(vector[sourceIndex] || 0);
      const lightness = 42 + Math.abs(value) * 25;
      cell.style.background = value >= 0 ? `hsl(261 84% ${lightness}%)` : `hsl(183 70% ${lightness}%)`;
      cell.classList.toggle("enabled", index < enabledCount);
      cell.classList.toggle("boundary", index === enabledCount - 1);
      cell.title = `维度 ${index + 1}: ${value.toFixed(3)}`;
    });
  }

  function updatePredictions() {
    const journey = $("#prediction-journey");
    const nodeShift = Number($("#selected-node").textContent.match(/\d+/)?.[0] || 871) % 3;
    journey.innerHTML = "";
    DIMS.forEach((dim, index) => {
      const remotePrediction = state.remoteResult?.selected_predictions?.[String(dim)];
      const remoteLabel = state.remoteResult?.selected_label;
      const correct = remotePrediction == null ? index >= Math.max(1, 3 - nodeShift) : Number(remotePrediction) === Number(remoteLabel);
      const step = document.createElement("div");
      step.className = `prediction-step ${correct ? "correct" : "wrong"}${index === state.active ? " active" : ""}`;
      step.innerHTML = `<b>${dim}D</b><span>${correct ? "✓" : "×"}</span><em>${correct ? "Neural" : index % 2 ? "Rule" : "Prob."}</em>`;
      journey.appendChild(step);
    });
  }

  function drawPlots() {
    drawPlot($("#baseline-canvas"), false);
    drawPlot($("#mcne-canvas"), true);
  }

  function drawPlot(canvas, isMcne) {
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    const remoteSeries = isMcne ? state.remoteResult?.mcne_plot : state.remoteResult?.baseline_plot;
    const remotePoints = remoteSeries?.[String(DIMS[state.active])];
    if (Array.isArray(remotePoints) && remotePoints.length) {
      remotePoints.forEach(([px, py, label, wrong]) => {
        const x = 28 + (Number(px) + 1) * .5 * (width - 56);
        const y = 24 + (Number(py) + 1) * .5 * (height - 48);
        ctx.beginPath(); ctx.arc(x, y, wrong ? 5.2 : 4, 0, Math.PI * 2);
        ctx.fillStyle = [COLORS[6], COLORS[4], COLORS[2], COLORS[3], COLORS[0], COLORS[1], COLORS[5]][Number(label) % 7];
        ctx.globalAlpha = .8; ctx.fill(); ctx.globalAlpha = 1;
        if (wrong) { ctx.strokeStyle = COLORS[0]; ctx.lineWidth = 2.1; ctx.stroke(); }
      });
      return;
    }
    const random = seededRandom((isMcne ? 7717 : 3319) + state.active * 83);
    const centers = [[130,105],[300,245],[470,110],[485,270]];
    const colors = [COLORS[6], COLORS[4], COLORS[2], COLORS[3]];
    const separation = isMcne ? 1 + state.active * .035 : .82 + state.active * .045;
    const errorRate = isMcne ? .20 - state.active * .018 : .34 - state.active * .03;
    for (let group = 0; group < centers.length; group += 1) {
      for (let j = 0; j < 42; j += 1) {
        const angle = random() * Math.PI * 2;
        const radius = Math.sqrt(random()) * (isMcne ? 66 : 84) / separation;
        const x = centers[group][0] + Math.cos(angle) * radius + (random() - .5) * 9;
        const y = centers[group][1] + Math.sin(angle) * radius + (random() - .5) * 9;
        const wrong = random() < errorRate;
        ctx.beginPath(); ctx.arc(x, y, wrong ? 5.4 : 4.1, 0, Math.PI * 2);
        ctx.fillStyle = colors[group]; ctx.globalAlpha = .8; ctx.fill(); ctx.globalAlpha = 1;
        if (wrong) { ctx.strokeStyle = COLORS[0]; ctx.lineWidth = 2.2; ctx.stroke(); }
      }
    }
  }

  function selectDeployment(index) {
    setDimension(index);
    const dim = DIMS[index];
    const accuracy = state.metrics.mcne[index];
    const fitsMemory = dim * 4 <= 1024;
    const fitsAccuracy = accuracy >= 82;
    const result = $("#mission-result");
    result.className = "mission-result";
    if (fitsMemory && fitsAccuracy) {
      result.classList.add("success"); result.innerHTML = `<span>${dim}D · ${formatBytes(dim * 4)} · ${accuracy.toFixed(1)}%</span><strong>任务成功 ✓</strong>`;
    } else if (fitsMemory) {
      result.classList.add("warning"); result.innerHTML = `<span>${dim}D 满足资源预算</span><strong>准确率不足</strong>`;
    } else {
      result.classList.add("failure"); result.innerHTML = `<span>${dim}D 需要 ${formatBytes(dim * 4)}</span><strong>超出预算</strong>`;
    }
  }

  function formatBytes(bytes) {
    return bytes >= 1024 ? `${(bytes / 1024).toFixed(bytes % 1024 ? 1 : 0)} KiB` : `${bytes} B`;
  }

  async function startTraining() {
    if (state.running) { stopTraining("训练已暂停"); return; }
    const epochs = Math.max(10, Math.min(1000, Number($("#epochs-input").value) || 100));
    const seed = Math.max(0, Number($("#seed-input").value) || 42);
    $("#epoch-total").textContent = epochs;
    if (state.connection.useGpu) await startRemoteTraining(epochs, seed);
    else startMockTraining(epochs, seed);
  }

  function startMockTraining(epochs, seed) {
    state.running = true; state.epoch = 0; state.loss = 4.8; state.startedAt = performance.now();
    setTrainingUi(true, "模拟训练中");
    const random = seededRandom(seed + 9001);
    state.timer = setInterval(() => {
      state.epoch = Math.min(epochs, state.epoch + Math.max(1, Math.ceil(epochs / 60)));
      const ratio = state.epoch / epochs;
      state.loss = 4.8 * Math.exp(-3.2 * ratio) + .28 + random() * .05;
      state.elapsed = (performance.now() - state.startedAt) / 1000;
      renderTrainingProgress(epochs);
      if (state.epoch >= epochs) {
        clearInterval(state.timer); state.timer = null; state.running = false;
        setTrainingUi(false, "模拟训练完成");
        $("#status-message").textContent = "演示训练完成。当前指标仍为页面示意数据；启用 GPU 后端后将替换为服务器真实结果。";
      }
    }, 180);
  }

  async function startRemoteTraining(epochs, seed) {
    if (!state.connection.apiUrl || state.connection.apiUrl.includes("YOUR_GPU_SERVER")) {
      showConnectionDialog("请先填写 GPU 训练服务器地址。"); return;
    }
    state.running = true; state.startedAt = performance.now();
    setTrainingUi(true, "连接 GPU");
    try {
      const response = await fetch(apiUrl("/api/train"), {
        method: "POST", headers: apiHeaders(true),
        body: JSON.stringify({epochs, seed, dimensions: DIMS, max_dimension: 768})
      });
      if (!response.ok) throw new Error(await response.text() || `HTTP ${response.status}`);
      const payload = await response.json();
      state.jobId = payload.job_id;
      $("#status-message").textContent = `GPU 训练任务已创建：${state.jobId}`;
      await pollRemoteJob(epochs);
    } catch (error) {
      stopTraining("连接失败");
      $("#status-message").textContent = `GPU 服务连接失败：${error.message}`;
    }
  }

  async function pollRemoteJob(epochs) {
    if (!state.running || !state.jobId) return;
    try {
      const response = await fetch(apiUrl(`/api/train/${encodeURIComponent(state.jobId)}`), {headers: apiHeaders()});
      if (!response.ok) throw new Error(await response.text() || `HTTP ${response.status}`);
      const job = await response.json();
      state.epoch = Number(job.epoch || 0); state.loss = job.loss == null ? null : Number(job.loss); state.elapsed = Number(job.elapsed_seconds || 0);
      if (job.metrics) applyRemoteMetrics(job.metrics);
      if (job.result) {
        state.remoteResult = job.result;
        if (job.result.selected_node != null) {
          $("#selected-node").textContent = `Node #${job.result.selected_node}`;
          $("#vector-node-label").textContent = `Node #${job.result.selected_node} · 同一条768维完整向量`;
        }
        if (job.result.selected_vector) updateVector();
        updatePredictions(); drawPlots();
      }
      renderTrainingProgress(Number(job.total_epochs || epochs));
      if (job.status === "completed") {
        state.running = false; setTrainingUi(false, "GPU 训练完成");
        $("#status-message").textContent = `真实训练完成，设备：${job.device || "GPU"}，耗时 ${state.elapsed.toFixed(1)} 秒。`;
        return;
      }
      if (job.status === "failed" || job.status === "cancelled") throw new Error(job.error || job.status);
      state.timer = setTimeout(() => pollRemoteJob(epochs), Number(cfg.POLL_INTERVAL_MS || 1000));
    } catch (error) {
      stopTraining("训练失败"); $("#status-message").textContent = `训练任务失败：${error.message}`;
    }
  }

  function applyRemoteMetrics(metrics) {
    const readSeries = (name, fallback) => DIMS.map((dim, index) => Number(metrics[name]?.[String(dim)] ?? metrics[name]?.[dim] ?? fallback[index]));
    state.metrics.baseline = readSeries("baseline_accuracy", state.metrics.baseline);
    state.metrics.mcne = readSeries("mcne_accuracy", state.metrics.mcne);
    state.metrics.baseErrors = readSeries("baseline_errors", state.metrics.baseErrors);
    state.metrics.mcneErrors = readSeries("mcne_errors", state.metrics.mcneErrors);
    updateScores(); drawPlots();
  }

  function renderTrainingProgress(total) {
    const ratio = Math.max(0, Math.min(1, state.epoch / total));
    $("#training-progress").style.width = `${ratio * 100}%`;
    $("#epoch-value").textContent = state.epoch;
    $("#epoch-total").textContent = total;
    $("#loss-value").textContent = state.loss == null ? "—" : state.loss.toFixed(3);
    $("#elapsed-value").textContent = `${state.elapsed.toFixed(1)}s`;
  }

  function setTrainingUi(running, label) {
    state.running = running;
    $("#training-status").textContent = label;
    $("#train-button span:last-child").textContent = running ? "暂停训练" : "从零训练";
    $("#train-button .play-icon").textContent = running ? "Ⅱ" : "▶";
  }

  function stopTraining(label) {
    state.running = false;
    if (state.timer) { clearInterval(state.timer); clearTimeout(state.timer); state.timer = null; }
    setTrainingUi(false, label);
  }

  function showConnectionDialog(message = "") {
    if (!cfg.ALLOW_RUNTIME_CONNECTION_OVERRIDE) return;
    $("#gpu-toggle").checked = state.connection.useGpu;
    $("#api-url-input").value = state.connection.apiUrl.includes("YOUR_GPU_SERVER") ? "" : state.connection.apiUrl;
    $("#api-key-input").value = state.connection.apiKey;
    if (message) $("#status-message").textContent = message;
    $("#connection-dialog").showModal();
  }

  function saveConnection(event) {
    event.preventDefault();
    state.connection = {
      useGpu: $("#gpu-toggle").checked,
      apiUrl: $("#api-url-input").value.trim(),
      apiKey: $("#api-key-input").value.trim()
    };
    sessionStorage.setItem("mcne-connection", JSON.stringify(state.connection));
    updateModeBadge(); $("#connection-dialog").close();
    $("#status-message").textContent = state.connection.useGpu ? "GPU 后端已启用。开始训练时将调用服务器接口。" : "已切换到浏览器演示数据。";
  }

  function updateModeBadge() {
    const badge = $("#mode-badge");
    badge.textContent = state.connection.useGpu ? "GPU 后端" : "演示数据";
    badge.style.color = state.connection.useGpu ? "var(--cyan)" : "var(--gold)";
  }

  function initNavigation() {
    $$(".nav-link").forEach(button => button.addEventListener("click", () => {
      document.getElementById(button.dataset.target)?.scrollIntoView({behavior:"smooth"});
      $$(".nav-link").forEach(item => item.classList.toggle("active", item === button));
    }));
  }

  function initEvents() {
    $("#train-button").addEventListener("click", startTraining);
    $("#connection-button").addEventListener("click", () => showConnectionDialog());
    $("#save-connection-button").addEventListener("click", saveConnection);
    $("#random-node-button").addEventListener("click", () => {
      const next = 700 + Math.floor(Math.random() * 1700);
      $("#selected-node").textContent = `Node #${next}`;
      $("#vector-node-label").textContent = `Node #${next} · 同一条768维完整向量`;
      state.remoteResult = null;
      updatePredictions(); updateVector();
    });
  }

  initDimensions(); initVector(); initNavigation(); initEvents(); updateModeBadge(); setDimension(state.active); renderTrainingProgress(100);
})();

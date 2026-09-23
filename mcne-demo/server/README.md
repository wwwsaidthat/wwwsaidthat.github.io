# MCNE GPU 训练服务

## 需要上传到 GPU 服务器的文件

上传整个 `server/` 目录：

```text
server/
├── app.py
├── trainer.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── nginx-example.conf
└── .env.example
```

服务器运行后还会生成：

```text
server/
├── data/          # Cora 数据；首次训练会自动下载
└── checkpoints/   # 最新训练得到的模型参数
```

如果服务器不能访问互联网，请在可联网机器上先运行一次，随后把生成的 `data/` 一并上传。

## 推荐部署：Docker + NVIDIA GPU

服务器需要：

- NVIDIA 驱动；
- Docker；
- NVIDIA Container Toolkit；
- 至少一个可用 GPU；
- 一个可从浏览器访问的 HTTPS 域名或地址。

复制环境配置：

```bash
cp .env.example .env
```

编辑 `.env`：

```dotenv
MCNE_API_KEY=请生成一个足够长的随机字符串
DEVICE_MODE=auto
ALLOWED_ORIGINS=https://wwwsaidthat.github.io,http://localhost:8000
DATA_ROOT=/app/data/Cora
```

启动：

```bash
docker compose up -d --build
```

健康检查：

```bash
curl http://127.0.0.1:8000/api/health
```

## 不使用 Docker

在匹配服务器 CUDA 版本的 Python 环境中安装 PyTorch 和 PyTorch Geometric，然后执行：

```bash
python -m pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

GPU 服务只应启动一个 worker，以免多个进程同时占用显存。API 内部也只允许一个训练任务运行，其余请求会收到“GPU忙”的响应。

## HTTPS 与公网访问

不建议直接把 `8000` 端口暴露到公网。使用 Nginx 将 HTTPS 请求转发到 `127.0.0.1:8000`，参考 `nginx-example.conf`。

最终前端需要的是：

```text
API_BASE_URL=https://你的训练服务域名
API_KEY=与服务器 MCNE_API_KEY 相同的临时访问密钥
```

不要向网页提供 SSH 用户名或 SSH 密码。网页通过 HTTPS API 访问训练服务，不通过 SSH 连接服务器。

## GPU / 演示数据开关

前端的 `USE_GPU_BACKEND` 控制是否调用本服务：

- `false`：前端使用写死的演示数据；
- `true`：前端调用本服务，服务根据 `DEVICE_MODE` 自动选择 CUDA 或 CPU。

`DEVICE_MODE=cuda` 会在没有 CUDA 时直接报错；`DEVICE_MODE=auto` 会自动回退到 CPU。

## 模型说明

`trainer.py` 是面向 Demo 的轻量 Cora 实现，包含普通高维截断基线和 MCNE 风格的多前缀训练。它保持网页所需的 API 数据结构。

如果需要与论文实验代码完全一致，可将 `run_training()` 内部替换为正式 MCNE 训练流程，并继续返回相同字段，前端无需修改。

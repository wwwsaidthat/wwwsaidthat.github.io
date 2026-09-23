# MCNE 七维套娃 Demo

该目录包含两部分：

- `index.html`、`styles.css`、`app.js`、`config.js`：部署在 GitHub Pages 的静态前端；
- `server/`：部署在 GPU 服务器的训练 API。

## 前端模式开关

编辑 `config.js`：

```js
window.MCNE_DEMO_CONFIG = {
  USE_GPU_BACKEND: false,
  API_BASE_URL: "https://YOUR_GPU_SERVER:PORT",
  POLL_INTERVAL_MS: 1000,
  ALLOW_RUNTIME_CONNECTION_OVERRIDE: true
};
```

- `USE_GPU_BACKEND: false`：完全使用浏览器内示意数据，不连接服务器；
- `USE_GPU_BACKEND: true`：点击训练后调用 `API_BASE_URL`；
- 页面右上角“服务器设置”也可以临时覆盖该配置，信息仅存入当前浏览器的 `sessionStorage`。

不要把 SSH 密码、服务器密码或真实 API Key 写进 `config.js`。如果 API 启用了访问密钥，在浏览器的“服务器设置”中临时输入。

## 本地预览

从网站仓库根目录执行：

```bash
python3 -m http.server 8000
```

访问：

```text
http://localhost:8000/mcne-demo/
```

## GitHub Pages

该个人主页从仓库 `main` 分支根目录部署。推送后 Demo 地址为：

```text
https://wwwsaidthat.github.io/mcne-demo/
```

GitHub Pages 只能托管静态前端，不能运行 PyTorch。真实训练必须部署 `server/` 到单独的 GPU 服务器。

## API 约定

- `GET /api/health`：检查 GPU 服务；
- `POST /api/train`：创建训练任务；
- `GET /api/train/{job_id}`：读取训练状态和结果；
- `DELETE /api/train/{job_id}`：请求取消任务。

如果服务器设置了 `MCNE_API_KEY`，前端通过 `X-API-Key` 请求头发送临时访问密钥。

## 视觉资产

`assets/seven-layer-matryoshka.png` 为本项目生成的原创七色科技套娃视觉，不使用官方动画角色素材。

window.MCNE_DEMO_CONFIG = {
  // false：完全使用页面内演示数据；true：连接 GPU 训练服务器。
  USE_GPU_BACKEND: false,

  // 部署 GPU 服务后填写，例如：https://gpu.example.edu:8443
  // 不要在 GitHub 中写入密码或真实 API Key。
  API_BASE_URL: "https://YOUR_GPU_SERVER:PORT",

  // 浏览器轮询训练状态的间隔。
  POLL_INTERVAL_MS: 1000,

  // 服务器可用时设为 true；页面也允许在“服务器设置”中临时覆盖。
  ALLOW_RUNTIME_CONNECTION_OVERRIDE: true
};

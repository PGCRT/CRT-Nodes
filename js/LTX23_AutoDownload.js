import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "CRT_LTX23AutoDownload";
const NODE_ALIASES = new Set(["LTX 2.3 AutoDownload (CRT)", "CRT_LTX23AutoDownload"]);
const STYLE_ID = "crt-ltx23-autodownload-v2";
const MIN_WIDTH = 420;
const MIN_HEIGHT = 1; // Minimal backend height
const DEBUG = false;

function log(...args) {
  if (DEBUG) console.log("[CRT LTX23 AutoDownload]", ...args);
}

// Model definitions matching the Python backend
const MODEL_DEFINITIONS = {
  diffusion_model: {
    label: "Diffusion Model",
    filename: "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors",
    size: "~23GB",
  },
  ic_lora: {
    label: "IC LoRA Union Control",
    filename: "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    size: "~2.8GB",
  },
  video_vae: {
    label: "Video VAE",
    filename: "LTX23_video_vae_bf16.safetensors",
    size: "~380MB",
  },
  audio_vae: {
    label: "Audio VAE",
    filename: "LTX23_audio_vae_bf16.safetensors",
    size: "~95MB",
  },
  tae_vae: {
    label: "TAE VAE (Preview)",
    filename: "taeltx2_3.safetensors",
    size: "~95MB",
  },
  clip_gemma: {
    label: "CLIP Gemma",
    filename: "gemma_3_12B_it_fp8_e4m3fn.safetensors",
    size: "~11GB",
  },
  clip_projection: {
    label: "CLIP Projection",
    filename: "ltx-2.3_text_projection_bf16.safetensors",
    size: "~25MB",
  },
  spatial_upscaler: {
    label: "Spatial Upscaler",
    filename: "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    size: "~6GB",
  },
};

const MODEL_ORDER = [
  "diffusion_model",
  "ic_lora",
  "video_vae",
  "audio_vae",
  "tae_vae",
  "clip_gemma",
  "clip_projection",
  "spatial_upscaler",
];

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;

  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .crt-ltx23-ad-root {
      width: 100%;
      box-sizing: border-box;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
      user-select: none;
      -webkit-user-select: none;
    }

    .crt-ltx23-ad-shell {
      --bg-base: #09090b;
      --bg-surface: #111114;
      --bg-elevated: #18181b;
      --bg-hover: #1f1f23;
      --border-subtle: rgba(255, 255, 255, 0.04);
      --border-default: rgba(255, 255, 255, 0.08);
      --border-focus: rgba(139, 92, 246, 0.5);
      --text-primary: #fafafa;
      --text-secondary: #71717a;
      --text-tertiary: #52525b;
      --accent: #8b5cf6;
      --accent-soft: rgba(139, 92, 246, 0.12);
      --accent-glow: rgba(139, 92, 246, 0.25);
      --success: #22c55e;
      --success-soft: rgba(34, 197, 94, 0.12);
      --success-glow: rgba(34, 197, 94, 0.35);
      --warning: #f97316;
      --warning-soft: rgba(249, 115, 22, 0.12);
      --warning-glow: rgba(249, 115, 22, 0.35);
      --error: #ef4444;

      width: calc(100% - 8px);
      margin: 4px;
      padding: 8px;
      border-radius: 12px;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      color: var(--text-primary);
      box-sizing: border-box;
      pointer-events: none;
      position: relative;
      overflow: hidden;
    }

    .crt-ltx23-ad-shell::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.3), transparent);
    }

    .crt-ltx23-ad-shell * {
      box-sizing: border-box;
      pointer-events: auto;
      user-select: none;
      -webkit-user-select: none;
    }

    .crt-ltx23-ad-title {
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--text-tertiary);
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .crt-ltx23-ad-title::before {
      content: '';
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 8px var(--accent-glow);
    }

    .crt-ltx23-ad-list {
      display: flex;
      flex-direction: column;
      gap: 4px;
      margin-bottom: 10px;
      padding-right: 0;
    }

    .crt-ltx23-ad-row {
      display: grid;
      grid-template-columns: 1fr auto auto;
      align-items: center;
      gap: 8px;
      padding: 6px 8px;
      background: var(--bg-base);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      transition: all 200ms ease;
      position: relative;
    }

    .crt-ltx23-ad-row.missing {
      border-color: rgba(249, 115, 22, 0.3);
    }

    .crt-ltx23-ad-row.missing.pulsing {
      animation: row-pulse-orange 2.5s ease-in-out infinite;
    }

    .crt-ltx23-ad-row.downloading {
      border-color: rgba(139, 92, 246, 0.4);
    }

    .crt-ltx23-ad-row.present {
      border-color: rgba(34, 197, 94, 0.3);
    }

    .crt-ltx23-ad-row.present.sync-pulse {
      animation: row-sync-pulse 0.8s ease-in-out;
    }

    .crt-ltx23-ad-row-info {
      display: flex;
      flex-direction: column;
      gap: 2px;
      min-width: 0;
    }

    .crt-ltx23-ad-row-label {
      font-size: 11px;
      font-weight: 500;
      color: var(--text-primary);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .crt-ltx23-ad-row-meta {
      font-size: 9px;
      color: var(--text-tertiary);
      display: flex;
      gap: 6px;
      align-items: center;
    }

    .crt-ltx23-ad-row-size {
      color: var(--text-secondary);
    }

    .crt-ltx23-ad-row-status {
      display: flex;
      align-items: center;
      gap: 4px;
      font-size: 10px;
      font-weight: 500;
    }

    .crt-ltx23-ad-row-status.missing {
      color: var(--warning);
    }

    .crt-ltx23-ad-row-status.present {
      color: var(--success);
    }

    .crt-ltx23-ad-status-dot {
      width: 5px;
      height: 5px;
      border-radius: 50%;
    }

    .crt-ltx23-ad-status-dot.missing {
      background: var(--warning);
      box-shadow: 0 0 4px var(--warning-glow);
    }

    .crt-ltx23-ad-status-dot.present {
      background: var(--success);
      box-shadow: 0 0 4px var(--success-glow);
    }

    .crt-ltx23-ad-download-btn {
      padding: 4px 10px;
      border-radius: 4px;
      border: 1px solid var(--border-subtle);
      background: var(--bg-surface);
      color: var(--text-secondary);
      font-size: 9px;
      font-weight: 600;
      cursor: pointer;
      transition: all 150ms ease;
      display: flex;
      align-items: center;
      gap: 4px;
      white-space: nowrap;
    }

    .crt-ltx23-ad-download-btn:hover:not(:disabled) {
      background: var(--bg-elevated);
      border-color: var(--border-default);
      color: var(--text-primary);
    }

    .crt-ltx23-ad-download-btn.want {
      background: var(--success-soft);
      border-color: var(--success);
      color: var(--success);
    }

    .crt-ltx23-ad-download-btn.want:hover:not(:disabled) {
      background: var(--success);
      color: var(--bg-base);
      box-shadow: 0 0 8px var(--success-glow);
    }

    .crt-ltx23-ad-download-btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    .crt-ltx23-ad-download-btn.downloading {
      background: var(--accent-soft);
      border-color: var(--accent);
      color: var(--accent);
      animation: btn-pulse 1.5s ease-in-out infinite;
    }

    .crt-ltx23-ad-progress {
      position: absolute;
      bottom: 0;
      left: 0;
      height: 2px;
      background: linear-gradient(90deg, var(--accent), var(--accent-glow));
      border-radius: 0 0 0 6px;
      transition: width 200ms ease;
    }

    .crt-ltx23-ad-progress-text {
      font-size: 9px;
      color: var(--accent);
      margin-left: 4px;
    }

    .crt-ltx23-ad-main-btn {
      width: 100%;
      padding: 8px 16px;
      border-radius: 6px;
      border: 1px solid var(--border-subtle);
      background: var(--bg-base);
      color: var(--text-secondary);
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      cursor: pointer;
      transition: all 200ms ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
    }

    .crt-ltx23-ad-main-btn:hover {
      background: var(--bg-elevated);
      border-color: var(--border-default);
      color: var(--text-primary);
    }

    .crt-ltx23-ad-main-btn.checking {
      background: var(--accent-soft);
      border-color: var(--accent);
      color: var(--accent);
    }

    .crt-ltx23-ad-main-btn.ok {
      background: var(--success-soft);
      border-color: var(--success);
      color: var(--success);
      animation: ok-pulse 2s ease-in-out infinite;
    }

    .crt-ltx23-ad-main-btn.ok::before {
      content: '✓';
    }

    .crt-ltx23-ad-hint {
      margin-top: 8px;
      padding: 6px 10px;
      background: var(--bg-base);
      border-radius: 4px;
      border: 1px solid var(--border-subtle);
      font-size: 9px;
      color: var(--text-tertiary);
      line-height: 1.3;
      text-align: center;
    }

    .crt-ltx23-ad-spinner {
      width: 12px;
      height: 12px;
      border: 2px solid var(--border-default);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    @keyframes row-pulse-orange {
      0%, 100% {
        box-shadow: 0 0 0 0 rgba(249, 115, 22, 0);
        border-color: rgba(249, 115, 22, 0.2);
      }
      50% {
        box-shadow: 0 0 12px 2px rgba(249, 115, 22, 0.15);
        border-color: rgba(249, 115, 22, 0.5);
      }
    }

    @keyframes row-sync-pulse {
      0% {
        box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.4);
        border-color: rgba(34, 197, 94, 0.3);
      }
      50% {
        box-shadow: 0 0 20px 4px rgba(34, 197, 94, 0.25);
        border-color: rgba(34, 197, 94, 0.7);
      }
      100% {
        box-shadow: 0 0 0 0 rgba(34, 197, 94, 0);
        border-color: rgba(34, 197, 94, 0.3);
      }
    }

    @keyframes btn-pulse {
      0%, 100% {
        opacity: 1;
      }
      50% {
        opacity: 0.7;
      }
    }

    @keyframes ok-pulse {
      0%, 100% {
        box-shadow: 0 0 0 0 rgba(34, 197, 94, 0);
      }
      50% {
        box-shadow: 0 0 16px 4px rgba(34, 197, 94, 0.2);
      }
    }

    /* Staggered animation delays for pulsing rows */
    .crt-ltx23-ad-row.missing.pulsing:nth-child(1) { animation-delay: 0ms; }
    .crt-ltx23-ad-row.missing.pulsing:nth-child(2) { animation-delay: 150ms; }
    .crt-ltx23-ad-row.missing.pulsing:nth-child(3) { animation-delay: 300ms; }
    .crt-ltx23-ad-row.missing.pulsing:nth-child(4) { animation-delay: 450ms; }
    .crt-ltx23-ad-row.missing.pulsing:nth-child(5) { animation-delay: 600ms; }
    .crt-ltx23-ad-row.missing.pulsing:nth-child(6) { animation-delay: 750ms; }
    .crt-ltx23-ad-row.missing.pulsing:nth-child(7) { animation-delay: 900ms; }
    .crt-ltx23-ad-row.missing.pulsing:nth-child(8) { animation-delay: 1050ms; }
  `;
  document.head.appendChild(style);
}

class LTX23AutoDownloadUI {
  constructor(node) {
    this.node = node;
    this.container = null;
    this.shell = null;
    this.rowElements = new Map();
    this.mainButton = null;
    this.status = "idle"; // idle, checking, checked
    this.modelStatus = {};
    this.downloadQueue = new Set();
    this.downloading = new Set();
    this.downloaded = new Set();
    this.pollInterval = null;

    this.init();
  }

  init() {
    ensureStyles();
    this.hideNativeWidgets();
    this.createContainer();
    this.buildLayout();
    this.startPolling();
  }

  hideNativeWidgets() {
    for (const widget of this.node.widgets || []) {
      if (widget.name === "ltx23_autodownload_ui") continue;
      widget.hidden = true;
      widget.computeSize = () => [0, -6];
    }
  }

  createContainer() {
    if (this.container) return;
    this.container = document.createElement("div");
    this.container.className = "crt-ltx23-ad-root";
    this.domWidget = this.node.addDOMWidget("ltx23_autodownload_ui", "div", this.container, {
      serialize: false,
    });
    if (this.domWidget) {
      // Stop DOM widget from contributing to node backend height
      this.domWidget.computeSize = () => [0, 0];
    }
  }

  buildLayout() {
    this.container.innerHTML = "";
    this.rowElements.clear();

    this.shell = document.createElement("div");
    this.shell.className = "crt-ltx23-ad-shell";

    const title = document.createElement("div");
    title.className = "crt-ltx23-ad-title";
    title.textContent = "LTX Video 2.3 - Model Manager";
    this.shell.appendChild(title);

    const list = document.createElement("div");
    list.className = "crt-ltx23-ad-list";

    for (const modelKey of MODEL_ORDER) {
      const def = MODEL_DEFINITIONS[modelKey];
      const row = this.createModelRow(modelKey, def);
      list.appendChild(row);
      this.rowElements.set(modelKey, row);
    }

    this.shell.appendChild(list);

    this.mainButton = document.createElement("button");
    this.mainButton.type = "button";
    this.mainButton.className = "crt-ltx23-ad-main-btn";
    this.mainButton.textContent = "Check Models";
    this.mainButton.addEventListener("click", () => this.onMainButtonClick());
    this.shell.appendChild(this.mainButton);

    const hint = document.createElement("div");
    hint.className = "crt-ltx23-ad-hint";
    hint.textContent = "Click 'Check Models' to verify model availability. Missing models will be highlighted.";
    this.shell.appendChild(hint);

    this.container.appendChild(this.shell);
  }

  createModelRow(modelKey, def) {
    const row = document.createElement("div");
    row.className = "crt-ltx23-ad-row";
    row.dataset.model = modelKey;

    const info = document.createElement("div");
    info.className = "crt-ltx23-ad-row-info";

    const label = document.createElement("div");
    label.className = "crt-ltx23-ad-row-label";
    label.textContent = def.label;
    info.appendChild(label);

    const meta = document.createElement("div");
    meta.className = "crt-ltx23-ad-row-meta";
    meta.innerHTML = `<span class="crt-ltx23-ad-row-filename">${def.filename}</span><span class="crt-ltx23-ad-row-size">${def.size}</span>`;
    info.appendChild(meta);

    row.appendChild(info);

    const status = document.createElement("div");
    status.className = "crt-ltx23-ad-row-status";
    status.innerHTML = `<span class="crt-ltx23-ad-status-dot"></span><span class="crt-ltx23-ad-status-text">Checking...</span>`;
    row.appendChild(status);

    const downloadBtn = document.createElement("button");
    downloadBtn.type = "button";
    downloadBtn.className = "crt-ltx23-ad-download-btn";
    downloadBtn.textContent = "I WANT THAT";
    downloadBtn.addEventListener("click", () => this.onDownloadClick(modelKey));
    row.appendChild(downloadBtn);

    const progressBar = document.createElement("div");
    progressBar.className = "crt-ltx23-ad-progress";
    progressBar.style.width = "0%";
    row.appendChild(progressBar);

    return row;
  }

  async onMainButtonClick() {
    if (this.status === "checking") return;

    if (this.status === "checked") {
      // Reset and check again
      this.resetState();
    }

    this.status = "checking";
    this.mainButton.className = "crt-ltx23-ad-main-btn checking";
    this.mainButton.innerHTML = '<span class="crt-ltx23-ad-spinner"></span> Checking...';

    try {
      await this.checkModels();
    } catch (err) {
      console.error("[LTX23 AutoDownload] Check failed:", err);
    }
  }

  resetState() {
    this.status = "idle";
    this.modelStatus = {};
    this.downloadQueue.clear();
    this.downloading.clear();
    this.downloaded.clear();

    for (const [key, row] of this.rowElements) {
      row.className = "crt-ltx23-ad-row";
      const statusEl = row.querySelector(".crt-ltx23-ad-row-status");
      const dotEl = row.querySelector(".crt-ltx23-ad-status-dot");
      const textEl = row.querySelector(".crt-ltx23-ad-status-text");
      const btn = row.querySelector(".crt-ltx23-ad-download-btn");
      const progress = row.querySelector(".crt-ltx23-ad-progress");

      statusEl.className = "crt-ltx23-ad-row-status";
      dotEl.className = "crt-ltx23-ad-status-dot";
      textEl.textContent = "Checking...";
      btn.className = "crt-ltx23-ad-download-btn";
      btn.textContent = "I WANT THAT";
      btn.disabled = false;
      progress.style.width = "0%";
    }

    this.mainButton.className = "crt-ltx23-ad-main-btn";
    this.mainButton.textContent = "Check Models";
  }

  async checkModels() {
    try {
      const resp = await api.fetchApi("/crt/ltx23/check_models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const data = await resp.json();
      this.modelStatus = data;

      let hasMissing = false;

      for (const [key, present] of Object.entries(data)) {
        const row = this.rowElements.get(key);
        if (!row) continue;

        const statusEl = row.querySelector(".crt-ltx23-ad-row-status");
        const dotEl = row.querySelector(".crt-ltx23-ad-status-dot");
        const textEl = row.querySelector(".crt-ltx23-ad-status-text");
        const btn = row.querySelector(".crt-ltx23-ad-download-btn");

        if (present) {
          row.className = "crt-ltx23-ad-row present";
          statusEl.className = "crt-ltx23-ad-row-status present";
          dotEl.className = "crt-ltx23-ad-status-dot present";
          textEl.textContent = "Present";
          btn.textContent = "✓ OK";
          btn.disabled = true;
          this.downloaded.add(key);
        } else {
          row.className = "crt-ltx23-ad-row missing pulsing";
          statusEl.className = "crt-ltx23-ad-row-status missing";
          dotEl.className = "crt-ltx23-ad-status-dot missing";
          textEl.textContent = "Missing";
          btn.className = "crt-ltx23-ad-download-btn want";
          hasMissing = true;
        }
      }

      this.status = "checked";

      if (hasMissing) {
        this.mainButton.className = "crt-ltx23-ad-main-btn";
        this.mainButton.textContent = "Check Again";
      } else {
        this.mainButton.className = "crt-ltx23-ad-main-btn ok";
        this.mainButton.textContent = "OK!";
      }
    } catch (err) {
      console.error("[LTX23 AutoDownload] Check failed:", err);
      this.status = "idle";
      this.mainButton.className = "crt-ltx23-ad-main-btn";
      this.mainButton.textContent = "Retry Check";
    }
  }

  async onDownloadClick(modelKey) {
    if (this.downloading.has(modelKey) || this.downloaded.has(modelKey)) return;

    this.downloadQueue.add(modelKey);

    const row = this.rowElements.get(modelKey);
    const btn = row.querySelector(".crt-ltx23-ad-download-btn");

    btn.className = "crt-ltx23-ad-download-btn downloading";
    btn.textContent = "Downloading...";
    btn.disabled = true;

    try {
      const resp = await api.fetchApi("/crt/ltx23/download_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_type: modelKey }),
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      this.downloading.add(modelKey);
      this.startPolling();
    } catch (err) {
      console.error(`[LTX23 AutoDownload] Download start failed for ${modelKey}:`, err);
      btn.className = "crt-ltx23-ad-download-btn want";
      btn.textContent = "I WANT THAT";
      btn.disabled = false;
      this.downloadQueue.delete(modelKey);
    }
  }

  startPolling() {
    if (this.pollInterval) return;

    this.pollInterval = setInterval(() => this.pollStatus(), 500);
  }

  stopPolling() {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  }

  async pollStatus() {
    const modelsToCheck = [...this.downloading, ...this.downloadQueue];
    if (modelsToCheck.length === 0) {
      this.stopPolling();
      return;
    }

    for (const modelKey of modelsToCheck) {
      try {
        const resp = await api.fetchApi("/crt/ltx23/download_status", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_type: modelKey }),
        });

        if (!resp.ok) continue;

        const data = await resp.json();
        this.updateDownloadProgress(modelKey, data);
      } catch (err) {
        // Ignore poll errors
      }
    }
  }

  updateDownloadProgress(modelKey, data) {
    const row = this.rowElements.get(modelKey);
    if (!row) return;

    const progress = row.querySelector(".crt-ltx23-ad-progress");
    const btn = row.querySelector(".crt-ltx23-ad-download-btn");

    if (data.complete) {
      // Download finished
      this.downloading.delete(modelKey);
      this.downloadQueue.delete(modelKey);
      this.downloaded.add(modelKey);

      row.className = "crt-ltx23-ad-row present sync-pulse";

      const statusEl = row.querySelector(".crt-ltx23-ad-row-status");
      const dotEl = row.querySelector(".crt-ltx23-ad-status-dot");
      const textEl = row.querySelector(".crt-ltx23-ad-status-text");

      statusEl.className = "crt-ltx23-ad-row-status present";
      dotEl.className = "crt-ltx23-ad-status-dot present";
      textEl.textContent = "Downloaded";
      btn.className = "crt-ltx23-ad-download-btn";
      btn.textContent = "✓ OK";
      btn.disabled = true;
      progress.style.width = "100%";

      // Check if all downloads are complete
      this.checkAllComplete();
    } else if (data.progress !== undefined) {
      // Still downloading
      progress.style.width = `${data.progress}%`;
      if (data.progress > 0) {
        btn.textContent = `${data.progress.toFixed(0)}%`;
      }
    }
  }

  checkAllComplete() {
    const allModels = MODEL_ORDER;
    const allDownloaded = allModels.every((key) => this.modelStatus[key] || this.downloaded.has(key));

    if (allDownloaded) {
      this.mainButton.className = "crt-ltx23-ad-main-btn ok";
      this.mainButton.textContent = "OK!";

      // Trigger synchronized pulse animation
      for (const row of this.rowElements.values()) {
        row.classList.remove("sync-pulse");
        void row.offsetWidth; // Force reflow
        row.classList.add("sync-pulse");
      }
    }
  }

  rebuild() {
    this.hideNativeWidgets();
    this.buildLayout();
    if (this.status === "checked") {
      // Restore state
      for (const [key, present] of Object.entries(this.modelStatus)) {
        const row = this.rowElements.get(key);
        if (!row) continue;

        const statusEl = row.querySelector(".crt-ltx23-ad-row-status");
        const dotEl = row.querySelector(".crt-ltx23-ad-status-dot");
        const textEl = row.querySelector(".crt-ltx23-ad-status-text");
        const btn = row.querySelector(".crt-ltx23-ad-download-btn");

        if (present || this.downloaded.has(key)) {
          row.className = "crt-ltx23-ad-row present";
          statusEl.className = "crt-ltx23-ad-row-status present";
          dotEl.className = "crt-ltx23-ad-status-dot present";
          textEl.textContent = this.downloaded.has(key) ? "Downloaded" : "Present";
          btn.textContent = "✓ OK";
          btn.disabled = true;
        } else {
          row.className = "crt-ltx23-ad-row missing pulsing";
          statusEl.className = "crt-ltx23-ad-row-status missing";
          dotEl.className = "crt-ltx23-ad-status-dot missing";
          textEl.textContent = "Missing";
          btn.className = "crt-ltx23-ad-download-btn want";
        }
      }

      const hasMissing = MODEL_ORDER.some((k) => !this.modelStatus[k] && !this.downloaded.has(k));
      if (!hasMissing) {
        this.mainButton.className = "crt-ltx23-ad-main-btn ok";
        this.mainButton.textContent = "OK!";
      }
    }
  }

  destroy() {
    this.stopPolling();
    this.container?.remove();
    this.container = null;
    this.domWidget = null;
  }
}

// Compute compact height for node - since this node has no inputs/outputs,
// it just needs minimal height (just the title bar area)
function compactHeightForNode(node) {
  const titleHeight = Number(globalThis.LiteGraph?.NODE_TITLE_HEIGHT) || 30;
  return Math.max(MIN_HEIGHT, titleHeight + 8);
}

function clampNodeBounds(node) {
  const targetHeight = compactHeightForNode(node);
  if (node.size?.[0] !== MIN_WIDTH || node.size?.[1] !== targetHeight) {
    node.size = [MIN_WIDTH, targetHeight];
    node.setDirtyCanvas?.(true, false);
  }
  return targetHeight;
}

function applyNodeVisuals(node) {
  node.bgcolor = "transparent";
  node.color = "transparent";
  node.title = "";
  node.resizable = false;
  node.clip_area = false;
  node.flags ??= {};
  node.flags.clip_area = false;

  // Store compact height function for UI use
  node._ltx23ADCompactHeight = () => clampNodeBounds(node);

  // Override computeSize
  if (!node._ltx23ADOriginalComputeSize) {
    node._ltx23ADOriginalComputeSize = node.computeSize;
  }
  node.computeSize = function(out) {
    const size = [MIN_WIDTH, compactHeightForNode(this)];
    if (out) {
      out[0] = size[0];
      out[1] = size[1];
      return out;
    }
    return size;
  };

  // Override setSize
  if (!node._ltx23ADOriginalSetSize) {
    node._ltx23ADOriginalSetSize = node.setSize;
  }
  node.setSize = function() {
    const clamped = [MIN_WIDTH, compactHeightForNode(this)];
    this.size = clamped;
    return clamped;
  };

  // Override draw hooks to enforce clamp
  if (!node._ltx23ADOriginalOnDrawBackground) {
    node._ltx23ADOriginalOnDrawBackground = node.onDrawBackground;
  }
  node.onDrawBackground = function() {
    clampNodeBounds(this);
    return this._ltx23ADOriginalOnDrawBackground?.apply(this, arguments);
  };

  if (!node._ltx23ADOriginalOnDrawForeground) {
    node._ltx23ADOriginalOnDrawForeground = node.onDrawForeground;
  }
  node.onDrawForeground = function() {
    clampNodeBounds(this);
    return this._ltx23ADOriginalOnDrawForeground?.apply(this, arguments);
  };

  // Apply initial clamp
  clampNodeBounds(node);
  node.setDirtyCanvas?.(true, true);
}

// Register extension
app.registerExtension({
  name: "CRT.LTX23AutoDownloadUI",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const registeredName = String(nodeData?.name || "");
    if (registeredName !== NODE_NAME && !NODE_ALIASES.has(registeredName)) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    const originalOnConfigure = nodeType.prototype.onConfigure;
    const originalOnRemoved = nodeType.prototype.onRemoved;

    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);

      applyNodeVisuals(this);

      if (!this.ltx23ADUI) {
        this.ltx23ADUI = new LTX23AutoDownloadUI(this);
      }

      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = originalOnConfigure?.apply(this, arguments);

      applyNodeVisuals(this);

      window.clearTimeout(this._ltx23ADRestoreTimer);
      this._ltx23ADRestoreTimer = window.setTimeout(() => {
        if (!this.ltx23ADUI) {
          this.ltx23ADUI = new LTX23AutoDownloadUI(this);
        } else {
          this.ltx23ADUI.rebuild();
        }
      }, 40);

      return result;
    };

    nodeType.prototype.onRemoved = function () {
      window.clearTimeout(this._ltx23ADRestoreTimer);
      this.ltx23ADUI?.destroy();
      this.ltx23ADUI = null;

      // Restore original methods
      if (this._ltx23ADOriginalSetSize) {
        this.setSize = this._ltx23ADOriginalSetSize;
        this._ltx23ADOriginalSetSize = null;
      }
      if (this._ltx23ADOriginalComputeSize) {
        this.computeSize = this._ltx23ADOriginalComputeSize;
        this._ltx23ADOriginalComputeSize = null;
      }
      if (this._ltx23ADOriginalOnDrawBackground) {
        this.onDrawBackground = this._ltx23ADOriginalOnDrawBackground;
        this._ltx23ADOriginalOnDrawBackground = null;
      }
      if (this._ltx23ADOriginalOnDrawForeground) {
        this.onDrawForeground = this._ltx23ADOriginalOnDrawForeground;
        this._ltx23ADOriginalOnDrawForeground = null;
      }
      this._ltx23ADCompactHeight = null;

      return originalOnRemoved?.apply(this, arguments);
    };
  },
});

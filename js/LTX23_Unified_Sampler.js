import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "CRT_LTX23UnifiedSampler";
const NODE_ALIASES = new Set(["LTX 2.3 Unified Sampler (CRT)", "CRT_LTX23UnifiedSampler"]);
const STYLE_ID = "crt-ltx23-unified-sampler-v8";
const MIN_WIDTH = 450;
const MIN_HEIGHT = 1;
const DEBUG = false;
const WORKFLOW_MODES = ["I2V", "T2V", "V2V"];
const V2V_MODES = ["Depth Control", "Outpaint"];

const MODE_FIELDS = {
  I2V: ["firstframe_strength"],
  T2V: ["aspect_ratio"],
  V2V: [],
};

const V2V_MODE_FIELDS = {
  "Depth Control": ["depth_megapixels", "firstframe_strength", "v2v_guide_strength"],
  "Outpaint": ["v2v_aspect_ratio"],
};

const ADVANCED_GROUPS = [
  {
    title: "Generation",
    fields: ["megapixels_target", "frame_count", "steps", "sampler_main", "sampler_refine"],
  },
  {
    title: "Audio",
    fields: ["frame_count_from_audio", "generated_audio_gain_db"],
  },
  {
    title: "Output",
    fields: ["vae_decode_tiled"],
  },
];

const FIELD_LABELS = {
  megapixels_target: "Megapixels",
  depth_megapixels: "Depth MP",
  frame_count: "Frames",
  steps: "Steps",
  aspect_ratio: "Aspect",
  firstframe_strength: "FirstFrame Strength",
  sampler_main: "Sampler",
  sampler_refine: "Refiner (LD)",
  v2v_mode: "V2V Mode",
  v2v_guide_strength: "Guide",
  v2v_aspect_ratio: "Aspect",
  frame_count_from_audio: "Audio Frame Count Override",
  vae_decode_tiled: "VAE Decode (Tiled)",
  generated_audio_gain_db: "Gain (dB)",
  depth_mouth_mask: "Mouth Mask",
  mouth_mask_expand: "Mouth Expand",
  mouth_mask_blur: "Mouth Blur",
};

function log(...args) {
  if (DEBUG) console.log("[CRT LTX23]", ...args);
}

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .crt-ltx23-root {
      width: 100%;
      height: 0;
      box-sizing: border-box;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
      user-select: none;
      -webkit-user-select: none;
      pointer-events: none;
    }

    .crt-ltx23-shell {
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
      --warning: #eab308;
      --warning-soft: rgba(234, 179, 8, 0.12);

      width: calc(100% - 12px);
      margin: 6px;
      padding: 10px;
      border-radius: 12px;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      color: var(--text-primary);
      box-sizing: border-box;
      pointer-events: auto;
      position: relative;
      overflow: visible;
    }
    
    .crt-ltx23-shell::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.3), transparent);
    }
    
    .crt-ltx23-shell * {
      box-sizing: border-box;
      pointer-events: auto;
      user-select: none;
      -webkit-user-select: none;
    }
    
    .crt-ltx23-title {
      font-size: 11px;
      font-weight: 500;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--text-tertiary);
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    
    .crt-ltx23-title::before {
      content: '';
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 8px var(--accent-glow);
    }
    
    .crt-ltx23-topbar {
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
      flex-wrap: nowrap;
    }
    
    .crt-ltx23-tabs {
      display: flex;
      gap: 2px;
      padding: 3px;
      background: var(--bg-base);
      border-radius: 8px;
      border: 1px solid var(--border-subtle);
      grid-column: 2;
      justify-self: center;
      justify-content: center;
      width: fit-content;
      transition: gap 220ms ease, padding 220ms ease;
    }

    .crt-ltx23-tabs.preview-hidden {
      gap: 2px;
    }
    
    .crt-ltx23-tab {
      height: 24px;
      padding: 0 10px;
      border-radius: 6px;
      border: none;
      background: transparent;
      color: var(--text-tertiary);
      font-size: 11px;
      font-weight: 500;
      cursor: pointer;
      transition: all 180ms ease;
      white-space: nowrap;
    }
    
    .crt-ltx23-tab:hover {
      color: var(--text-secondary);
    }
    
    .crt-ltx23-tab.mode-active {
      background: var(--bg-elevated);
      color: var(--accent);
      border: 1px solid rgba(139, 92, 246, 0.55);
      text-shadow: 0 0 10px rgba(139, 92, 246, 0.65);
      box-shadow: 0 0 0 1px rgba(139, 92, 246, 0.35), 0 0 16px rgba(139, 92, 246, 0.32);
      animation: mode-glow-pulse 2.6s ease-in-out infinite;
    }
    
    .crt-ltx23-tab.view-active:not(.mode-active) {
      background: var(--bg-elevated);
      color: var(--text-primary);
    }
    
    .crt-ltx23-tab.adv {
      color: var(--accent);
    }
    
    .crt-ltx23-tab.adv.view-active {
      color: var(--accent);
    }
    
    .crt-ltx23-tab.preview {
      color: var(--text-tertiary);
      overflow: hidden;
      max-width: 80px;
      transition:
        color 5s ease,
        text-shadow 5s ease,
        border-color 5s ease,
        max-width 220ms ease,
        opacity 220ms ease,
        transform 220ms ease,
        padding 220ms ease,
        margin 220ms ease,
        border-width 220ms ease;
    }

    .crt-ltx23-tab.preview.is-hidden {
      max-width: 0;
      opacity: 0;
      transform: translateX(8px);
      padding: 0;
      margin: 0;
      border-width: 0;
      pointer-events: none;
    }
    
    .crt-ltx23-tab.preview.view-active {
      color: var(--text-secondary);
    }

    .crt-ltx23-tab.preview.generating {
      color: var(--success);
      text-shadow: 0 0 10px rgba(34, 197, 94, 0.6);
      animation: preview-pulse 2.8s ease-in-out infinite;
    }
    
    .crt-ltx23-hq {
      width: 26px;
      height: 26px;
      padding: 0;
      border-radius: 999px;
      border: 1px solid var(--border-subtle);
      background: var(--bg-base);
      color: var(--text-tertiary);
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0;
      cursor: pointer;
      transition: all 120ms ease;
      text-transform: uppercase;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      grid-column: 3;
      justify-self: end;
    }
    
    .crt-ltx23-hq:hover {
      border-color: var(--border-default);
      color: var(--text-secondary);
    }
    
    .crt-ltx23-hq.on {
      border-color: var(--success);
      background: var(--success-soft);
      color: var(--success);
    }

    .crt-ltx23-preview-toggle {
      min-width: 108px;
      height: 24px;
      padding: 0 10px;
      border-radius: 6px;
      border: 1px solid var(--border-subtle);
      background: var(--bg-surface);
      color: var(--text-tertiary);
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      cursor: pointer;
      transition: all 120ms ease;
    }

    .crt-ltx23-preview-toggle:hover {
      border-color: var(--border-default);
      color: var(--text-secondary);
    }

    .crt-ltx23-preview-toggle.on {
      border-color: var(--success);
      background: var(--success-soft);
      color: var(--success);
    }
    
    .crt-ltx23-hint {
      margin-bottom: 8px;
      padding: 6px 10px;
      background: var(--bg-base);
      border-radius: 6px;
      color: var(--text-tertiary);
      font-size: 11px;
      line-height: 1.4;
    }
    
    .crt-ltx23-panel {
      display: none;
    }
    
    .crt-ltx23-panel.active {
      display: flex;
      flex-direction: column;
      gap: 6px;
      animation: fade-in 150ms ease;
    }

    @keyframes mode-glow-pulse {
      0%, 100% {
        box-shadow: 0 0 0 1px rgba(139, 92, 246, 0.3), 0 0 10px rgba(139, 92, 246, 0.2);
        text-shadow: 0 0 7px rgba(139, 92, 246, 0.45);
      }
      50% {
        box-shadow: 0 0 0 1px rgba(139, 92, 246, 0.7), 0 0 20px rgba(139, 92, 246, 0.45);
        text-shadow: 0 0 14px rgba(139, 92, 246, 0.9);
      }
    }

    @keyframes preview-pulse {
      0%, 100% {
        text-shadow: 0 0 6px rgba(34, 197, 94, 0.45);
      }
      50% {
        text-shadow: 0 0 14px rgba(34, 197, 94, 0.9);
      }
    }
    
    @keyframes fade-in {
      from { opacity: 0; transform: translateY(2px); }
      to { opacity: 1; transform: translateY(0); }
    }
    
    .crt-ltx23-section-title {
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-tertiary);
      padding-bottom: 4px;
      margin-top: 2px;
      border-bottom: 1px solid var(--border-subtle);
    }
    
    .crt-ltx23-field {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 5px 8px;
      background: var(--bg-base);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      transition: border-color 120ms ease;
    }
    
    .crt-ltx23-field:hover {
      border-color: var(--border-default);
    }
    
    .crt-ltx23-label {
      font-size: 10px;
      font-weight: 450;
      color: var(--text-secondary);
      white-space: nowrap;
      min-width: 80px;
    }
    
    .crt-ltx23-control {
      position: relative;
      min-height: 24px;
      flex: 1;
      display: flex;
      justify-content: flex-end;
    }
    
    .crt-ltx23-input {
      width: 100%;
      max-width: 180px;
      height: 24px;
      padding: 0 10px;
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      background: var(--bg-surface);
      color: var(--text-primary);
      font-size: 12px;
      font-family: inherit;
      transition: all 120ms ease;
      outline: none;
    }
    
    .crt-ltx23-input:hover {
      border-color: var(--border-default);
    }
    
    .crt-ltx23-input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px var(--accent-soft);
    }
    
    .crt-ltx23-select-wrap {
      position: relative;
      width: 100%;
      max-width: 180px;
      height: 24px;
    }
    
    .crt-ltx23-select {
      width: 100%;
      height: 100%;
      padding: 0 28px 0 10px;
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      background: var(--bg-surface);
      color: var(--text-primary);
      font-size: 12px;
      font-family: inherit;
      cursor: pointer;
      appearance: none;
      transition: all 120ms ease;
      outline: none;
    }
    
    .crt-ltx23-select:hover {
      border-color: var(--border-default);
    }
    
    .crt-ltx23-select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px var(--accent-soft);
    }
    
    .crt-ltx23-select-wrap::after {
      content: '';
      position: absolute;
      right: 8px;
      top: 50%;
      transform: translateY(-50%);
      width: 0;
      height: 0;
      border-left: 3px solid transparent;
      border-right: 3px solid transparent;
      border-top: 4px solid var(--text-tertiary);
      pointer-events: none;
    }
    
    .crt-ltx23-select option {
      background: var(--bg-surface);
      color: var(--text-primary);
    }
    
    .crt-ltx23-bool {
      display: flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      height: 24px;
    }
    
    .crt-ltx23-bool-text {
      font-size: 11px;
      color: var(--text-tertiary);
      transition: color 120ms ease;
    }
    
    .crt-ltx23-bool:hover .crt-ltx23-bool-text {
      color: var(--text-secondary);
    }
    
    .crt-ltx23-toggle {
      width: 30px;
      height: 14px;
      border-radius: 7px;
      background: var(--bg-elevated);
      border: 1px solid var(--border-subtle);
      position: relative;
      transition: all 120ms ease;
      cursor: pointer;
    }
    
    .crt-ltx23-toggle::before {
      content: "";
      position: absolute;
      top: 1px;
      left: 1px;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--text-tertiary);
      transition: all 120ms cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .crt-ltx23-toggle.on {
      background: var(--accent-soft);
      border-color: var(--accent);
    }
    
    .crt-ltx23-toggle.on::before {
      left: 17px;
      background: var(--accent);
      box-shadow: 0 0 6px var(--accent-glow);
    }
    
    .crt-ltx23-num {
      display: inline-flex;
      align-items: center;
      gap: 0;
      height: 24px;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      overflow: hidden;
      transition: all 120ms ease;
    }
    
    .crt-ltx23-num:hover {
      border-color: var(--border-default);
    }
    
    .crt-ltx23-num:focus-within {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px var(--accent-soft);
    }
    
    .crt-ltx23-num-btn {
      width: 24px;
      height: 100%;
      border: none;
      background: transparent;
      color: var(--text-tertiary);
      font-size: 14px;
      font-weight: 300;
      cursor: pointer;
      transition: all 120ms ease;
      display: flex;
      align-items: center;
      justify-content: center;
      user-select: none;
    }
    
    .crt-ltx23-num-btn:hover {
      background: var(--bg-hover);
      color: var(--text-secondary);
    }
    
    .crt-ltx23-num-btn:active {
      background: var(--bg-elevated);
    }
    
    .crt-ltx23-num-input {
      width: 56px;
      height: 100%;
      border: none;
      border-left: 1px solid var(--border-subtle);
      border-right: 1px solid var(--border-subtle);
      background: transparent;
      color: var(--text-primary);
      font-size: 12px;
      font-weight: 500;
      text-align: center;
      font-family: inherit;
      outline: none;
    }
    
    .crt-ltx23-num-input.editing {
      background: var(--accent-soft);
    }

    .crt-ltx23-input,
    .crt-ltx23-select,
    .crt-ltx23-num-input {
      user-select: none;
      -webkit-user-select: none;
    }
    
    .crt-ltx23-preview {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 12px;
      padding: 16px;
      min-height: 160px;
    }
    
    .crt-ltx23-preview-status {
      font-size: 11px;
      color: var(--text-tertiary);
      text-align: center;
      padding: 6px 14px;
      border-radius: 20px;
      background: var(--bg-base);
      border: 1px solid var(--border-subtle);
    }
    
    .crt-ltx23-preview-status.live {
      color: var(--success);
      background: var(--success-soft);
      border-color: rgba(34, 197, 94, 0.2);
    }
    
    .crt-ltx23-preview-image {
      display: none;
      max-width: 100%;
      max-height: 360px;
      border-radius: 8px;
      border: 1px solid var(--border-subtle);
    }
    
    .crt-ltx23-shell ::-webkit-scrollbar {
      width: 6px;
      height: 6px;
    }
    
    .crt-ltx23-shell ::-webkit-scrollbar-track {
      background: transparent;
    }
    
    .crt-ltx23-shell ::-webkit-scrollbar-thumb {
      background: var(--bg-elevated);
      border-radius: 3px;
    }
    
    .crt-ltx23-shell ::-webkit-scrollbar-thumb:hover {
      background: var(--bg-hover);
    }

    .crt-ltx23-v2v-mode-btn {
      padding: 4px 12px;
      border-radius: 6px;
      border: 1px solid var(--border-subtle);
      background: var(--bg-base);
      color: var(--text-tertiary);
      font-size: 11px;
      font-weight: 500;
      cursor: pointer;
      transition: all 150ms ease;
      white-space: nowrap;
    }

    .crt-ltx23-v2v-mode-btn:hover {
      background: var(--bg-elevated);
      border-color: var(--border-default);
      color: var(--text-secondary);
    }

    .crt-ltx23-v2v-mode-btn.active {
      background: var(--accent-soft);
      border-color: var(--accent);
      color: var(--accent);
      box-shadow: 0 0 8px var(--accent-glow);
    }
  `;
  document.head.appendChild(style);
}

function getWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name) || null;
}

function getComboOptions(widget) {
  const values = widget?.options?.values;
  if (Array.isArray(values)) return values;
  if (typeof values === "function") {
    try {
      const result = values();
      if (Array.isArray(result)) return result;
    } catch {
      // Ignore
    }
  }
  if (Array.isArray(widget?.options)) return widget.options;
  return [];
}

function isOneDecimalField(name) {
  return name === "megapixels_target";
}

function formatNumberForField(name, value) {
  if (value === null || value === undefined) return "";
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  if (isOneDecimalField(name)) return number.toFixed(1);
  return String(number);
}

function parseNumber(rawValue, widget, fieldName = "") {
  if (rawValue === "" || rawValue === null || rawValue === undefined) {
    return widget.value;
  }
  let number = Number(rawValue);
  if (!Number.isFinite(number)) return widget.value;
  
  const min = widget?.options?.min;
  const max = widget?.options?.max;
  if (typeof min === "number") number = Math.max(min, number);
  if (typeof max === "number") number = Math.min(max, number);
  
  const step = widget?.options?.step;
  const hasFractionalStep = typeof step === "number" && step > 0 && step < 1;
  if (isOneDecimalField(fieldName)) {
    number = Math.round(number * 10) / 10;
    return number;
  }
  const integerLike = !hasFractionalStep && (Number.isInteger(widget?.value) || Number.isInteger(step));
  if (integerLike) {
    number = Math.round(number);
  } else {
    number = Math.round(number * 10) / 10;
  }

  return number;
}

function fieldLabel(name) {
  return FIELD_LABELS[name] || name;
}

class LTX23UnifiedSamplerUI {
  constructor(node) {
    this.node = node;
    this.controls = new Map();
    this.panels = new Map();
    this.tabs = new Map();
    this.resizeTimer = null;
    this.previewUrl = null;
    this.previewHandler = null;
    this.previewMetaHandler = null;
    this.livePreviewButton = null;
    this.cleanups = [];
    this.generationActive = false;
    this.generationHandlers = null;

    const modeWidget = getWidget(node, "workflow_mode");
    this.mode = String(modeWidget?.value || "I2V");
    const saved = this.node.properties?.ltx23_view;
    this.activeView = ["ADV", "PREVIEW"].includes(saved) ? saved : this.mode;

    this.init();
  }

  init() {
    ensureStyles();
    this.hideNativeWidgets();
    this.createContainer();
    this.buildLayout();
    this.syncFromWidgets();
    this.refresh();
    this.bindPreview();
    this.bindGeneration();
    this.scheduleResize();
  }

  hideNativeWidgets() {
    for (const widget of this.node.widgets || []) {
      if (widget.name === "ltx23_ui") continue;
      widget.hidden = true;
      widget.computeSize = () => [0, -6];
    }
  }

  createContainer() {
    if (this.container) return;
    this.container = document.createElement("div");
    this.container.className = "crt-ltx23-root";
    this.domWidget = this.node.addDOMWidget("ltx23_ui", "div", this.container, {
      serialize: false,
    });
    if (this.domWidget) {
      this.domWidget.computeSize = () => [0, 0];
    }
    this.syncDOMHitbox();
  }

  syncDOMHitbox() {
    const parent = this.container?.parentElement;
    const elements = [
      this.domWidget?.element,
      this.container,
    ];
    if (parent && parent !== document.body && parent.children?.length === 1) {
      elements.push(parent);
    }
    for (const element of elements) {
      if (!element?.style) continue;
      element.style.pointerEvents = "none";
      element.style.overflow = "visible";
    }
    if (this.container?.style) {
      this.container.style.height = "0px";
    }
  }

  buildLayout() {
    this.container.innerHTML = "";
    this.controls.clear();
    this.panels.clear();
    this.tabs.clear();

    const shell = document.createElement("div");
    shell.className = "crt-ltx23-shell";
    this.shell = shell;

    const title = document.createElement("div");
    title.className = "crt-ltx23-title";
    title.textContent = "LTX Video 2.3";
    shell.appendChild(title);

    const topbar = document.createElement("div");
    topbar.className = "crt-ltx23-topbar";
    shell.appendChild(topbar);

    const tabsWrap = document.createElement("div");
    tabsWrap.className = "crt-ltx23-tabs";
    topbar.appendChild(tabsWrap);
    this.tabsWrap = tabsWrap;

    for (const mode of WORKFLOW_MODES) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "crt-ltx23-tab";
      button.textContent = mode;
      button.addEventListener("click", () => this.setMode(mode));
      tabsWrap.appendChild(button);
      this.tabs.set(mode, button);
    }

    const advButton = document.createElement("button");
    advButton.type = "button";
    advButton.className = "crt-ltx23-tab adv";
    advButton.textContent = "Advanced";
    advButton.addEventListener("click", () => {
      this.activeView = "ADV";
      this.persistView();
      this.refresh();
    });
    tabsWrap.appendChild(advButton);
    this.tabs.set("ADV", advButton);

    const previewButton = document.createElement("button");
    previewButton.type = "button";
    previewButton.className = "crt-ltx23-tab preview";
    previewButton.textContent = "Preview";
    previewButton.addEventListener("click", () => {
      this.activeView = "PREVIEW";
      this.persistView();
      this.ensureTAESDPreviewMethod();
      this.refresh();
    });
    tabsWrap.appendChild(previewButton);
    this.tabs.set("PREVIEW", previewButton);

    this.hqButton = document.createElement("button");
    this.hqButton.type = "button";
    this.hqButton.className = "crt-ltx23-hq";
    this.hqButton.addEventListener("click", () => this.toggleHQ());
    topbar.appendChild(this.hqButton);

    this.panelHost = document.createElement("div");
    shell.appendChild(this.panelHost);

    this.buildPanels();
    this.container.appendChild(shell);
    this.syncDOMHitbox();
  }

  buildPanels() {
    this.panelHost.innerHTML = "";
    this.panels.clear();

    for (const mode of WORKFLOW_MODES) {
      const panel = document.createElement("div");
      panel.className = "crt-ltx23-panel";

      const fields = MODE_FIELDS[mode] || [];
      for (const name of fields) {
        const row = this.buildFieldRow(name);
        if (row) panel.appendChild(row);
      }

      // Add V2V mode-specific fields
      if (mode === "V2V") {
        const v2vModeWidget = getWidget(this.node, "v2v_mode");
        const currentV2vMode = v2vModeWidget?.value || "Depth Control";

        // V2V Mode toggle row
        const modeRow = this.buildV2VModeRow();
        if (modeRow) panel.appendChild(modeRow);

        // Mode-specific fields
        const v2vFields = V2V_MODE_FIELDS[currentV2vMode] || [];
        for (const name of v2vFields) {
          const row = this.buildFieldRow(name);
          if (row) panel.appendChild(row);
        }

        // Depth Control extras
        if (currentV2vMode === "Depth Control") {
          const mmRow = this.buildFieldRow("depth_mouth_mask");
          if (mmRow) panel.appendChild(mmRow);
          const mouthOn = Boolean(getWidget(this.node, "depth_mouth_mask")?.value);
          if (mouthOn) {
            const expandRow = this.buildFieldRow("mouth_mask_expand");
            if (expandRow) panel.appendChild(expandRow);
            const blurRow = this.buildFieldRow("mouth_mask_blur");
            if (blurRow) panel.appendChild(blurRow);
          }
          panel.appendChild(this.buildDepthPreviewSection());
        }
      }

      this.panelHost.appendChild(panel);
      this.panels.set(mode, panel);
    }

    const advPanel = document.createElement("div");
    advPanel.className = "crt-ltx23-panel";
    for (const group of ADVANCED_GROUPS) {
      const title = document.createElement("div");
      title.className = "crt-ltx23-section-title";
      title.textContent = group.title;
      advPanel.appendChild(title);
      
      for (const name of group.fields) {
        const row = this.buildFieldRow(name);
        if (row) advPanel.appendChild(row);
      }
    }

    const previewToggleRow = this.buildLivePreviewRow();
    if (previewToggleRow) {
      const title = document.createElement("div");
      title.className = "crt-ltx23-section-title";
      title.textContent = "Preview";
      advPanel.appendChild(title);
      advPanel.appendChild(previewToggleRow);
    }

    this.panelHost.appendChild(advPanel);
    this.panels.set("ADV", advPanel);

    const previewPanel = document.createElement("div");
    previewPanel.className = "crt-ltx23-panel";
    previewPanel.appendChild(this.buildPreviewPanel());
    this.panelHost.appendChild(previewPanel);
    this.panels.set("PREVIEW", previewPanel);
  }

  buildPreviewPanel() {
    const wrap = document.createElement("div");
    wrap.className = "crt-ltx23-preview";
    
    this.previewStatus = document.createElement("div");
    this.previewStatus.className = "crt-ltx23-preview-status";
    wrap.appendChild(this.previewStatus);

    this.previewImage = document.createElement("img");
    this.previewImage.className = "crt-ltx23-preview-image";
    this.previewImage.alt = "Preview";
    wrap.appendChild(this.previewImage);
    
    this.updatePreviewStatus();
    return wrap;
  }

  buildFieldRow(name) {
    const widget = getWidget(this.node, name);
    if (!widget) return null;

    const row = document.createElement("div");
    row.className = "crt-ltx23-field";

    const label = document.createElement("div");
    label.className = "crt-ltx23-label";
    label.textContent = fieldLabel(name);
    row.appendChild(label);

    const controlWrap = document.createElement("div");
    controlWrap.className = "crt-ltx23-control";
    row.appendChild(controlWrap);

    const options = getComboOptions(widget);
    const hasNumericRange = Boolean(
      widget?.options &&
      (typeof widget.options.min === "number" ||
        typeof widget.options.max === "number" ||
        typeof widget.options.step === "number")
    );
    const isBool = typeof widget.value === "boolean" || widget.type === "toggle";
    const isCombo = options.length > 0 || widget.type === "combo";

    // Prefer number controls when a widget has numeric metadata,
    // even if stale workflow values temporarily look boolean.
    if (hasNumericRange) {
      controlWrap.appendChild(this.makeNumber(name, widget));
    } else if (isBool) {
      controlWrap.appendChild(this.makeBool(name, widget));
    } else if (isCombo) {
      const selectWrap = document.createElement("div");
      selectWrap.className = "crt-ltx23-select-wrap";
      selectWrap.appendChild(this.makeCombo(name, widget));
      controlWrap.appendChild(selectWrap);
    } else if (typeof widget.value === "number") {
      controlWrap.appendChild(this.makeNumber(name, widget));
    } else {
      controlWrap.appendChild(this.makeString(name, widget));
    }

    return row;
  }

  buildV2VModeRow() {
    const widget = getWidget(this.node, "v2v_mode");
    if (!widget) return null;

    const row = document.createElement("div");
    row.className = "crt-ltx23-field";

    const label = document.createElement("div");
    label.className = "crt-ltx23-label";
    label.textContent = "V2V Mode";
    row.appendChild(label);

    const controlWrap = document.createElement("div");
    controlWrap.className = "crt-ltx23-control";
    row.appendChild(controlWrap);

    const toggleWrap = document.createElement("div");
    toggleWrap.style.display = "flex";
    toggleWrap.style.gap = "4px";
    toggleWrap.style.alignItems = "center";
    toggleWrap.style.justifyContent = "center";

    const v2vModeWidget = getWidget(this.node, "v2v_mode");
    const currentMode = v2vModeWidget?.value || "Depth Control";

    for (const mode of V2V_MODES) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "crt-ltx23-v2v-mode-btn";
      btn.textContent = mode;
      btn.dataset.mode = mode;
      if (mode === currentMode) {
        btn.classList.add("active");
      }
      btn.addEventListener("click", () => {
        // Update widget value
        if (v2vModeWidget) {
          this.writeWidget("v2v_mode", v2vModeWidget, mode);
        }
        // Update UI
        toggleWrap.querySelectorAll(".crt-ltx23-v2v-mode-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        // Rebuild panels to show/hide mode-specific fields
        this.rebuildPanels();
      });
      toggleWrap.appendChild(btn);
    }

    controlWrap.appendChild(toggleWrap);
    row.appendChild(controlWrap);

    return row;
  }

  rebuildPanels() {
    const v2vModeWidget = getWidget(this.node, "v2v_mode");
    const currentV2vMode = v2vModeWidget?.value || "Depth Control";

    const v2vPanel = this.panels.get("V2V");
    if (!v2vPanel) return;

    // Clear V2V panel
    v2vPanel.innerHTML = "";

    // Rebuild V2V panel with current mode
    // Only v2v_mode is in base, v2v_guide_strength is in V2V_MODE_FIELDS
    const modeSpecificFields = V2V_MODE_FIELDS[currentV2vMode] || [];

    // V2V Mode toggle row
    const modeRow = this.buildV2VModeRow();
    if (modeRow) v2vPanel.appendChild(modeRow);

    // Mode-specific fields (includes v2v_guide_strength and either depth fields or aspect_ratio)
    for (const name of modeSpecificFields) {
      const row = this.buildFieldRow(name);
      if (row) v2vPanel.appendChild(row);
    }

    // Depth Control extras
    if (currentV2vMode === "Depth Control") {
      const mmRow = this.buildFieldRow("depth_mouth_mask");
      if (mmRow) v2vPanel.appendChild(mmRow);
      const mouthOn = Boolean(getWidget(this.node, "depth_mouth_mask")?.value);
      if (mouthOn) {
        const expandRow = this.buildFieldRow("mouth_mask_expand");
        if (expandRow) v2vPanel.appendChild(expandRow);
        const blurRow = this.buildFieldRow("mouth_mask_blur");
        if (blurRow) v2vPanel.appendChild(blurRow);
      }
      v2vPanel.appendChild(this.buildDepthPreviewSection());
    }
  }

  makeString(name, widget) {
    const input = document.createElement("input");
    input.type = "text";
    input.className = "crt-ltx23-input";
    input.value = String(widget.value ?? "");
    input.addEventListener("input", () => this.writeWidget(name, widget, input.value));
    this.controls.set(name, { kind: "string", element: input, widget });
    return input;
  }

  makeCombo(name, widget) {
    const select = document.createElement("select");
    select.className = "crt-ltx23-select";
    
    for (const optionValue of getComboOptions(widget)) {
      const option = document.createElement("option");
      option.value = String(optionValue);
      option.textContent = String(optionValue);
      select.appendChild(option);
    }
    
    select.value = String(widget.value ?? "");
    select.addEventListener("change", () => this.writeWidget(name, widget, select.value));
    this.controls.set(name, { kind: "combo", element: select, widget });
    return select;
  }

  makeBool(name, widget) {
    const root = document.createElement("label");
    root.className = "crt-ltx23-bool";

    const text = document.createElement("span");
    text.className = "crt-ltx23-bool-text";
    text.textContent = Boolean(widget.value) ? "Enabled" : "Disabled";

    const hidden = document.createElement("input");
    hidden.type = "checkbox";
    hidden.style.display = "none";
    hidden.checked = Boolean(widget.value);

    const toggle = document.createElement("span");
    toggle.className = "crt-ltx23-toggle";
    if (hidden.checked) toggle.classList.add("on");

    hidden.addEventListener("change", () => {
      const checked = hidden.checked;
      text.textContent = checked ? "Enabled" : "Disabled";
      toggle.classList.toggle("on", checked);
      this.writeWidget(name, widget, checked);
    });

    root.addEventListener("click", (event) => {
      event.preventDefault();
      hidden.checked = !hidden.checked;
      hidden.dispatchEvent(new Event("change"));
    });

    root.appendChild(text);
    root.appendChild(hidden);
    root.appendChild(toggle);
    
    this.controls.set(name, {
      kind: "bool",
      element: hidden,
      label: text,
      toggle,
      widget,
    });
    return root;
  }

  makeNumber(name, widget) {
    const wrap = document.createElement("div");
    wrap.className = "crt-ltx23-num";

    const minus = document.createElement("button");
    minus.type = "button";
    minus.className = "crt-ltx23-num-btn";
    minus.textContent = "−";

    const input = document.createElement("input");
    input.type = "text";
    input.className = "crt-ltx23-num-input";
    const defaultNumber = typeof widget?.options?.default === "number" ? widget.options.default : 0;
    const initialNumber = typeof widget.value === "number" ? widget.value : defaultNumber;
    input.value = formatNumberForField(name, initialNumber);

    const plus = document.createElement("button");
    plus.type = "button";
    plus.className = "crt-ltx23-num-btn";
    plus.textContent = "+";

    const min = typeof widget?.options?.min === "number" ? widget.options.min : null;
    const max = typeof widget?.options?.max === "number" ? widget.options.max : null;
    const configuredStep = typeof widget?.options?.step === "number" && widget.options.step > 0
      ? widget.options.step
      : null;
    const step = configuredStep && configuredStep < 1 ? 0.1 : (Number.isInteger(widget?.value) ? 1 : 0.1);

    const commit = (rawValue) => {
      const parsed = parseNumber(rawValue, widget, name);
      input.value = formatNumberForField(name, parsed);
      this.writeWidget(name, widget, parsed);
      return parsed;
    };

    const applyDelta = (deltaSteps) => {
      const current = typeof widget.value === "number" ? widget.value : Number(input.value ?? defaultNumber);
      const next = parseNumber(current + deltaSteps * step, {
        ...widget,
        value: current,
        options: { min, max, step },
      }, name);
      input.value = formatNumberForField(name, next);
      this.writeWidget(name, widget, next);
    };

    const startHold = (direction) => {
      applyDelta(direction);
      let repeatId = null;
      const kickoffId = window.setTimeout(() => {
        repeatId = window.setInterval(() => applyDelta(direction), 64);
      }, 240);
      const stop = () => {
        window.clearTimeout(kickoffId);
        if (repeatId !== null) window.clearInterval(repeatId);
        window.removeEventListener("mouseup", stop);
      };
      window.addEventListener("mouseup", stop, { once: true });
    };

    minus.addEventListener("mousedown", (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      startHold(-1);
    });

    plus.addEventListener("mousedown", (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      startHold(1);
    });

    let dragging = false;
    let suppressClick = false;
    
    input.addEventListener("mousedown", (event) => {
      if (event.button !== 0) return;
      const startX = event.clientX;
      const startValue = Number(widget.value ?? input.value ?? 0);
      
      const onMove = (moveEvent) => {
        const dx = moveEvent.clientX - startX;
        if (!dragging && Math.abs(dx) < 4) return;
        dragging = true;
        input.classList.remove("editing");
         const stepped = Math.round(dx / 24);
        const next = parseNumber(startValue + stepped * step, {
          ...widget,
          value: startValue,
          options: { min, max, step },
        }, name);
        input.value = formatNumberForField(name, next);
        this.writeWidget(name, widget, next);
        moveEvent.preventDefault();
      };
      
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        if (dragging) {
          suppressClick = true;
          dragging = false;
          input.blur();
        }
      };
      
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });

    input.addEventListener("click", (event) => {
      if (suppressClick) {
        suppressClick = false;
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      input.classList.add("editing");
    });

    input.addEventListener("blur", () => {
      input.classList.remove("editing");
      commit(input.value);
    });

    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        commit(input.value);
        input.blur();
      }
      if (event.key === "Escape") {
        input.value = formatNumberForField(name, widget.value ?? "");
        input.blur();
      }
    });

    wrap.appendChild(minus);
    wrap.appendChild(input);
    wrap.appendChild(plus);
    
    this.controls.set(name, {
      kind: "number",
      element: input,
      minus,
      plus,
      widget,
    });
    return wrap;
  }

  buildDepthPreviewSection() {
    const section = document.createElement("div");
    section.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:8px;margin-top:6px;width:100%;";

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "crt-ltx23-preview-toggle";
    btn.textContent = "Depth Preview: OFF";

    const img = document.createElement("img");
    img.style.cssText = "display:none;max-width:100%;border-radius:8px;border:1px solid var(--border-subtle);";
    img.alt = "Depth Preview";

    let shown = false;
    let blobUrl = null;

    btn.addEventListener("click", async () => {
      shown = !shown;
      if (shown) {
        try {
          const res = await fetch(`/crt/ltx23/depth_preview?t=${Date.now()}`);
          if (res.ok) {
            const blob = await res.blob();
            if (blobUrl) URL.revokeObjectURL(blobUrl);
            blobUrl = URL.createObjectURL(blob);
            img.src = blobUrl;
            img.style.display = "block";
            btn.textContent = "Depth Preview: ON";
            btn.classList.add("on");
          } else {
            shown = false;
            btn.textContent = "Depth Preview: No data yet";
            setTimeout(() => { btn.textContent = "Depth Preview: OFF"; }, 2000);
          }
        } catch {
          shown = false;
        }
      } else {
        img.style.display = "none";
        btn.textContent = "Depth Preview: OFF";
        btn.classList.remove("on");
      }
      this.scheduleResize();
    });

    section.appendChild(btn);
    section.appendChild(img);
    return section;
  }

  writeWidget(name, widget, value) {
    const previous = widget.value;
    widget.value = value;
    if (typeof widget.callback === "function") {
      try {
        widget.callback(value, app.canvas, this.node, undefined, widget);
      } catch (error) {
        log("widget callback failed", name, error);
      }
    }
    if (previous !== value) {
      this.node.setDirtyCanvas(true, true);
    }
    if (name === "workflow_mode") {
      this.mode = String(value || "I2V");
      if (!["ADV", "PREVIEW"].includes(this.activeView)) {
        this.activeView = this.mode;
      }
      this.persistView();
      this.refresh();
    }
    if (name === "v2v_mode") {
      this.refresh();
    }
    if (name === "depth_mouth_mask") {
      this.rebuildPanels();
      this.scheduleResize();
    }
    if (name === "hq") {
      this.updateHQButton();
      this.scheduleResize();
    }
  }

  setMode(mode) {
    const modeWidget = getWidget(this.node, "workflow_mode");
    this.mode = mode;
    if (modeWidget) {
      this.writeWidget("workflow_mode", modeWidget, mode);
    }
    this.activeView = mode;
    this.persistView();
    this.refresh();
  }

  toggleHQ() {
    const widget = getWidget(this.node, "hq");
    if (!widget) return;
    this.writeWidget("hq", widget, !Boolean(widget.value));
  }

  isLivePreviewEnabled() {
    const widget = getWidget(this.node, "live_preview");
    if (!widget) return false;
    return Boolean(widget.value);
  }

  toggleLivePreview() {
    const widget = getWidget(this.node, "live_preview");
    if (!widget) return;
    const next = !Boolean(widget.value);
    this.writeWidget("live_preview", widget, next);
    this.refresh();
  }

  updatePreviewTabVisibility() {
    const previewTab = this.tabs.get("PREVIEW");
    if (!previewTab) return;
    const enabled = this.isLivePreviewEnabled();
    previewTab.classList.toggle("is-hidden", !enabled);
    this.tabsWrap?.classList.toggle("preview-hidden", !enabled);
    if (!enabled && this.activeView === "PREVIEW") {
      this.activeView = this.mode;
      this.persistView();
    }
  }

  buildLivePreviewRow() {
    const widget = getWidget(this.node, "live_preview");
    if (!widget) return null;

    const row = document.createElement("div");
    row.className = "crt-ltx23-field";

    const label = document.createElement("div");
    label.className = "crt-ltx23-label";
    label.textContent = "Live Preview";
    row.appendChild(label);

    const controlWrap = document.createElement("div");
    controlWrap.className = "crt-ltx23-control";
    row.appendChild(controlWrap);

    const button = document.createElement("button");
    button.type = "button";
    button.className = "crt-ltx23-preview-toggle";
    button.addEventListener("click", () => this.toggleLivePreview());
    controlWrap.appendChild(button);

    this.livePreviewButton = button;
    this.updateLivePreviewButton();
    return row;
  }

  syncFromWidgets() {
    const modeWidget = getWidget(this.node, "workflow_mode");
    if (modeWidget) {
      this.mode = String(modeWidget.value || "I2V");
      if (!WORKFLOW_MODES.includes(this.mode)) this.mode = "I2V";
    }
    for (const [name, control] of this.controls.entries()) {
      const widget = getWidget(this.node, name);
      if (!widget) continue;
      control.widget = widget;
      if (control.kind === "bool") {
        const checked = Boolean(widget.value);
        control.element.checked = checked;
        control.label.textContent = checked ? "Enabled" : "Disabled";
        control.toggle.classList.toggle("on", checked);
      } else if (control.kind === "number") {
        const parsed = parseNumber(widget.value, widget, name);
        control.element.value = formatNumberForField(name, parsed);
      } else {
        control.element.value = String(widget.value ?? "");
      }
    }
  }

  persistView() {
    this.node.properties ??= {};
    if (["ADV", "PREVIEW"].includes(this.activeView)) {
      this.node.properties.ltx23_view = this.activeView;
    } else {
      delete this.node.properties.ltx23_view;
    }
  }

  updateHQButton() {
    const enabled = Boolean(getWidget(this.node, "hq")?.value);
    this.hqButton.classList.toggle("on", enabled);
    this.hqButton.textContent = "HQ";
  }

  refresh() {
    this.syncFromWidgets();
    if (![
      "ADV",
      "PREVIEW",
    ].includes(this.activeView)) {
      this.activeView = this.mode;
    }
    this.updatePreviewTabVisibility();
    for (const [key, button] of this.tabs.entries()) {
      button.classList.toggle("mode-active", WORKFLOW_MODES.includes(key) && key === this.mode);
      button.classList.toggle("view-active", key === this.activeView);
    }
    this.updatePreviewTabState();
    if (!this.panels.has(this.activeView)) {
      this.activeView = this.mode;
      this.persistView();
    }
    for (const [key, panel] of this.panels.entries()) {
      panel.classList.toggle("active", key === this.activeView);
    }
    this.updateHQButton();
    this.updateLivePreviewButton();
    if (this.activeView === "PREVIEW") {
      this.updatePreviewStatus();
    }
    this.scheduleResize();
  }

  rebuild() {
    this.hideNativeWidgets();
    this.refresh();
  }

  isTAESDActive() {
    try {
      const value =
        app.ui?.settings?.getSettingValue?.("Comfy.PreviewMethod") ??
        app.ui?.settings?.getSettingValue?.("preview_method") ??
        "";
      return String(value).toLowerCase().includes("taesd");
    } catch {
      return false;
    }
  }

  ensureTAESDPreviewMethod() {
    if (!this.isLivePreviewEnabled()) return;
    if (this.isTAESDActive()) return;
    try {
      const settings = app.ui?.settings;
      if (typeof settings?.setSettingValue === "function") {
        settings.setSettingValue("Comfy.PreviewMethod", "taesd");
        settings.setSettingValue("preview_method", "taesd");
      }
    } catch {
      // Ignore
    }
  }

  updatePreviewStatus() {
    if (!this.previewStatus) return;
    if (!this.isLivePreviewEnabled()) {
      this.previewStatus.classList.remove("live");
      this.previewStatus.textContent = "Live preview inactive";
      return;
    }
    this.ensureTAESDPreviewMethod();
    const live = this.isTAESDActive();
    this.previewStatus.classList.toggle("live", live);
    this.previewStatus.textContent = live
      ? "Live preview active"
      : "Enable TAESD in settings for live preview";
  }

  updateLivePreviewButton() {
    if (!this.livePreviewButton) return;
    const enabled = this.isLivePreviewEnabled();
    this.livePreviewButton.classList.toggle("on", enabled);
    this.livePreviewButton.textContent = enabled ? "Active" : "Inactive";
  }

  bindPreview() {
    const applyPreviewBlob = (blob) => {
      if (!(blob instanceof Blob) || !this.previewImage) return;
      const url = URL.createObjectURL(blob);
      if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
      this.previewUrl = url;
      this.previewImage.src = url;
      this.previewImage.style.display = "block";
    };

    const applyPreviewHTMLImage = (img) => {
      if (!this.previewImage || !(img instanceof HTMLImageElement) || !img.src) return;
      if (this.previewImage.src === img.src) return;
      this.previewImage.src = img.src;
      this.previewImage.style.display = "block";
    };

    // Only handle previews tagged with our node's id; ignore unrelated nodes.
    this.previewMetaHandler = (event) => {
      const payload = event.detail;
      if (!payload) return;
      const nodeId = payload.nodeId ?? payload.node_id ?? payload.displayNodeId;
      if (nodeId !== undefined && String(nodeId) !== String(this.node.id)) return;
      if (payload instanceof Blob) {
        applyPreviewBlob(payload);
      } else if (payload.blob instanceof Blob) {
        applyPreviewBlob(payload.blob);
      } else if (payload.image instanceof Blob) {
        applyPreviewBlob(payload.image);
      }
    };

    api.addEventListener("b_preview_with_metadata", this.previewMetaHandler);

    // Intercept node.imgs — ComfyUI may deliver previews by setting this property.
    // For live preview blobs (blob: URLs) we show in our tab but suppress the
    // standard node thumbnail by not storing them in _imgs.
    this._bindNodeImgs(applyPreviewHTMLImage);
  }

  _bindNodeImgs(onImage) {
    const node = this.node;
    if (node._ltx23ImgsIntercepted) return;
    node._ltx23ImgsIntercepted = true;
    let _imgs;
    Object.defineProperty(node, "imgs", {
      get() { return _imgs; },
      set(value) {
        // Blob-URL images are live previews — show in our tab but don't store
        // so LiteGraph doesn't draw them as a node thumbnail.
        if (Array.isArray(value) && value.length > 0) {
          const src = value[0]?.src ?? "";
          if (src.startsWith("blob:")) {
            onImage(value[0]);
            return; // do not update _imgs → no node thumbnail
          }
          // Final output images — store normally and show in our tab
          _imgs = value;
          onImage(value[0]);
          return;
        }
        _imgs = value;
      },
      configurable: true,
      enumerable: true,
    });
  }

  _unbindNodeImgs() {
    const node = this.node;
    if (!node?._ltx23ImgsIntercepted) return;
    const current = node.imgs;
    delete node._ltx23ImgsIntercepted;
    try {
      Object.defineProperty(node, "imgs", {
        value: current,
        writable: true,
        configurable: true,
        enumerable: true,
      });
    } catch {
      // Ignore
    }
  }

  bindGeneration() {
    if (this.generationHandlers) return;

    const onStart = () => this.setGenerating(true);
    const onExecuting = ({ detail }) => {
      if (detail === null) this.setGenerating(false);
    };
    const onStop = () => this.setGenerating(false);

    this.generationHandlers = { onStart, onExecuting, onStop };
    api.addEventListener("execution_start", onStart);
    api.addEventListener("executing", onExecuting);
    api.addEventListener("execution_error", onStop);
    api.addEventListener("execution_interrupted", onStop);
  }

  setGenerating(active) {
    const next = Boolean(active);
    if (this.generationActive === next) return;
    this.generationActive = next;
    this.updatePreviewTabState();
  }

  updatePreviewTabState() {
    const previewTab = this.tabs.get("PREVIEW");
    if (!previewTab) return;
    previewTab.classList.toggle("generating", this.generationActive);
  }

  scheduleResize() {
    window.clearTimeout(this.resizeTimer);
    this.resizeTimer = window.setTimeout(() => {
      this.resizeTimer = null;
      this.updateSize();
    }, 36);
  }

  updateSize() {
    this.syncDOMHitbox();
    const targetWidth = MIN_WIDTH;
    const targetHeight =
      typeof this.node._ltx23CompactHeight === "function"
        ? this.node._ltx23CompactHeight()
        : MIN_HEIGHT;
    if (this.node.size?.[0] !== targetWidth || this.node.size?.[1] !== targetHeight) {
      this.node.size = [targetWidth, targetHeight];
      this.node.setDirtyCanvas(true, true);
    }
  }

  destroy() {
    this._unbindNodeImgs();
    window.clearTimeout(this.resizeTimer);
    this.resizeTimer = null;
    for (const cleanup of this.cleanups) {
      try {
        cleanup();
      } catch {
        // Ignore
      }
    }
    this.cleanups = [];
    if (this.previewMetaHandler) {
      api.removeEventListener("b_preview_with_metadata", this.previewMetaHandler);
      this.previewMetaHandler = null;
    }
    if (this.generationHandlers) {
      api.removeEventListener("execution_start", this.generationHandlers.onStart);
      api.removeEventListener("executing", this.generationHandlers.onExecuting);
      api.removeEventListener("execution_error", this.generationHandlers.onStop);
      api.removeEventListener("execution_interrupted", this.generationHandlers.onStop);
      this.generationHandlers = null;
    }
    if (this.previewUrl) {
      URL.revokeObjectURL(this.previewUrl);
      this.previewUrl = null;
    }
    this.controls.clear();
    this.panels.clear();
    this.tabs.clear();
    this.container?.remove();
    this.domWidget = null;
    this.container = null;
    this.shell = null;
    this.previewImage = null;
    this.previewStatus = null;
  }
}

app.registerExtension({
  name: "CRT.LTX23UnifiedSamplerUI",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const registeredName = String(nodeData?.name || "");
    if (registeredName !== NODE_NAME && !NODE_ALIASES.has(registeredName)) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    const originalOnConfigure = nodeType.prototype.onConfigure;
    const originalOnRemoved = nodeType.prototype.onRemoved;

    const compactHeightForNode = (node) => {
      const probe = [0, 0];
      let maxVisibleY = Number(globalThis.LiteGraph?.NODE_TITLE_HEIGHT) || 30;

      const hasPosGetter = typeof node.getConnectionPos === "function";
      if (hasPosGetter) {
        if (Array.isArray(node.inputs)) {
          for (let i = 0; i < node.inputs.length; i++) {
            const slot = node.inputs[i];
            if (!slot || slot.hidden) continue;
            const pos = node.getConnectionPos(true, i, probe);
            const y = (Array.isArray(pos) ? pos[1] : probe[1]) - (node.pos?.[1] || 0);
            if (Number.isFinite(y)) maxVisibleY = Math.max(maxVisibleY, y);
          }
        }
        if (Array.isArray(node.outputs)) {
          for (let i = 0; i < node.outputs.length; i++) {
            const slot = node.outputs[i];
            if (!slot || slot.hidden) continue;
            const pos = node.getConnectionPos(false, i, probe);
            const y = (Array.isArray(pos) ? pos[1] : probe[1]) - (node.pos?.[1] || 0);
            if (Number.isFinite(y)) maxVisibleY = Math.max(maxVisibleY, y);
          }
        }
      }

      return Math.max(MIN_HEIGHT, Math.ceil(maxVisibleY + 8));
    };

    const clampNodeBounds = (node) => {
      const targetHeight = compactHeightForNode(node);
      if (node.size?.[0] !== MIN_WIDTH || node.size?.[1] !== targetHeight) {
        node.size = [MIN_WIDTH, targetHeight];
        node.setDirtyCanvas?.(true, false);
      }
      return targetHeight;
    };

    const applyNodeVisuals = (node) => {
      node.bgcolor = "transparent";
      node.color = "transparent";
      node.title = "";
      node.resizable = false;
      node.clip_area = false;
      node.flags ??= {};
      node.flags.clip_area = false;
      node._ltx23CompactHeight = () => clampNodeBounds(node);

      if (!node._ltx23OriginalSetSize) {
        node._ltx23OriginalSetSize = node.setSize;
      }
      if (!node._ltx23OriginalComputeSize) {
        node._ltx23OriginalComputeSize = node.computeSize;
      }
      if (!node._ltx23OriginalOnDrawForeground) {
        node._ltx23OriginalOnDrawForeground = node.onDrawForeground;
      }
      if (!node._ltx23OriginalOnDrawBackground) {
        node._ltx23OriginalOnDrawBackground = node.onDrawBackground;
      }

      node.computeSize = function (out) {
        const size = [MIN_WIDTH, compactHeightForNode(this)];
        if (out) {
          out[0] = size[0];
          out[1] = size[1];
          return out;
        }
        return size;
      };

      node.setSize = function () {
        const clamped = [MIN_WIDTH, compactHeightForNode(this)];
        this.size = clamped;
        return clamped;
      };

      node.onDrawBackground = function () {
        clampNodeBounds(this);
        return this._ltx23OriginalOnDrawBackground?.apply(this, arguments);
      };

      node.onDrawForeground = function () {
        clampNodeBounds(this);
        return this._ltx23OriginalOnDrawForeground?.apply(this, arguments);
      };

      clampNodeBounds(node);
      node.setDirtyCanvas?.(true, true);
    };

    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      const workflowMode = getWidget(this, "workflow_mode");
      const modeValue = String(workflowMode?.value || "");
      const validMode = WORKFLOW_MODES.includes(modeValue);
      if (workflowMode && !validMode) {
        workflowMode.value = "I2V";
      }
      this.properties ??= {};
      const viewValue = String(this.properties.ltx23_view || "");
      const validView = ["ADV", "PREVIEW"].includes(viewValue);
      if (!validView) {
        delete this.properties.ltx23_view;
      }
      applyNodeVisuals(this);
      if (!this.ltx23UI) {
        this.ltx23UI = new LTX23UnifiedSamplerUI(this);
      }
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = originalOnConfigure?.apply(this, arguments);
      applyNodeVisuals(this);
      window.clearTimeout(this._ltx23RestoreTimer);
      this._ltx23RestoreTimer = window.setTimeout(() => {
        if (!this.ltx23UI) {
          this.ltx23UI = new LTX23UnifiedSamplerUI(this);
        } else {
          this.ltx23UI.rebuild();
        }
      }, 40);
      return result;
    };

    nodeType.prototype.onRemoved = function () {
      window.clearTimeout(this._ltx23RestoreTimer);
      this._ltx23RestoreTimer = null;
      this.ltx23UI?.destroy();
      this.ltx23UI = null;
      if (this._ltx23OriginalSetSize) {
        this.setSize = this._ltx23OriginalSetSize;
        this._ltx23OriginalSetSize = null;
      }
      if (this._ltx23OriginalComputeSize) {
        this.computeSize = this._ltx23OriginalComputeSize;
        this._ltx23OriginalComputeSize = null;
      }
      if (this._ltx23OriginalOnDrawBackground) {
        this.onDrawBackground = this._ltx23OriginalOnDrawBackground;
        this._ltx23OriginalOnDrawBackground = null;
      }
      if (this._ltx23OriginalOnDrawForeground) {
        this.onDrawForeground = this._ltx23OriginalOnDrawForeground;
        this._ltx23OriginalOnDrawForeground = null;
      }
      this._ltx23CompactHeight = null;
      return originalOnRemoved?.apply(this, arguments);
    };
  },
});

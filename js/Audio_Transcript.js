import { app } from "../../scripts/app.js";

const NODE_NAME = "CRT_AudioTranscript";
const NODE_ALIASES = new Set(["Audio Transcript (CRT)", "CRT_AudioTranscript"]);
const STYLE_ID = "crt-audio-transcript-ui-v1";
const MIN_WIDTH = 360;
const MIN_HEIGHT = 1;

function getWidget(node, name) {
  return (node.widgets || []).find((widget) => widget?.name === name) || null;
}

function getComboOptions(widget) {
  const values = widget?.options?.values;
  if (Array.isArray(values)) return values;
  return [];
}

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;

  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .crt-at-root {
      width: 100%;
      height: 0;
      box-sizing: border-box;
      pointer-events: none;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    }

    .crt-at-shell {
      --bg-base: #09090b;
      --bg-surface: #111114;
      --bg-elevated: #17171c;
      --border-subtle: rgba(255, 255, 255, 0.05);
      --border-default: rgba(255, 255, 255, 0.09);
      --text-primary: #fafafa;
      --text-secondary: #a1a1aa;
      --text-tertiary: #6b7280;
      --accent: #38bdf8;
      --accent-soft: rgba(56, 189, 248, 0.12);
      --accent-strong: rgba(56, 189, 248, 0.55);
      --success: #22c55e;

      width: calc(100% - 12px);
      margin: 6px;
      padding: 12px;
      border-radius: 14px;
      border: 1px solid var(--border-subtle);
      background: linear-gradient(180deg, rgba(17,17,20,0.98), rgba(9,9,11,0.98));
      color: var(--text-primary);
      box-sizing: border-box;
      pointer-events: auto;
      position: relative;
    }

    .crt-at-shell::before {
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, var(--accent-strong), transparent);
    }

    .crt-at-shell * {
      box-sizing: border-box;
      pointer-events: auto;
    }

    .crt-at-title {
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--text-tertiary);
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .crt-at-title::before {
      content: "";
      width: 6px;
      height: 6px;
      border-radius: 999px;
      background: var(--accent);
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
    }

    .crt-at-hint {
      margin-bottom: 10px;
      padding: 8px 10px;
      border-radius: 8px;
      background: var(--bg-base);
      border: 1px solid var(--border-subtle);
      color: var(--text-secondary);
      font-size: 11px;
      line-height: 1.4;
    }

    .crt-at-section {
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--border-subtle);
    }

    .crt-at-section:first-of-type {
      margin-top: 0;
      padding-top: 0;
      border-top: none;
    }

    .crt-at-section-title {
      margin-bottom: 8px;
      color: var(--text-tertiary);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .crt-at-row {
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 10px;
      min-height: 30px;
      margin-bottom: 8px;
    }

    .crt-at-row:last-child {
      margin-bottom: 0;
    }

    .crt-at-label {
      color: var(--text-primary);
      font-size: 12px;
      line-height: 1.2;
    }

    .crt-at-subtle {
      color: var(--text-tertiary);
      font-size: 10px;
      margin-top: 3px;
    }

    .crt-at-toggle {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      user-select: none;
    }

    .crt-at-toggle-text {
      color: var(--text-secondary);
      font-size: 11px;
      min-width: 56px;
      text-align: right;
    }

    .crt-at-toggle-track {
      width: 34px;
      height: 20px;
      border-radius: 999px;
      background: var(--bg-elevated);
      border: 1px solid var(--border-default);
      position: relative;
      transition: all 160ms ease;
    }

    .crt-at-toggle-track::after {
      content: "";
      position: absolute;
      top: 2px;
      left: 2px;
      width: 14px;
      height: 14px;
      border-radius: 999px;
      background: #d4d4d8;
      transition: transform 160ms ease, background 160ms ease;
    }

    .crt-at-toggle-track.on {
      background: var(--accent-soft);
      border-color: var(--accent-strong);
    }

    .crt-at-toggle-track.on::after {
      transform: translateX(14px);
      background: var(--accent);
    }

    .crt-at-select {
      min-width: 150px;
      height: 30px;
      padding: 0 10px;
      border-radius: 8px;
      border: 1px solid var(--border-default);
      background: var(--bg-elevated);
      color: var(--text-primary);
      font-size: 12px;
      outline: none;
    }

    .crt-at-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 5px 8px;
      border-radius: 999px;
      background: rgba(34, 197, 94, 0.1);
      border: 1px solid rgba(34, 197, 94, 0.25);
      color: var(--success);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
  `;
  document.head.appendChild(style);
}

class AudioTranscriptUI {
  constructor(node) {
    this.node = node;
    this.init();
  }

  init() {
    ensureStyles();
    this.hideNativeWidgets();
    this.createContainer();
    this.render();
    this.scheduleResize();
  }

  hideNativeWidgets() {
    for (const widget of this.node.widgets || []) {
      if (widget.name === "crt_audio_transcript_ui") continue;
      widget.hidden = true;
      widget.computeSize = () => [0, -6];
    }
  }

  createContainer() {
    if (this.container) return;
    this.container = document.createElement("div");
    this.container.className = "crt-at-root";
    this.domWidget = this.node.addDOMWidget("crt_audio_transcript_ui", "div", this.container, {
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

  writeWidget(widget, value) {
    if (!widget) return;
    widget.value = value;
    widget.callback?.(value);
    this.node.setDirtyCanvas?.(true, true);
  }

  buildToggleRow(name, label, detail = "") {
    const widget = getWidget(this.node, name);
    if (!widget) return null;

    const row = document.createElement("div");
    row.className = "crt-at-row";

    const textWrap = document.createElement("div");
    const title = document.createElement("div");
    title.className = "crt-at-label";
    title.textContent = label;
    textWrap.appendChild(title);
    if (detail) {
      const subtle = document.createElement("div");
      subtle.className = "crt-at-subtle";
      subtle.textContent = detail;
      textWrap.appendChild(subtle);
    }
    row.appendChild(textWrap);

    const root = document.createElement("label");
    root.className = "crt-at-toggle";

    const state = document.createElement("span");
    state.className = "crt-at-toggle-text";
    const track = document.createElement("span");
    track.className = "crt-at-toggle-track";

    const sync = () => {
      const on = Boolean(widget.value);
      state.textContent = on ? "Enabled" : "Disabled";
      track.classList.toggle("on", on);
    };

    sync();
    root.addEventListener("click", (event) => {
      event.preventDefault();
      this.writeWidget(widget, !Boolean(widget.value));
      sync();
      this.render();
      this.scheduleResize();
    });

    root.appendChild(state);
    root.appendChild(track);
    row.appendChild(root);
    return row;
  }

  buildComboRow(name, label) {
    const widget = getWidget(this.node, name);
    if (!widget) return null;

    const row = document.createElement("div");
    row.className = "crt-at-row";

    const textWrap = document.createElement("div");
    const title = document.createElement("div");
    title.className = "crt-at-label";
    title.textContent = label;
    textWrap.appendChild(title);

    const subtle = document.createElement("div");
    subtle.className = "crt-at-subtle";
    subtle.textContent = "Used for llama translation and OmniVoice target language";
    textWrap.appendChild(subtle);
    row.appendChild(textWrap);

    const select = document.createElement("select");
    select.className = "crt-at-select";
    for (const optionValue of getComboOptions(widget)) {
      const option = document.createElement("option");
      option.value = String(optionValue);
      option.textContent = String(optionValue);
      select.appendChild(option);
    }
    select.value = String(widget.value ?? "");
    select.addEventListener("change", () => this.writeWidget(widget, select.value));
    row.appendChild(select);

    return row;
  }

  buildBadge(text) {
    const badge = document.createElement("div");
    badge.className = "crt-at-badge";
    badge.textContent = text;
    return badge;
  }

  render() {
    this.container.innerHTML = "";

    const shell = document.createElement("div");
    shell.className = "crt-at-shell";
    this.shell = shell;

    const title = document.createElement("div");
    title.className = "crt-at-title";
    title.textContent = "Audio Transcript";
    shell.appendChild(title);

    const hint = document.createElement("div");
    hint.className = "crt-at-hint";
    hint.textContent = "Whisper transcription is always run. Translation and OmniVoice are optional chained stages with automatic model offload after use.";
    shell.appendChild(hint);

    const transcriptSection = document.createElement("div");
    transcriptSection.className = "crt-at-section";
    const transcriptTitle = document.createElement("div");
    transcriptTitle.className = "crt-at-section-title";
    transcriptTitle.textContent = "Transcript";
    transcriptSection.appendChild(transcriptTitle);
    const isolateRow = this.buildToggleRow(
      "isolate_voice",
      "Voice Isolation",
      "Optional MelBandRoFormer preprocessing before Whisper and voice clone"
    );
    if (isolateRow) transcriptSection.appendChild(isolateRow);
    shell.appendChild(transcriptSection);

    const translationEnabled = Boolean(getWidget(this.node, "enable_translation")?.value);
    const omnivoiceWidget = getWidget(this.node, "enable_omnivoice");
    if (!translationEnabled && omnivoiceWidget?.value) {
      this.writeWidget(omnivoiceWidget, false);
    }
    const translationSection = document.createElement("div");
    translationSection.className = "crt-at-section";
    const translationTitle = document.createElement("div");
    translationTitle.className = "crt-at-section-title";
    translationTitle.textContent = "Translation";
    translationSection.appendChild(translationTitle);
    const translationToggle = this.buildToggleRow(
      "enable_translation",
      "Llama CPP",
      "Gliese Qwen3.5 4B Q8 translation with auto-download and cleanup"
    );
    if (translationToggle) translationSection.appendChild(translationToggle);
    if (translationEnabled) {
      const languageRow = this.buildComboRow("translation_language", "Target Language");
      if (languageRow) translationSection.appendChild(languageRow);
    }
    shell.appendChild(translationSection);

    if (translationEnabled) {
      const omnivoiceSection = document.createElement("div");
      omnivoiceSection.className = "crt-at-section";
      const omnivoiceTitle = document.createElement("div");
      omnivoiceTitle.className = "crt-at-section-title";
      omnivoiceTitle.textContent = "Voice Clone";
      omnivoiceSection.appendChild(omnivoiceTitle);
      const omnivoiceToggle = this.buildToggleRow(
        "enable_omnivoice",
        "OmniVoice",
        "Uses translated text and input audio as the voice reference"
      );
      if (omnivoiceToggle) omnivoiceSection.appendChild(omnivoiceToggle);
      const badgeRow = document.createElement("div");
      badgeRow.className = "crt-at-row";
      const badgeLabel = document.createElement("div");
      badgeLabel.className = "crt-at-label";
      badgeLabel.textContent = "Model Handling";
      badgeRow.appendChild(badgeLabel);
      badgeRow.appendChild(this.buildBadge("Auto Offload"));
      omnivoiceSection.appendChild(badgeRow);
      shell.appendChild(omnivoiceSection);
    }

    this.container.appendChild(shell);
    this.syncDOMHitbox();
  }

  scheduleResize() {
    window.requestAnimationFrame(() => {
      this.syncDOMHitbox();
      const targetHeight = typeof this.node._crtAtCompactHeight === "function"
        ? this.node._crtAtCompactHeight()
        : MIN_HEIGHT;
      this.node.size = [Math.max(MIN_WIDTH, this.node.size?.[0] || MIN_WIDTH), targetHeight];
      this.node.setDirtyCanvas?.(true, true);
    });
  }
}

app.registerExtension({
  name: "CRT.AudioTranscript",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const registeredName = String(nodeData?.name || "");
    if (registeredName !== NODE_NAME && !NODE_ALIASES.has(registeredName)) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    const originalOnConfigure = nodeType.prototype.onConfigure;
    const originalOnRemoved = nodeType.prototype.onRemoved;

    const compactHeightForNode = (node) => {
      const probe = [0, 0];
      let maxVisibleY = Number(globalThis.LiteGraph?.NODE_TITLE_HEIGHT) || 30;

      if (typeof node.getConnectionPos === "function") {
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
      node._crtAtCompactHeight = () => clampNodeBounds(node);

      if (!node._crtAtVisualsPatched) {
        node._crtAtVisualsPatched = true;
        node._crtAtOriginalSetSize = node.setSize;
        node._crtAtOriginalComputeSize = node.computeSize;
        node._crtAtOriginalOnDrawForeground = node.onDrawForeground;
        node._crtAtOriginalOnDrawBackground = node.onDrawBackground;

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
          return this._crtAtOriginalOnDrawBackground?.apply(this, arguments);
        };

        node.onDrawForeground = function () {
          clampNodeBounds(this);
          return this._crtAtOriginalOnDrawForeground?.apply(this, arguments);
        };
      }

      clampNodeBounds(node);
      node.setDirtyCanvas?.(true, true);
    };

    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      applyNodeVisuals(this);
      if (!this.__crtAudioTranscriptUI) {
        this.__crtAudioTranscriptUI = new AudioTranscriptUI(this);
      }
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = originalOnConfigure?.apply(this, arguments);
      applyNodeVisuals(this);
      window.clearTimeout(this._crtAtRestoreTimer);
      this._crtAtRestoreTimer = window.setTimeout(() => {
        if (!this.__crtAudioTranscriptUI) {
          this.__crtAudioTranscriptUI = new AudioTranscriptUI(this);
        } else {
          this.__crtAudioTranscriptUI.render();
          this.__crtAudioTranscriptUI.scheduleResize();
        }
      }, 40);
      return result;
    };

    nodeType.prototype.onRemoved = function () {
      window.clearTimeout(this._crtAtRestoreTimer);
      this._crtAtRestoreTimer = null;
      if (this._crtAtOriginalSetSize) {
        this.setSize = this._crtAtOriginalSetSize;
        this._crtAtOriginalSetSize = null;
      }
      if (this._crtAtOriginalComputeSize) {
        this.computeSize = this._crtAtOriginalComputeSize;
        this._crtAtOriginalComputeSize = null;
      }
      if (this._crtAtOriginalOnDrawBackground) {
        this.onDrawBackground = this._crtAtOriginalOnDrawBackground;
        this._crtAtOriginalOnDrawBackground = null;
      }
      if (this._crtAtOriginalOnDrawForeground) {
        this.onDrawForeground = this._crtAtOriginalOnDrawForeground;
        this._crtAtOriginalOnDrawForeground = null;
      }
      this._crtAtCompactHeight = null;
      this._crtAtVisualsPatched = false;
      this.__crtAudioTranscriptUI = null;
      return originalOnRemoved?.apply(this, arguments);
    };
  },
});

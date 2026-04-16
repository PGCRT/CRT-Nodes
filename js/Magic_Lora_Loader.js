/**
 * CRT Magic LoRA Loader
 *
 * Features:
 *   - Dry/Wet multiplier (0–200%) applied globally to all lora strengths
 *   - Global Σ strength indicator in the header
 *   - Global block weights (Flux double_blocks 0–18, single_blocks 0–37)
 *   - Per-LoRA block weight overrides
 *   - Preset saver / loader (server-side JSON)
 */

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { api } from "../../scripts/api.js";

const PLL_DEBUG = false;

function pllDebug(...args) {
  if (PLL_DEBUG) {
    console.log(...args);
  }
}

class LogSession {
  constructor(name) { this.name = name; }
  errorParts(message, ...args) { return ["error", [`${this.name} ${message}`, ...args]]; }
  newSession(name) { return new LogSession(`${this.name}${name}`); }
}
const logger = new LogSession("[crt]");
const rgthree = {
  logger,
  loadingApiJson: null,
  lastCanvasMouseEvent: null,
  newLogSession(name) { return logger.newSession(name); },
  invokeExtensionsAsync() { return Promise.resolve(); },
};
const _origMouseDown = LGraphCanvas.prototype.processMouseDown;
if (!_origMouseDown.__crtWrapped) {
  LGraphCanvas.prototype.processMouseDown = function (e) {
    rgthree.lastCanvasMouseEvent = e;
    return _origMouseDown.apply(this, arguments);
  };
  LGraphCanvas.prototype.processMouseDown.__crtWrapped = true;
}
function addConnectionLayoutSupport() { return; }
function moveArrayItem(array, itemOrFrom, to) {
  const from = typeof itemOrFrom === "number" ? itemOrFrom : array.indexOf(itemOrFrom);
  if (from === -1) return;
  const item = array[from];
  array.splice(from, 1);
  array.splice(Math.max(0, Math.min(to, array.length)), 0, item);
}
function removeArrayItem(array, item) { const i = array.indexOf(item); if (i > -1) array.splice(i, 1); }
function isLowQuality() { return app.canvas?.ds?.scale ? app.canvas.ds.scale < 0.5 : false; }
function fitString(ctx, str, maxWidth) {
  const ellipsis = "…";
  const eW = ctx.measureText(ellipsis).width;
  if (ctx.measureText(str).width <= maxWidth) return str;
  let out = str;
  while (out.length && ctx.measureText(out).width > maxWidth - eW) out = out.slice(0, -1);
  return out + ellipsis;
}
function drawRoundedRectangle(ctx, options) {
  const [x, y] = options.pos;
  const [w, h] = options.size;
  const r = options.borderRadius || h * 0.5;
  ctx.save();
  ctx.strokeStyle = options.colorStroke || LiteGraph.WIDGET_OUTLINE_COLOR;
  ctx.fillStyle = options.colorBackground || LiteGraph.WIDGET_BGCOLOR;
  ctx.beginPath();
  if (isLowQuality()) ctx.rect(x, y, w, h); else ctx.roundRect(x, y, w, h, [r]);
  ctx.fill();
  if (!isLowQuality()) ctx.stroke();
  ctx.restore();
}
function drawTogglePart(ctx, options) {
  const { posX, posY, height, value } = options;
  const lowQuality = isLowQuality();
  const toggleRadius = height * 0.36;
  const toggleBgWidth = height * 1.5;
  ctx.save();
  if (!lowQuality) {
    ctx.beginPath();
    ctx.roundRect(posX + 4, posY + 4, toggleBgWidth - 8, height - 8, [height * 0.5]);
    ctx.globalAlpha = app.canvas.editor_alpha * 0.25;
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.fill();
    ctx.globalAlpha = app.canvas.editor_alpha;
  }
  ctx.fillStyle = value === true ? "#89B" : "#888";
  const toggleX = lowQuality || value === false ? posX + height * 0.5 : value === true ? posX + height : posX + height * 0.75;
  ctx.beginPath();
  ctx.arc(toggleX, posY + height * 0.5, toggleRadius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  return [posX, toggleBgWidth];
}
function drawInfoIcon(ctx, x, y, size) {
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(x, y, size, size, [size * 0.1]);
  ctx.fillStyle = "#2f82ec";
  ctx.fill();
  ctx.strokeStyle = "#FFF";
  ctx.lineWidth = 2;
  const midX = x + size / 2;
  const s = size * 0.175;
  ctx.stroke(new Path2D(`M ${midX} ${y + size * 0.15} v 2 M ${midX - s} ${y + size * 0.45} h ${s} v ${size * 0.325} h ${s} h -${s * 2}`));
  ctx.restore();
}
drawNumberWidgetPart.WIDTH_TOTAL = 60;
function drawNumberWidgetPart(ctx, options) {
  const aw = 9, ah = 10, im = 3, nw = 32;
  const left = [0, 0], text = [0, 0], right = [0, 0];
  ctx.save();
  let x = options.posX;
  const { posY, height, value, textColor } = options;
  const midY = posY + height / 2;
  if (options.direction === -1) x = x - aw - im - nw - im - aw;
  ctx.fill(new Path2D(`M ${x} ${midY} l ${aw} ${ah / 2} l 0 -${ah} z`));
  left[0] = x; left[1] = aw; x += aw + im;
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  const old = ctx.fillStyle; if (textColor) ctx.fillStyle = textColor;
  ctx.fillText(fitString(ctx, value.toFixed(2), nw), x + nw / 2, midY);
  ctx.fillStyle = old;
  text[0] = x; text[1] = nw; x += nw + im;
  ctx.fill(new Path2D(`M ${x} ${midY - ah / 2} l ${aw} ${ah / 2} l -${aw} ${ah / 2} z`));
  right[0] = x; right[1] = aw;
  ctx.restore();
  return [left, text, right];
}
function drawWidgetButton(ctx, options, text = null, isMouseDownedAndOver = false) {
  const radius = isLowQuality() ? 0 : (options.borderRadius ?? 4);
  drawRoundedRectangle(ctx, {
    size: options.size,
    pos: [options.pos[0], options.pos[1] + (isMouseDownedAndOver ? 1 : 0)],
    borderRadius: radius,
    colorBackground: isMouseDownedAndOver ? "#444" : LiteGraph.WIDGET_BGCOLOR,
    colorStroke: "transparent",
  });
  if (!isLowQuality() && text) {
    ctx.save();
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.fillText(text, options.size[0] / 2, options.pos[1] + options.size[1] / 2 + (isMouseDownedAndOver ? 1 : 0));
    ctx.restore();
  }
}
class RgthreeBaseWidget {
  constructor(name) {
    this.name = name;
    this._value = null;
    this.type = "custom";
    this.last_y = 0;
    this.hitAreas = {};
    this.mouseDowned = null;
    this.downedHitAreasForMove = [];
    this.downedHitAreasForClick = [];
    this.isMouseDownedAndOver = false;
  }
  draw(ctx, node, width, posY, height) {
    this.last_y = posY;
  }
  clickWasWithinBounds(pos, bounds) {
    const xStart = bounds[0];
    const xEnd = xStart + (bounds.length > 2 ? bounds[2] : bounds[1]);
    const inX = pos[0] >= xStart && pos[0] <= xEnd;
    if (bounds.length === 2) return inX;
    return inX && pos[1] >= bounds[1] && pos[1] <= bounds[1] + bounds[3];
  }
  mouse(event, pos, node) {
    const t = event.type;
    const isDown = t === "pointerdown" || t === "mousedown";
    const isUp = t === "pointerup" || t === "mouseup";
    const isMove = t === "pointermove" || t === "mousemove";
    const local = [pos[0], pos[1] - (this.last_y ?? 0)];
    if (isDown) {
      this.mouseDowned = [...local];
      this.downedHitAreasForMove.length = 0;
      this.downedHitAreasForClick.length = 0;
      for (const part of Object.values(this.hitAreas)) {
        if (!this.clickWasWithinBounds(local, part.bounds)) continue;
        if (part.onMove) this.downedHitAreasForMove.push(part);
        if (part.onClick) this.downedHitAreasForClick.push(part);
        if (part.onDown) part.onDown.apply(this, [event, local, node, part]);
      }
      return this.onMouseDown(event, local, node) ?? true;
    }
    if (isUp) {
      const wasDowned = !!this.mouseDowned;
      let anyHandled = false;
      for (const part of this.downedHitAreasForClick) {
        if (this.clickWasWithinBounds(local, part.bounds)) {
          anyHandled = part.onClick?.apply(this, [event, local, node, part]) === true || anyHandled;
        }
      }
      if (wasDowned) {
        anyHandled = this.onMouseClick(event, local, node) === true || anyHandled;
      }
      this.downedHitAreasForClick.length = 0;
      this.cancelMouseDown();
      return this.onMouseUp(event, local, node) ?? anyHandled ?? true;
    }
    if (isMove) {
      for (const part of this.downedHitAreasForMove) part.onMove?.apply(this, [event, local, node, part]);
      return this.onMouseMove(event, local, node) ?? true;
    }
    return false;
  }
  onMouseDown() {}
  onMouseUp() {}
  onMouseMove() {}
  onMouseClick() {}
  cancelMouseDown() { this.mouseDowned = null; this.downedHitAreasForMove.length = 0; }
  serializeValue() { return this.value ?? this._value; }
}
class RgthreeDividerWidget extends RgthreeBaseWidget {
  constructor(options = {}) { super("divider"); this.options = options; this.value = { type: "divider" }; }
  draw(ctx, node, width, posY) { this.last_y = posY; }
}
class RgthreeBetterButtonWidget extends RgthreeBaseWidget {
  constructor(name, callback) { super(name); this.callback = callback; this.value = name; }
  draw(ctx, node, width, posY, height) { this.last_y = posY; drawWidgetButton(ctx, { size: [width - 30, height], pos: [15, posY] }, this.name, this.isMouseDownedAndOver); }
  onMouseClick(event, pos, node) { return this.callback(event, pos, node); }
}
const OVERRIDDEN_SERVER_NODES = new Map();
const _oldregisterNodeType = LiteGraph.registerNodeType;
LiteGraph.registerNodeType = async function (nodeId, baseClass) {
  const clazz = OVERRIDDEN_SERVER_NODES.get(baseClass) || baseClass;
  return _oldregisterNodeType.call(LiteGraph, nodeId, clazz);
};
class RgthreeBaseServerNode extends LGraphNode {
  constructor(title) {
    super(title);
    this.widgets = this.widgets || [];
    this.properties = this.properties || {};
    this.serialize_widgets = true;
    this.setupFromServerNodeData();
  }
  getWidgets() { return ComfyWidgets; }
  async setupFromServerNodeData() {
    const nodeData = this.constructor.nodeData;
    if (!nodeData) return;
    this.comfyClass = nodeData.name;
    let inputs = nodeData.input.required;
    if (nodeData.input.optional != undefined) inputs = Object.assign({}, inputs, nodeData.input.optional);
    const WIDGETS = this.getWidgets();
    for (const inputName in inputs) {
      const inputData = inputs[inputName];
      const type = inputData[0];
      if (inputData[1]?.forceInput) this.addInput(inputName, type);
      else if (Array.isArray(type)) WIDGETS.COMBO(this, inputName, inputData, app);
      else if (`${type}:${inputName}` in WIDGETS) WIDGETS[`${type}:${inputName}`](this, inputName, inputData, app);
      else if (type in WIDGETS) WIDGETS[type](this, inputName, inputData, app);
      else this.addInput(inputName, type);
    }
    for (const o in nodeData.output) {
      let output = nodeData.output[o];
      if (output instanceof Array) output = "COMBO";
      const outputName = nodeData.output_name[o] || output;
      this.addOutput(outputName, output);
    }
    this.size = this.computeSize();
  }
  removeWidget(widget) {
    if (typeof widget === "number") widget = this.widgets[widget];
    if (!widget) return;
    const i = this.widgets.indexOf(widget);
    if (i > -1) this.widgets.splice(i, 1);
    if (widget.onRemove) widget.onRemove();
  }
  defaultGetSlotMenuOptions() { return []; }
  static registerForOverride(comfyClass, nodeData, klass) {
    OVERRIDDEN_SERVER_NODES.set(comfyClass, klass);
    if (!klass.__registeredForOverride__) {
      klass.__registeredForOverride__ = true;
      klass.nodeType = comfyClass;
      klass.nodeData = nodeData;
      klass.onRegisteredForOverride(comfyClass, klass);
    }
  }
  static onRegisteredForOverride() {}
}
RgthreeBaseServerNode.nodeType = null;
RgthreeBaseServerNode.nodeData = null;
RgthreeBaseServerNode.__registeredForOverride__ = false;
const rgthreeApi = {
  lorasPromise: null,
  loras: [],
  getLoras(refresh = false) {
    if (!this.lorasPromise || refresh) {
      this.lorasPromise = api.fetchApi("/crt-pll/api/loras")
        .then(async (r) => {
          this.loras = await r.json();
          return this.loras;
        })
        .catch(() =>
          api.fetchApi("/object_info/LoraLoader")
            .then((r) => r.json())
            .then((info) => {
              const list = info?.LoraLoader?.input?.required?.lora_name?.[0] || [];
              this.loras = list.map((name) => ({ file: name }));
              return this.loras;
            })
            .catch(() => []),
        );
    }
    return this.lorasPromise;
  },
};
const LORA_INFO_SERVICE = {
  cache: new Map(),
  getInfo(loraName, force = false) {
    if (!force && this.cache.has(loraName)) return Promise.resolve(this.cache.get(loraName));
    return api.fetchApi(`/crt-pll/api/loras/info?file=${encodeURIComponent(loraName)}`)
      .then((r) => r.json())
      .then((d) => { this.cache.set(loraName, d); return d; })
      .catch(() => null);
  },
};
class RgthreeLoraInfoDialog extends EventTarget {
  constructor(loraName) { super(); this.loraName = loraName; }
  show() { LORA_INFO_SERVICE.getInfo(this.loraName, true).then((info) => { if (info) pllDebug(info); }); return this; }
}
function showLoraChooser(event, callback, currentValue, loras) {
  const existing = document.getElementById("pgc-lora-picker");
  if (existing) existing.remove();

  // Only show .safetensors files
  loras = loras.filter((n) => n.toLowerCase().endsWith(".safetensors"));

  const ev = rgthree.lastCanvasMouseEvent || event;
  const originX = ev?.clientX ?? 200;
  const originY = ev?.clientY ?? 200;

  const overlay = document.createElement("div");
  overlay.id = "pgc-lora-picker";

  const search = document.createElement("input");
  search.type = "text";
  search.className = "pgc-lora-picker-search";
  search.placeholder = "Search…";

  const ul = document.createElement("ul");
  ul.className = "pgc-lora-picker-list";

  function renderList(filter) {
    ul.innerHTML = "";
    const q = filter.toLowerCase();
    const filtered = q ? loras.filter((n) => n.toLowerCase().includes(q)) : loras;
    for (const name of filtered) {
      const li = document.createElement("li");
      li.className = "pgc-lora-picker-item" + (name === currentValue ? " active" : "");
      li.tabIndex = -1;
      const _esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      const _sep = name.search(/[/\\][^/\\]*$/);
      li.innerHTML = _sep === -1
        ? _esc(name)
        : `<span class="pgc-lora-picker-folder">${_esc(name.slice(0, _sep + 1))}</span>${_esc(name.slice(_sep + 1))}`;
      li.title = name;
      li.addEventListener("click", () => { overlay.remove(); callback(name); });
      ul.appendChild(li);
    }
    if (!filtered.length) {
      const li = document.createElement("li");
      li.className = "pgc-lora-picker-empty";
      li.textContent = "No results";
      ul.appendChild(li);
    }
  }

  search.addEventListener("input", () => renderList(search.value));
  search.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { overlay.remove(); return; }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      ul.querySelector(".pgc-lora-picker-item")?.focus();
    }
  });
  ul.addEventListener("keydown", (e) => {
    const items = [...ul.querySelectorAll(".pgc-lora-picker-item")];
    const idx = items.indexOf(document.activeElement);
    if (e.key === "ArrowDown") { e.preventDefault(); items[Math.min(idx + 1, items.length - 1)]?.focus(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); idx <= 0 ? search.focus() : items[idx - 1]?.focus(); }
    else if (e.key === "Enter" && idx >= 0) { items[idx].click(); }
    else if (e.key === "Escape") { overlay.remove(); }
  });

  overlay.appendChild(search);
  overlay.appendChild(ul);
  document.body.appendChild(overlay);

  const pickerMaxH = 340;
  const vw = window.innerWidth, vh = window.innerHeight;
  const _measureCtx = document.createElement("canvas").getContext("2d");
  _measureCtx.font = "12px sans-serif";
  const _longestPx = loras.reduce((max, n) => Math.max(max, _measureCtx.measureText(n).width), 0);
  const pickerW = Math.min(Math.max(300, Math.ceil(_longestPx) + 24), vw - 16);
  overlay.style.width = pickerW + "px";
  let left = originX, top = originY + 6;
  if (left + pickerW > vw - 8) left = vw - pickerW - 8;
  if (top + pickerMaxH > vh - 8) top = Math.max(8, originY - pickerMaxH - 6);
  overlay.style.left = left + "px";
  overlay.style.top = top + "px";

  renderList("");

  const onOutside = (e) => {
    if (!overlay.contains(e.target)) {
      overlay.remove();
      document.removeEventListener("pointerdown", onOutside, true);
      document.removeEventListener("mousedown", onOutside, true);
    }
  };
  setTimeout(() => {
    document.addEventListener("pointerdown", onOutside, true);
    document.addEventListener("mousedown", onOutside, true);
  }, 0);
  search.focus();
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const PGC_NODE_TYPE = "Magic LoRA Loader";

const PROP_LABEL_SHOW_STRENGTHS        = "Show Strengths";
const PROP_LABEL_SHOW_STRENGTHS_STATIC = `@${PROP_LABEL_SHOW_STRENGTHS}`;
const PROP_VALUE_SHOW_STRENGTHS_SINGLE   = "Single Strength";
const PROP_VALUE_SHOW_STRENGTHS_SEPARATE = "Separate Model & Clip";

const ACCENT      = "#7c6bff";
const ACCENT_DIM  = "rgba(124, 107, 255, 0.18)";
const ACCENT_MED  = "rgba(124, 107, 255, 0.45)";
const SIGMA_COLOR = "#a89aff";
const CAP_COLOR   = "#ff9f6b";
const CAP_DIM     = "rgba(255, 159, 107, 0.18)";
const CAP_MED     = "rgba(255, 159, 107, 0.42)";

const _AW = 9;
const _AH = 10;
const _IM = 3;
const _NW = 38;
const WET_WIDGET_TOTAL = _AW + _IM + _NW + _IM + _AW;

// Supported model types and their block architectures
const MODEL_TYPES = ["Flux2Klein", "LTX2.3", "ZImageTurbo", "WAN2.2"];
const BLOCK_CONFIGS = {
  "Flux2Klein": [
    { key: "double",      label: "Double Blocks (0–7)",        count: 8  },
    { key: "single",      label: "Single Blocks (0–23)",       count: 24 },
  ],
  "LTX2.3": [
    { key: "transformer", label: "Transformer Blocks (0–47)",  count: 48 },
  ],
  "ZImageTurbo": [
    { key: "layers",      label: "Layers (0–29)",              count: 30 },
  ],
  "WAN2.2": [
    { key: "blocks",      label: "Blocks (0–39)",              count: 40 },
  ],
};

// Keep legacy constants for any code that references them directly
const DOUBLE_BLOCK_COUNT = 8;
const SINGLE_BLOCK_COUNT = 24;

// ---------------------------------------------------------------------------
// Block weight helpers
// ---------------------------------------------------------------------------
function _defaultBlocks(modelType = "Flux2Klein") {
  const cfg = BLOCK_CONFIGS[modelType] || BLOCK_CONFIGS["Flux2Klein"];
  const result = {};
  for (const { key, count } of cfg) result[key] = Array(count).fill(1.0);
  return result;
}

function _isDefaultBlocks(b) {
  if (!b) return true;
  return Object.values(b).every((arr) => arr.every((v) => v === 1.0));
}

function _countNonDefault(b) {
  if (!b) return 0;
  let n = 0;
  for (const arr of Object.values(b)) for (const v of arr) if (v !== 1.0) n++;
  return n;
}

// ---------------------------------------------------------------------------
// CSS injection
// ---------------------------------------------------------------------------
function _injectStyles() {
  if (document.getElementById("pgc-pll-styles")) return;
  const s = document.createElement("style");
  s.id = "pgc-pll-styles";
  s.textContent = `
    /* ---- Preset widget ---- */
    .pgc-pll-presets {
      display: flex; align-items: center; gap: 5px;
      padding: 0 10px; box-sizing: border-box;
      width: 100%; height: 22px; background: transparent;
    }
    .pgc-pll-presets-label {
      font-size: 11px; color: rgba(255,255,255,0.38);
      white-space: nowrap; flex-shrink: 0;
      font-family: sans-serif; pointer-events: none;
    }
    .pgc-pll-select {
      flex: 1 1 0; min-width: 0;
      background: #000;
      border: 1px solid rgba(124,107,255,0.35);
      border-radius: 4px; color: #ccc; padding: 2px 5px;
      font-size: 11px; cursor: pointer; outline: none;
      font-family: sans-serif; transition: border-color 0.15s;
      appearance: none;
      -webkit-appearance: none;
    }
    .pgc-pll-select:hover, .pgc-pll-select:focus { border-color: #7c6bff; color: #fff; }
    .pgc-pll-input {
      flex: 1.1 1 0; min-width: 0;
      background: rgba(0,0,0,0.45);
      border: 1px solid rgba(124,107,255,0.25);
      border-radius: 4px; color: #fff; padding: 2px 6px;
      font-size: 11px; outline: none;
      font-family: sans-serif; transition: border-color 0.15s;
    }
    .pgc-pll-input::placeholder { color: rgba(255,255,255,0.28); }
    .pgc-pll-input:focus { border-color: #7c6bff; }
    .pgc-pll-btn {
      flex-shrink: 0;
      background: rgba(40,36,70,0.85);
      border: 1px solid rgba(124,107,255,0.3);
      border-radius: 4px; color: #bbb;
      padding: 2px 8px; font-size: 11px; cursor: pointer;
      font-family: sans-serif; white-space: nowrap;
      transition: background 0.12s, border-color 0.12s, color 0.12s;
      line-height: 1.5;
    }
    .pgc-pll-btn:hover { background: rgba(80,70,140,0.9); border-color: #7c6bff; color: #fff; }
    .pgc-pll-btn-save { color: #7fd88f; }
    .pgc-pll-btn-save:hover { color: #a2f0b0; }
    .pgc-pll-btn-del  { color: #d88f7f; padding: 2px 6px; }
    .pgc-pll-btn-del:hover { color: #f0a89a; }
    .pgc-pll-btn-normalize { color: #bbb; }
    .pgc-pll-btn-normalize.active { color: #7fd88f; border-color: rgba(127,216,143,0.5); background: rgba(60,120,70,0.35); }
    .pgc-pll-btn-normalize.active:hover { background: rgba(80,150,90,0.5); border-color: #7fd88f; }

    /* ---- Block editor panel ---- */
    .pgc-blocks-panel {
      position: fixed; z-index: 9999;
      background: #1a1825;
      border: 1px solid rgba(124,107,255,0.5);
      border-radius: 8px;
      width: 420px; max-height: 78vh;
      display: none; flex-direction: column;
      font-family: sans-serif; color: #ccc;
      box-shadow: 0 8px 32px rgba(0,0,0,0.7);
      user-select: none;
    }
    .pgc-blocks-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 9px 14px;
      border-bottom: 1px solid rgba(124,107,255,0.25);
      flex-shrink: 0;
    }
    .pgc-blocks-title { font-size: 13px; font-weight: 600; color: #a89aff; flex: 1 1 auto; }
    .pgc-blocks-reduction-wrap { display: flex; align-items: center; gap: 2px; margin-right: 6px; flex-shrink: 0; }
    .pgc-blocks-close {
      background: none; border: none; color: #888;
      cursor: pointer; font-size: 14px; padding: 2px 6px; border-radius: 3px;
    }
    .pgc-blocks-close:hover { color: #fff; background: rgba(255,255,255,0.1); }
    .pgc-blocks-lock {
      background: none; border: 1px solid transparent;
      color: #666; cursor: pointer; font-size: 13px;
      padding: 2px 6px; border-radius: 3px; line-height: 1;
      transition: color 0.12s, border-color 0.12s, background 0.12s;
    }
    .pgc-blocks-lock:hover { color: #aaa; background: rgba(255,255,255,0.08); }
    .pgc-blocks-lock.locked {
      color: #a89aff; border-color: rgba(168,154,255,0.4);
      background: rgba(124,107,255,0.12);
    }
    .pgc-blocks-content { overflow-y: auto; flex: 1; padding: 6px 14px 8px; }
    .pgc-blocks-section-heading {
      font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
      color: rgba(168,154,255,0.75); text-transform: uppercase;
      padding: 6px 0 3px;
      border-bottom: 1px solid rgba(124,107,255,0.18);
      margin-bottom: 3px;
      display: flex; align-items: center;
    }
    .pgc-blocks-section-heading-label { flex: 1; }
    .pgc-blocks-section-btns { display: flex; gap: 3px; }
    .pgc-blocks-section-btn {
      background: transparent;
      border: 1px solid rgba(124,107,255,0.2);
      color: rgba(255,255,255,0.35);
      font-size: 9px; cursor: pointer;
      padding: 1px 5px; border-radius: 3px;
      line-height: 1.4; font-family: sans-serif;
      text-transform: none; letter-spacing: 0;
    }
    .pgc-blocks-section-btn:hover { color: #fff; border-color: rgba(124,107,255,0.7); }
    .pgc-blocks-grid {
      display: grid; grid-template-columns: 1fr 1fr;
      gap: 1px 10px; margin-bottom: 6px;
    }
    .pgc-blocks-row {
      display: flex; align-items: center; gap: 4px; height: 20px;
    }
    .pgc-blocks-row-reset {
      background: transparent; border: none;
      color: rgba(124,107,255,0.3);
      font-size: 11px; cursor: pointer;
      padding: 0; line-height: 1; flex-shrink: 0; width: 14px; text-align: center;
      opacity: 0; transition: opacity 0.12s, color 0.12s;
    }
    .pgc-blocks-row:hover .pgc-blocks-row-reset { opacity: 1; }
    .pgc-blocks-row-reset:hover { color: #fff; }
    .pgc-blocks-lbl {
      font-size: 10px; color: #777; width: 14px;
      text-align: right; flex-shrink: 0;
    }
    .pgc-blocks-track {
      flex: 1; height: 7px;
      background: rgba(0,0,0,0.4);
      border: 1px solid rgba(124,107,255,0.18);
      border-radius: 4px; cursor: pointer;
      position: relative; overflow: hidden;
    }
    .pgc-blocks-fill {
      height: 100%; border-radius: 4px;
      background: #7c6bff; width: 50%;
      pointer-events: none;
    }
    .pgc-blocks-num {
      width: 36px;
      background: rgba(0,0,0,0.3);
      border: 1px solid rgba(124,107,255,0.18);
      border-radius: 3px; color: #ccc;
      font-size: 10px; padding: 1px 3px;
      text-align: center; outline: none; flex-shrink: 0;
    }
    .pgc-blocks-num:focus { border-color: #7c6bff; color: #fff; }
    .pgc-blocks-footer {
      display: flex; gap: 6px; flex-wrap: wrap;
      padding: 7px 14px;
      border-top: 1px solid rgba(124,107,255,0.2);
      flex-shrink: 0;
    }
    .pgc-blocks-fbtn {
      flex: 1;
      background: rgba(40,36,70,0.85);
      border: 1px solid rgba(124,107,255,0.3);
      border-radius: 4px; color: #bbb;
      padding: 4px 8px; font-size: 11px;
      cursor: pointer; font-family: sans-serif;
      white-space: nowrap;
    }
    .pgc-blocks-fbtn:hover { background: rgba(80,70,140,0.9); border-color: #7c6bff; color: #fff; }
    .pgc-blocks-fbtn-accent { color: #a89aff; border-color: rgba(168,154,255,0.35); }
    .pgc-blocks-fbtn-accent:hover { color: #fff; }
    .pgc-blocks-fbtn-intensity { color: #7fd88f; border-color: rgba(127,216,143,0.35); }
    .pgc-blocks-fbtn-intensity:hover { color: #fff; border-color: #7fd88f; }
    .pgc-blocks-fbtn-intensity:disabled { opacity: 0.5; cursor: default; }
    .pgc-blocks-fbtn-invert { color: #ff9f6b; border-color: rgba(255,159,107,0.35); }
    .pgc-blocks-fbtn-invert:hover { color: #fff; border-color: #ff9f6b; }

    /* ---- Stack Heatmap Panel ---- */
    .pgc-heat-panel {
      position: fixed; z-index: 9998;
      background: #1a1825;
      border: 1px solid rgba(124,107,255,0.5);
      border-radius: 8px; width: 368px;
      display: none; flex-direction: column;
      font-family: sans-serif; color: #ccc;
      box-shadow: 0 8px 32px rgba(0,0,0,0.7);
      user-select: none;
    }
    .pgc-heat-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 8px 14px;
      border-bottom: 1px solid rgba(124,107,255,0.25); flex-shrink: 0;
    }
    .pgc-heat-title { font-size: 12px; font-weight: 600; color: #a89aff; }
    .pgc-heat-close {
      background: none; border: none; color: #888;
      cursor: pointer; font-size: 14px; padding: 2px 6px; border-radius: 3px;
    }
    .pgc-heat-close:hover { color: #fff; background: rgba(255,255,255,0.1); }
    .pgc-heat-content { padding: 8px 12px 10px; }
    .pgc-heat-heading {
      font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
      color: rgba(168,154,255,0.75); text-transform: uppercase;
      padding: 4px 0 3px;
      border-bottom: 1px solid rgba(124,107,255,0.18); margin-bottom: 4px;
    }
    .pgc-heat-grid { display: grid; gap: 3px; margin-bottom: 8px; }
    .pgc-heat-db { grid-template-columns: repeat(8, 1fr); }
    .pgc-heat-sb { grid-template-columns: repeat(8, 1fr); }
    .pgc-heat-tb { grid-template-columns: repeat(8, 1fr); }
    .pgc-heat-lock {
      background: none; border: 1px solid transparent;
      color: #666; cursor: pointer; font-size: 13px;
      padding: 2px 6px; border-radius: 3px; line-height: 1;
      transition: color 0.12s, border-color 0.12s, background 0.12s; flex-shrink: 0;
    }
    .pgc-heat-lock:hover { color: #aaa; background: rgba(255,255,255,0.08); }
    .pgc-heat-lock.locked { color: #a89aff; border-color: rgba(168,154,255,0.4); background: rgba(124,107,255,0.12); }
    .pgc-panel-draggable { cursor: grab; user-select: none; }
    .pgc-panel-draggable:active { cursor: grabbing; }
    .pgc-pll-model-select-wrap {
      display: flex; align-items: center; gap: 5px;
      padding: 0 10px; box-sizing: border-box;
      width: 100%; height: 22px; background: transparent;
    }
    .pgc-heat-cell {
      border-radius: 4px; padding: 3px 2px;
      text-align: center; cursor: default;
      background: rgba(0,0,0,0.3);
    }
    .pgc-heat-idx  { display: block; font-size: 9px; color: rgba(255,255,255,0.35); line-height: 1; margin-bottom: 2px; }
    .pgc-heat-bar  { height: 22px; border-radius: 2px; background: rgba(255,255,255,0.07); margin-bottom: 2px; position: relative; overflow: hidden; }
    .pgc-heat-fill { position: absolute; bottom: 0; left: 0; right: 0; transition: height 0.18s, background-color 0.18s; }
    .pgc-heat-val  { display: block; font-size: 8px; color: rgba(255,255,255,0.55); line-height: 1; }
    .pgc-heat-note { font-size: 10px; color: rgba(255,255,255,0.3); text-align: center; padding: 4px 0 2px; font-style: italic; }

    /* ---- LoRA searchable picker ---- */
    #pgc-lora-picker {
      position: fixed; z-index: 10000;
      background: #1a1825;
      border: 1px solid rgba(124,107,255,0.5);
      border-radius: 7px;
      box-shadow: 0 6px 24px rgba(0,0,0,0.7);
      display: flex; flex-direction: column;
      overflow: hidden;
      font-family: sans-serif;
    }
    .pgc-lora-picker-search {
      padding: 7px 10px; border: none;
      border-bottom: 1px solid rgba(124,107,255,0.25);
      background: rgba(0,0,0,0.5); color: #fff;
      font-size: 12px; outline: none; flex-shrink: 0;
    }
    .pgc-lora-picker-search::placeholder { color: rgba(255,255,255,0.3); }
    .pgc-lora-picker-search:focus { border-bottom-color: #7c6bff; }
    .pgc-lora-picker-list {
      list-style: none; margin: 0; padding: 4px 0;
      overflow-y: auto; max-height: 320px;
    }
    .pgc-lora-picker-item {
      padding: 5px 10px; cursor: pointer;
      font-size: 12px; color: #ccc;
      white-space: nowrap;
      outline: none;
    }
    .pgc-lora-picker-item:hover,
    .pgc-lora-picker-item:focus { background: rgba(124,107,255,0.25); color: #fff; }
    .pgc-lora-picker-item.active { color: #a89aff; font-weight: 600; }
    .pgc-lora-picker-empty {
      padding: 8px 10px; font-size: 11px;
      color: rgba(255,255,255,0.3); font-style: italic;
    }
    .pgc-lora-picker-folder { color: #4cff6e; }
  `;
  document.head.appendChild(s);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function _arrowLeft(posX, midY) {
  return new Path2D(`M ${posX} ${midY} l ${_AW} ${_AH / 2} l 0 -${_AH} z`);
}
function _arrowRight(posX, midY) {
  return new Path2D(`M ${posX} ${midY - _AH / 2} l ${_AW} ${_AH / 2} l -${_AW} ${_AH / 2} z`);
}

function _drawBlocksIcon(ctx, x, y, size, color) {
  const cell = (size - 1) / 2;
  ctx.fillStyle = color;
  ctx.fillRect(x,          y,          cell, cell);
  ctx.fillRect(x + cell+1, y,          cell, cell);
  ctx.fillRect(x,          y + cell+1, cell, cell);
  ctx.fillRect(x + cell+1, y + cell+1, cell, cell);
}

// ---------------------------------------------------------------------------
// Shared drag-by-header helper
// ---------------------------------------------------------------------------
function _makeDraggable(el, handleEl) {
  let ox = 0, oy = 0, startX = 0, startY = 0, dragging = false;
  handleEl.classList.add("pgc-panel-draggable");
  handleEl.addEventListener("mousedown", (e) => {
    // Only drag on left-button, not on interactive children
    if (e.button !== 0 || e.target.tagName === "BUTTON" || e.target.tagName === "INPUT") return;
    dragging = true;
    startX = e.clientX; startY = e.clientY;
    const rect = el.getBoundingClientRect();
    ox = rect.left; oy = rect.top;
    e.preventDefault(); e.stopPropagation();
    const onMove = (me) => {
      if (!dragging) return;
      el.style.left = Math.max(0, ox + me.clientX - startX) + "px";
      el.style.top  = Math.max(0, oy + me.clientY - startY) + "px";
    };
    const onUp = () => { dragging = false; document.removeEventListener("mousemove", onMove); document.removeEventListener("mouseup", onUp); };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup",   onUp);
  });
}

// ---------------------------------------------------------------------------
// Block editor panel — singleton floating DOM widget
// ---------------------------------------------------------------------------
class BlockEditorPanel {
  constructor() {
    this._el              = null;
    this._titleEl         = null;
    this._useGlobBtn      = null;
    this._intensityBtn    = null;
    this._inputs          = {};
    this._blocks          = null;
    this._onChange        = null;
    this._onUseGlobal     = null;
    this._closeHandler    = null;
    this._locked          = false;
    this._lockBtn         = null;
    this._reductionInput  = null;
    this._onReductionChange = null;
    this._contentEl       = null;
    this._footerEl        = null;
    this._builtModelType  = null;
    this._modelType       = "Flux2Klein";
  }

  _ensureBuilt() {
    if (this._el) return;
    _injectStyles();

    const el = document.createElement("div");
    el.className = "pgc-blocks-panel";
    el.addEventListener("mousedown", (e) => e.stopPropagation());
    el.addEventListener("click",     (e) => e.stopPropagation());
    el.addEventListener("wheel",     (e) => e.stopPropagation());

    // Header
    const hdr = document.createElement("div");
    hdr.className = "pgc-blocks-header";
    this._titleEl = document.createElement("span");
    this._titleEl.className = "pgc-blocks-title";
    const reductionWrap = document.createElement("span");
    reductionWrap.className = "pgc-blocks-reduction-wrap";
    reductionWrap.style.display = "none";
    const reductionLbl = document.createElement("span");
    reductionLbl.textContent = "Reduction ×";
    reductionLbl.style.cssText = "font-size:10px;color:rgba(255,255,255,0.5);margin-right:4px;";
    this._reductionInput = document.createElement("input");
    this._reductionInput.type  = "number";
    this._reductionInput.min   = "0"; this._reductionInput.max = "5"; this._reductionInput.step = "0.1";
    this._reductionInput.value = "1.0";
    this._reductionInput.className = "pgc-pll-input";
    this._reductionInput.style.cssText = "width:50px;flex:0 0 50px;text-align:center;font-size:11px;padding:2px 4px;";
    this._reductionInput.title = "Per-LoRA automix reduction multiplier: 0 = no change, 1 = as computed, >1 = amplify reduction";
    this._reductionInput.addEventListener("change", () => {
      const v = Math.max(0, parseFloat(this._reductionInput.value) || 0);
      this._reductionInput.value = v.toFixed(1);
      this._onReductionChange?.(v);
    });
    this._reductionInput.addEventListener("click",    e => e.stopPropagation());
    this._reductionInput.addEventListener("mousedown", e => e.stopPropagation());
    reductionWrap.append(reductionLbl, this._reductionInput);

    this._lockBtn = document.createElement("button");
    this._lockBtn.className = "pgc-blocks-lock";
    this._lockBtn.title = "Lock open (won't close on outside click)";
    this._lockBtn.textContent = "🔓";
    this._lockBtn.addEventListener("click", () => {
      this._locked = !this._locked;
      this._lockBtn.textContent = this._locked ? "🔒" : "🔓";
      this._lockBtn.classList.toggle("locked", this._locked);
      if (!this._locked) {
        if (this._closeHandler) document.removeEventListener("mousedown", this._closeHandler);
        this._closeHandler = (e) => { if (!this._el.contains(e.target)) this.hide(); };
        setTimeout(() => document.addEventListener("mousedown", this._closeHandler), 0);
      } else {
        if (this._closeHandler) {
          document.removeEventListener("mousedown", this._closeHandler);
          this._closeHandler = null;
        }
      }
    });
    const closeBtn = document.createElement("button");
    closeBtn.className = "pgc-blocks-close";
    closeBtn.textContent = "✕";
    closeBtn.addEventListener("click", () => this.hide());
    hdr.append(this._titleEl, reductionWrap, this._lockBtn, closeBtn);
    this._reductionWrap = reductionWrap;

    // Footer (content inserted before this)
    const footer = document.createElement("div");
    footer.className = "pgc-blocks-footer";
    this._footerEl = footer;

    const resetBtn = document.createElement("button");
    resetBtn.className = "pgc-blocks-fbtn";
    resetBtn.textContent = "Reset All";
    resetBtn.addEventListener("click", () => this._setAll(1.0));

    const disableBtn = document.createElement("button");
    disableBtn.className = "pgc-blocks-fbtn";
    disableBtn.textContent = "Disable All";
    disableBtn.addEventListener("click", () => this._setAll(0.0));

    const invertBtn = document.createElement("button");
    invertBtn.className = "pgc-blocks-fbtn pgc-blocks-fbtn-invert";
    invertBtn.textContent = "🔄 Invert";
    invertBtn.title = "Invert all block strengths (1.0→0.0, 0.0→1.0)";
    invertBtn.addEventListener("click", () => this._invertAll());

    this._useGlobBtn = document.createElement("button");
    this._useGlobBtn.className = "pgc-blocks-fbtn pgc-blocks-fbtn-accent";
    this._useGlobBtn.textContent = "Clear Blocks";
    this._useGlobBtn.style.display = "none";
    this._useGlobBtn.addEventListener("click", () => { this._onUseGlobal?.(); this.hide(); });

    this._intensityBtn = document.createElement("button");
    this._intensityBtn.className = "pgc-blocks-fbtn pgc-blocks-fbtn-intensity";
    this._intensityBtn.textContent = "📊 From Intensity";
    this._intensityBtn.title = "Set block weights from this LoRA's per-block training intensity";
    this._intensityBtn.style.display = "none";
    this._intensityBtn.addEventListener("click", () => this._applyIntensity());

    footer.append(resetBtn, disableBtn, invertBtn, this._intensityBtn, this._useGlobBtn);
    el.append(hdr, footer);
    document.body.appendChild(el);
    _makeDraggable(el, hdr);
    this._el = el;
  }

  // Rebuild the content section when model type changes
  _ensureContentForModelType(modelType) {
    if (this._builtModelType === modelType && this._contentEl) return;
    if (this._contentEl) this._contentEl.remove();
    this._inputs = {};

    const content = document.createElement("div");
    content.className = "pgc-blocks-content";
    const cfg = BLOCK_CONFIGS[modelType] || BLOCK_CONFIGS["Flux2Klein"];
    for (const { key, label, count } of cfg) {
      content.append(this._buildSection(label, key, count));
    }
    this._footerEl.before(content);
    this._contentEl = content;
    this._builtModelType = modelType;
  }

  _buildSection(label, type, count) {
    const sec = document.createElement("div");
    const heading = document.createElement("div");
    heading.className = "pgc-blocks-section-heading";
    const headingLabel = document.createElement("span");
    headingLabel.className = "pgc-blocks-section-heading-label";
    headingLabel.textContent = label;
    const sectionBtns = document.createElement("div");
    sectionBtns.className = "pgc-blocks-section-btns";
    const secResetBtn = document.createElement("button");
    secResetBtn.className = "pgc-blocks-section-btn";
    secResetBtn.textContent = "↺ Reset";
    secResetBtn.title = "Reset this section to 1.0";
    secResetBtn.addEventListener("click", (e) => { e.stopPropagation(); this._setSection(type, 1.0); });
    const secZeroBtn = document.createElement("button");
    secZeroBtn.className = "pgc-blocks-section-btn";
    secZeroBtn.textContent = "✕ Zero";
    secZeroBtn.title = "Zero this section";
    secZeroBtn.addEventListener("click", (e) => { e.stopPropagation(); this._setSection(type, 0.0); });
    sectionBtns.append(secResetBtn, secZeroBtn);
    heading.append(headingLabel, sectionBtns);
    sec.appendChild(heading);

    const grid = document.createElement("div");
    grid.className = "pgc-blocks-grid";
    this._inputs[type] = [];

    for (let i = 0; i < count; i++) {
      const row   = document.createElement("div");  row.className = "pgc-blocks-row";
      const lbl   = document.createElement("span"); lbl.className = "pgc-blocks-lbl"; lbl.textContent = i;
      const track = document.createElement("div");  track.className = "pgc-blocks-track";
      const fill  = document.createElement("div");  fill.className = "pgc-blocks-fill";
      track.appendChild(fill);

      const num = document.createElement("input");
      num.type = "number"; num.className = "pgc-blocks-num";
      num.min = "0"; num.max = "2"; num.step = "0.05"; num.value = "1.00";
      num._fill = fill;
      num._type = type; num._idx = i;

      // Drag / click on track
      let dragging = false;
      track.addEventListener("mousedown", (e) => {
        dragging = true;
        const apply = (me) => {
          const r = track.getBoundingClientRect();
          const v = Math.round(Math.max(0, Math.min(2, (me.clientX - r.left) / r.width * 2)) * 100) / 100;
          this._setOne(type, i, v, num);
        };
        apply(e);
        const onMove = (me) => { if (dragging) apply(me); };
        const onUp   = ()   => { dragging = false; document.removeEventListener("mousemove", onMove); document.removeEventListener("mouseup", onUp); };
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup",   onUp);
        e.preventDefault(); e.stopPropagation();
      });

      num.addEventListener("change", () => {
        const v = Math.round(Math.max(0, Math.min(2, parseFloat(num.value) || 0)) * 100) / 100;
        this._setOne(type, i, v, num);
      });
      num.addEventListener("click", (e) => e.stopPropagation());
      num.addEventListener("mousedown", (e) => e.stopPropagation());

      const rowReset = document.createElement("button");
      rowReset.className = "pgc-blocks-row-reset";
      rowReset.textContent = "↺";
      rowReset.title = "Reset to 1.0";
      rowReset.addEventListener("click", (e) => { e.stopPropagation(); this._setOne(type, i, 1.0, num); });
      rowReset.addEventListener("mousedown", (e) => e.stopPropagation());

      this._inputs[type][i] = num;
      row.append(lbl, track, num, rowReset);
      grid.appendChild(row);
    }
    sec.appendChild(grid);
    return sec;
  }

  _setOne(type, i, val, num) {
    if (!this._blocks[type]) this._blocks[type] = [];
    this._blocks[type][i] = val;
    num.value = val.toFixed(2);
    const fill = num._fill;
    fill.style.width = (val / 2 * 100) + "%";
    fill.style.background = val === 0 ? "#444" : val < 1.0 ? "#a89aff" : "#7c6bff";
    this._onChange?.(this._blocks);
  }

  _setAll(val) {
    for (const [type, inputs] of Object.entries(this._inputs)) {
      inputs.forEach((num, i) => { if (num) this._setOne(type, i, val, num); });
    }
  }

  _setSection(type, val) {
    (this._inputs[type] || []).forEach((num, i) => { if (num) this._setOne(type, i, val, num); });
  }

  _invertAll() {
    for (const [type, inputs] of Object.entries(this._inputs)) {
      inputs.forEach((num, i) => {
        if (num) {
          const current = this._blocks[type]?.[i] ?? 1.0;
          const inverted = Math.round(Math.max(0, Math.min(1, 1.0 - current)) * 100) / 100;
          this._setOne(type, i, inverted, num);
        }
      });
    }
  }

  async _applyIntensity() {
    if (!this._loraName) { console.warn("[pgc-pll] _applyIntensity: no loraName"); return; }
    if (!this._inputs || Object.keys(this._inputs).length === 0) {
      console.warn("[pgc-pll] _applyIntensity: panel inputs not initialized");
      alert("Please reopen the block editor panel and try again.");
      return;
    }
    const origText = this._intensityBtn.textContent;
    this._intensityBtn.textContent = "⏳ Loading…";
    this._intensityBtn.disabled = true;
    try {
      const r = await fetch("/crt-pll/api/block_profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: this._loraName, model_type: this._modelType }),
      });
      if (!r.ok) {
        const text = await r.text();
        console.error(`[pgc-pll] block_profile HTTP ${r.status}:`, text);
        this._intensityBtn.textContent = `❌ Error ${r.status}`;
        setTimeout(() => { this._intensityBtn.textContent = origText; this._intensityBtn.disabled = false; }, 2500);
        return;
      }
      const data = await r.json();
      if (data.error) {
        console.error("[pgc-pll] block_profile server error:", data.error);
        this._intensityBtn.textContent = "❌ Server error";
        setTimeout(() => { this._intensityBtn.textContent = origText; this._intensityBtn.disabled = false; }, 2500);
        return;
      }
      const p = data.profile;
      pllDebug("[pgc-pll] block_profile response:", p);
      pllDebug("[pgc-pll] _inputs keys:", Object.keys(this._inputs));
      pllDebug("[pgc-pll] _blocks:", this._blocks);
      const reduction = Math.max(0, parseFloat(this._reductionInput?.value) || 1.0);
      const scale = (w) => Math.round(Math.max(0.0, 1.0 - reduction * (1.0 - w)) * 1000) / 1000;
      let applied = 0;
      for (const [type, inputs] of Object.entries(this._inputs)) {
        if (!p[type]) {
          console.warn(`[pgc-pll] No profile data for type '${type}'`);
          continue;
        }
        (p[type] || []).forEach((v, i) => {
          const val = scale(v);
          const num = inputs[i];
          if (num) { this._setOne(type, i, val, num); applied++; }
        });
      }
      pllDebug(`[pgc-pll] block_profile applied to ${applied} blocks`);
      if (applied === 0) console.warn("[pgc-pll] block_profile: no blocks updated — check _inputs state");
    } catch (e) {
      console.error("[pgc-pll] block_profile fetch exception:", e);
      this._intensityBtn.textContent = "❌ Failed";
      setTimeout(() => { this._intensityBtn.textContent = origText; this._intensityBtn.disabled = false; }, 2500);
      return;
    }
    this._intensityBtn.textContent = origText;
    this._intensityBtn.disabled = false;
  }

  _refreshInputs() {
    for (const [type, inputs] of Object.entries(this._inputs)) {
      const arr = this._blocks[type] || [];
      inputs.forEach((num, i) => {
        if (!num) return;
        const val = arr[i] ?? 1.0;
        num.value = val.toFixed(2);
        num._fill.style.width = (val / 2 * 100) + "%";
        num._fill.style.background = val === 0 ? "#444" : val < 1.0 ? "#a89aff" : "#7c6bff";
      });
    }
  }

  show({ blocks, title, isPerLora = false, reduction = 1.0, loraName = null, modelType = "Flux2Klein", onChange, onReductionChange, onUseGlobal, clientX, clientY }) {
    this._ensureBuilt();
    this._ensureContentForModelType(modelType);
    this._modelType    = modelType;
    this._blocks       = JSON.parse(JSON.stringify(blocks));
    this._onChange     = onChange;
    this._onUseGlobal  = onUseGlobal;
    this._onReductionChange = onReductionChange ?? null;
    this._loraName     = loraName;
    this._titleEl.textContent = title;
    this._useGlobBtn.style.display = isPerLora ? "" : "none";
    this._intensityBtn.style.display = (isPerLora && loraName) ? "" : "none";
    this._reductionWrap.style.display = isPerLora ? "flex" : "none";
    if (this._reductionInput) this._reductionInput.value = (reduction ?? 1.0).toFixed(1);
    this._refreshInputs();

    this._el.style.display = "flex";

    // Position near cursor, keeping panel inside viewport
    const pw = 420, ph = Math.min(window.innerHeight * 0.78, 600);
    let left = (clientX ?? window.innerWidth  / 2) + 12;
    let top  = (clientY ?? window.innerHeight / 2) - 20;
    if (left + pw > window.innerWidth  - 8) left = Math.max(8, window.innerWidth  - pw - 8);
    if (top  + ph > window.innerHeight - 8) top  = Math.max(8, window.innerHeight - ph - 8);
    this._el.style.left = left + "px";
    this._el.style.top  = top  + "px";
    this._el.style.transform = "";

    // Close on outside click (only when not locked)
    if (this._closeHandler) document.removeEventListener("mousedown", this._closeHandler);
    this._closeHandler = null;
    if (!this._locked) {
      this._closeHandler = (e) => { if (!this._el.contains(e.target)) this.hide(); };
      setTimeout(() => document.addEventListener("mousedown", this._closeHandler), 0);
    }
  }

  hide() {
    if (this._el) this._el.style.display = "none";
    if (this._closeHandler) {
      document.removeEventListener("mousedown", this._closeHandler);
      this._closeHandler = null;
    }
  }
}

const _blockEditorPanel = new BlockEditorPanel();

// ---------------------------------------------------------------------------
// Stack Heatmap Panel — real-time per-block load visualisation
// ---------------------------------------------------------------------------
function _heatColor(t) {
  // t in [0,1]: dark-blue → cyan → yellow → red
  const h = Math.round(240 - 240 * Math.min(1, t));
  const s = 75;
  const l = Math.round(12 + t * 50);
  return `hsl(${h},${s}%,${l}%)`;
}

function _computeStackProfile(node, profiles) {
  const modelType = node._pllModelType || "Flux2Klein";
  const cfg = BLOCK_CONFIGS[modelType] || BLOCK_CONFIGS["Flux2Klein"];
  const result = {};
  for (const { key, count } of cfg) result[key] = Array(count).fill(0);
  const wetWidget = (node.widgets || []).find(w => w.value?.__pgc_wet === true);
  const wet = Math.max(0, Math.min(2, wetWidget?.value?.value ?? 1.0));
  const globalBlocks = node._pllGlobalBlocksWidget?._blocks ?? {};
  const loraWidgets = (node.widgets || []).filter(w => w.name?.startsWith("lora_") && w.value?.lora);
  loraWidgets.forEach((w, i) => {
    if (!w.value?.on) return;
    const profile = profiles?.[i];  // null = automix not run yet; use uniform 1.0
    const strength = Math.abs(w.value?.strength ?? 1.0) * wet;
    const blocks = w.value?.blocks;
    for (const { key, count } of cfg) {
      for (let b = 0; b < count; b++) {
        const gb = globalBlocks[key]?.[b] ?? 1.0;
        result[key][b] += strength * (blocks?.[key]?.[b] ?? 1.0) * (profile ? (profile[key]?.[b] ?? 1.0) : 1.0) * gb;
      }
    }
  });
  return result;
}

class PgcStackHeatmapPanel {
  constructor() {
    this._el             = null;
    this._contentEl      = null;
    this._noteEl         = null;
    this._node           = null;
    this._profiles       = null;
    this._cells          = {};
    this._rafId          = null;
    this._builtModelType = null;
    this._locked         = false;
    this._lockBtn        = null;
    this._closeHandler   = null;
  }

  _ensureBuilt() {
    if (this._el) return;
    _injectStyles();
    const el = document.createElement("div");
    el.className = "pgc-heat-panel";
    el.addEventListener("mousedown", e => e.stopPropagation());
    el.addEventListener("click",     e => e.stopPropagation());
    el.addEventListener("wheel",     e => e.stopPropagation());

    const hdr = document.createElement("div"); hdr.className = "pgc-heat-header";
    const title = document.createElement("span"); title.className = "pgc-heat-title"; title.textContent = "Stack Block Load";

    this._lockBtn = document.createElement("button");
    this._lockBtn.className = "pgc-heat-lock";
    this._lockBtn.title = "Lock open (won't close on outside click)";
    this._lockBtn.textContent = "🔓";
    this._lockBtn.addEventListener("click", () => {
      this._locked = !this._locked;
      this._lockBtn.textContent = this._locked ? "🔒" : "🔓";
      this._lockBtn.classList.toggle("locked", this._locked);
      if (!this._locked) {
        if (this._closeHandler) document.removeEventListener("mousedown", this._closeHandler);
        this._closeHandler = (e) => { if (!this._el.contains(e.target)) this.hide(); };
        setTimeout(() => document.addEventListener("mousedown", this._closeHandler), 0);
      } else {
        if (this._closeHandler) { document.removeEventListener("mousedown", this._closeHandler); this._closeHandler = null; }
      }
    });

    const closeBtn = document.createElement("button"); closeBtn.className = "pgc-heat-close"; closeBtn.textContent = "✕";
    closeBtn.addEventListener("click", () => this.hide());
    hdr.append(title, this._lockBtn, closeBtn);

    const note = document.createElement("div"); note.className = "pgc-heat-note";
    note.textContent = "Updates live — reflects enabled LoRAs, strengths & block weights";
    this._noteEl = note;

    el.append(hdr);
    document.body.appendChild(el);
    _makeDraggable(el, hdr);
    this._el = el;
  }

  _ensureContentForModelType(modelType) {
    if (this._builtModelType === modelType && this._contentEl) return;
    if (this._contentEl) this._contentEl.remove();
    this._cells = {};

    const content = document.createElement("div"); content.className = "pgc-heat-content";
    const cfg = BLOCK_CONFIGS[modelType] || BLOCK_CONFIGS["Flux2Klein"];
    for (const { key, label, count } of cfg) {
      content.append(this._buildSection(label, key, count));
    }
    content.appendChild(this._noteEl);
    this._el.appendChild(content);
    this._contentEl = content;
    this._builtModelType = modelType;
  }

  _buildSection(label, type, count) {
    const cssKey = type === "double" ? "db" : type === "single" ? "sb" : "tb";
    const sec = document.createElement("div");
    const h = document.createElement("div"); h.className = "pgc-heat-heading"; h.textContent = label;
    sec.appendChild(h);
    const grid = document.createElement("div");
    grid.className = "pgc-heat-grid pgc-heat-" + cssKey;
    this._cells[type] = [];
    for (let i = 0; i < count; i++) {
      const cell = document.createElement("div"); cell.className = "pgc-heat-cell";
      const idx  = document.createElement("span"); idx.className  = "pgc-heat-idx";  idx.textContent = i;
      const bar  = document.createElement("div");  bar.className  = "pgc-heat-bar";
      const fill = document.createElement("div");  fill.className = "pgc-heat-fill"; fill.style.height = "0%";
      bar.appendChild(fill);
      const val = document.createElement("span"); val.className = "pgc-heat-val"; val.textContent = "0.00";
      cell.append(idx, bar, val);
      grid.appendChild(cell);
      this._cells[type][i] = { cell, fill, val };
    }
    sec.appendChild(grid);
    return sec;
  }

  refresh() {
    if (!this._el || this._el.style.display === "none") return;
    if (!this._node) return;

    const stack = _computeStackProfile(this._node, this._profiles);
    const allVals = Object.values(stack).flat();
    const maxVal = Math.max(...allVals, 1e-6);

    for (const [type, vals] of Object.entries(stack)) {
      vals.forEach((v, i) => {
        const t = v / maxVal;
        const c = this._cells[type]?.[i];
        if (!c) return;
        c.cell.style.background = _heatColor(t * 0.45);
        c.fill.style.height     = (t * 100).toFixed(1) + "%";
        c.fill.style.backgroundColor = _heatColor(t);
        c.val.textContent = v.toFixed(2);
      });
    }
  }

  _startPolling() {
    if (this._rafId) return;
    const tick = () => {
      if (!this._el || this._el.style.display === "none") { this._rafId = null; return; }
      this.refresh();
      this._rafId = requestAnimationFrame(tick);
    };
    this._rafId = requestAnimationFrame(tick);
  }

  show(node, profiles, { clientX, clientY } = {}) {
    this._ensureBuilt();
    this._node = node;
    if (profiles !== undefined) this._profiles = profiles;
    this._ensureContentForModelType(node._pllModelType || "Flux2Klein");
    this._el.style.display = "flex";
    const pw = 368, ph = 340;
    let left = (clientX ?? window.innerWidth  / 2) + 16;
    let top  = (clientY ?? window.innerHeight / 2) - 20;
    if (left + pw > window.innerWidth  - 8) left = Math.max(8, window.innerWidth  - pw - 8);
    if (top  + ph > window.innerHeight - 8) top  = Math.max(8, window.innerHeight - ph - 8);
    this._el.style.left = left + "px";
    this._el.style.top  = top  + "px";
    if (this._closeHandler) document.removeEventListener("mousedown", this._closeHandler);
    this._closeHandler = null;
    if (!this._locked) {
      this._closeHandler = (e) => { if (!this._el.contains(e.target)) this.hide(); };
      setTimeout(() => document.addEventListener("mousedown", this._closeHandler), 0);
    }
    this.refresh();
    this._startPolling();
  }

  hide() {
    if (this._el) this._el.style.display = "none";
    if (this._rafId) { cancelAnimationFrame(this._rafId); this._rafId = null; }
    if (this._closeHandler) { document.removeEventListener("mousedown", this._closeHandler); this._closeHandler = null; }
  }

  toggle(node, event) {
    if (this._el?.style.display === "flex" && this._node === node) {
      this.hide();
    } else {
      this.show(node, node._pllProfiles, { clientX: event?.clientX, clientY: event?.clientY });
    }
  }
}

const _stackHeatmapPanel = new PgcStackHeatmapPanel();

// ---------------------------------------------------------------------------
// Auto-mix helper
// ---------------------------------------------------------------------------
async function _autoMix(node) {
  pllDebug("[crt-pll] Auto Mix clicked");
  const loraWidgets = (node.widgets || []).filter(
    (w) => w.name?.startsWith("lora_") && w.value?.lora && w.value?.on
  );
  pllDebug("[crt-pll] lora widgets:", loraWidgets.length);
  if (loraWidgets.length === 0) {
    console.warn("[crt-pll] Auto Mix aborted: no selected loras");
    return;
  }

  const modelType = node._pllModelType || "Flux2Klein";
  const btnEl = node._pllAutomixButtonEl;
  if (btnEl) btnEl.textContent = "⏳ Analyzing…";

  try {
    const r = await fetch("/crt-pll/api/automix", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_type: modelType,
        loras: loraWidgets.map((w) => ({ name: w.value.lora, strength: w.value.strength ?? 1 })),
      }),
    });
    if (!r.ok) {
      const errText = await r.text();
      console.error("[pgc-pll] automix http error:", r.status, errText);
      return;
    }
    const data = await r.json();
    pllDebug("[crt-pll] automix response:", data);

    if (data.error) { console.error("[pgc-pll] automix:", data.error); return; }
    if (data.errors?.length) console.warn("[pgc-pll] automix warnings:", data.errors);

    node._pllProfiles = data.profiles;
    _stackHeatmapPanel._node = node;

    const cfg = BLOCK_CONFIGS[modelType] || BLOCK_CONFIGS["Flux2Klein"];
    const globalMult = Math.max(0, node._pllReductionMult ?? 1.0);
    (data.weights || []).forEach((bw, i) => {
      if (!loraWidgets[i]) return;
      const loraReduction = Math.max(0, loraWidgets[i].value.reduction ?? 1.0);
      const mult = globalMult * loraReduction;
      const scale = (w) => Math.round(Math.max(0.0, 1.0 - mult * (1.0 - w)) * 1000) / 1000;
      const blocks = {};
      for (const { key } of cfg) blocks[key] = (bw[key] || []).map(scale);
      loraWidgets[i].value.blocks = blocks;
    });

    // Normalize AFTER reduction: scale per-block so Σ(|strength_i| × block_weight_i[b]) ≤ cap
    if (node._pllNormalize) {
      const cap = Math.max(0.01, node._pllNormalizeCap ?? 1.0);
      const strengths = loraWidgets.map((w) => Math.abs(w.value.strength ?? 1.0));
      for (const { key, count } of cfg) {
        for (let b = 0; b < count; b++) {
          const total = loraWidgets.reduce(
            (sum, w, i) => sum + strengths[i] * (w.value.blocks?.[key]?.[b] ?? 1.0), 0
          );
          if (total > cap + 1e-9) {
            const factor = cap / total;
            loraWidgets.forEach((w) => {
              if (w.value.blocks?.[key]?.[b] != null)
                w.value.blocks[key][b] = Math.round(w.value.blocks[key][b] * factor * 1000) / 1000;
            });
          }
        }
      }
    }

    node.setDirtyCanvas(true, true);
    pllDebug("[crt-pll] Auto Mix applied");
  } catch (e) {
    console.error("[pgc-pll] automix fetch:", e);
  } finally {
    if (btnEl) btnEl.textContent = "⚡ Auto Mix";
  }
}

// ---------------------------------------------------------------------------
// Preset API helpers
// ---------------------------------------------------------------------------
async function _fetchPresets(modelType = "Flux2Klein") {
  try {
    const r = await fetch(`/crt-pll/api/presets?model_type=${encodeURIComponent(modelType)}`);
    const text = await r.text();
    if (!text) return {};
    let json;
    try {
      json = JSON.parse(text);
    } catch {
      return {};
    }
    return json?.data || {};
  } catch (e) {
    console.error("[pgc-pll] fetch presets:", e);
    return null;
  }
}

function _applyPreset(presetData, node) {
  if (!node.widgets) return;
  node.widgets = node.widgets.filter((w) => !w.name?.startsWith("lora_"));
  node.loraWidgetsCounter = 0;
  for (const loraData of presetData.loras || []) {
    const w = node.addNewLoraWidget();
    w.value = { ...loraData };
  }
  const wetWidget = node.widgets.find((w) => w.name === "wet");
  if (wetWidget && presetData.wet != null) {
    wetWidget._pct = Math.round(Math.max(0, Math.min(200, presetData.wet * 100)));
  }
  const capWidget = node.widgets.find((w) => w.name === "pgc_cap");
  if (capWidget && presetData.cap != null) {
    capWidget._pct = Math.round(Math.max(0, Math.min(200, presetData.cap * 100)));
  }
  node.setDirtyCanvas(true, true);
  node.size[1] = node.computeSize()[1];
}

// ---------------------------------------------------------------------------
// Automix controls DOM widget — intensity slider + stack view toggle
// ---------------------------------------------------------------------------
function _buildAutomixControlsDomWidget(node) {
  _injectStyles();
  const container = document.createElement("div");
  container.className = "pgc-pll-presets";

  const label = document.createElement("span");
  label.className  = "pgc-pll-presets-label";
  label.textContent = "Reduction ×";

  const reductionInput = document.createElement("input");
  reductionInput.type  = "number";
  reductionInput.min   = "0"; reductionInput.max = "5"; reductionInput.step = "0.1";
  reductionInput.value = (node._pllReductionMult ?? 1.0).toFixed(1);
  reductionInput.className = "pgc-pll-input";
  reductionInput.style.cssText = "flex: 0 0 54px; width: 54px; text-align: center;";
  reductionInput.title = "Reduction multiplier: 0 = no change, 1 = automix as-is, 2 = double the reduction, etc.";
  reductionInput.addEventListener("change", () => {
    node._pllReductionMult = Math.max(0, parseFloat(reductionInput.value) || 1);
    reductionInput.value = node._pllReductionMult.toFixed(1);
  });
  reductionInput.addEventListener("click",    e => e.stopPropagation());
  reductionInput.addEventListener("mousedown", e => e.stopPropagation());

  const stackBtn = document.createElement("button");
  stackBtn.className   = "pgc-pll-btn";
  stackBtn.textContent = "📊 Stack";
  stackBtn.title       = "Toggle real-time stack block load heatmap";
  stackBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    _stackHeatmapPanel.toggle(node, e);
  });

  const globalBlocksBtn = document.createElement("button");
  globalBlocksBtn.className   = "pgc-pll-btn";
  globalBlocksBtn.textContent = "⊞ Global";
  globalBlocksBtn.title       = "Edit global per-block multipliers applied to all LoRAs";
  globalBlocksBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const gbWidget = node._pllGlobalBlocksWidget;
    if (!gbWidget) return;
    _blockEditorPanel.show({
      blocks: JSON.parse(JSON.stringify(gbWidget._blocks)),
      title: "Global Block Multipliers",
      isPerLora: false,
      modelType: node._pllModelType || "Flux2Klein",
      onChange: (b) => { gbWidget._blocks = b; },
      clientX: e.clientX, clientY: e.clientY,
    });
  });

  const normalizeBtn = document.createElement("button");
  normalizeBtn.className = "pgc-pll-btn pgc-pll-btn-normalize";
  normalizeBtn.textContent = "⊜ Normalize";
  normalizeBtn.title = "When ON: after Auto Mix, scale per-block so Σ(|strength_i| × weight_i) ≤ cap";
  const _updateNormalizeBtn = () => {
    normalizeBtn.classList.toggle("active", !!node._pllNormalize);
  };
  normalizeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    node._pllNormalize = !node._pllNormalize;
    _updateNormalizeBtn();
  });
  _updateNormalizeBtn();

  const normCapInput = document.createElement("input");
  normCapInput.type  = "number";
  normCapInput.min   = "0.01"; normCapInput.max = "10"; normCapInput.step = "0.05";
  normCapInput.value = (node._pllNormalizeCap ?? 1.0).toFixed(2);
  normCapInput.className = "pgc-pll-input";
  normCapInput.style.cssText = "flex: 0 0 54px; width: 54px; text-align: center;";
  normCapInput.title = "Normalize cap — maximum allowed combined block contribution (default 1.0)";
  normCapInput.addEventListener("change", () => {
    node._pllNormalizeCap = Math.max(0.01, parseFloat(normCapInput.value) || 1.0);
    normCapInput.value = node._pllNormalizeCap.toFixed(2);
  });
  normCapInput.addEventListener("click",    (e) => e.stopPropagation());
  normCapInput.addEventListener("mousedown", (e) => e.stopPropagation());

  container.append(label, reductionInput, stackBtn, globalBlocksBtn, normalizeBtn, normCapInput);
  const widget = node.addDOMWidget("pgc_automix_ctrl", "div", container, { serialize: false });
  widget.computeSize = (w) => [Math.max(120, (w ?? node.size?.[0] ?? 380) - 20), 22];
  return widget;
}

function _buildAutoMixDomWidget(node) {
  _injectStyles();
  const container = document.createElement("div");
  container.className = "pgc-pll-presets";

  const btn = document.createElement("button");
  btn.className = "pgc-pll-btn";
  btn.style.width = "100%";
  btn.textContent = "⚡ Auto Mix";
  const runAutoMix = (e) => {
    e.stopPropagation();
    e.preventDefault();
    _autoMix(node);
  };
  btn.addEventListener("click", runAutoMix);
  btn.addEventListener("pointerup", runAutoMix);

  container.append(btn);
  node._pllAutomixButtonEl = btn;
  const widget = node.addDOMWidget("pgc_automix_btn", "div", container, { serialize: false });
  widget.computeSize = (w) => [Math.max(120, (w ?? node.size?.[0] ?? 380) - 20), 22];
  return widget;
}

function _buildPresetDomWidget(node) {
  _injectStyles();

  const container = document.createElement("div");
  container.className = "pgc-pll-presets";

  const label   = document.createElement("span");   label.className   = "pgc-pll-presets-label"; label.textContent = "Presets";
  const select  = document.createElement("select"); select.className  = "pgc-pll-select";
  const input   = document.createElement("input");  input.className   = "pgc-pll-input"; input.type = "text"; input.placeholder = "Name…";
  const loadBtn = document.createElement("button"); loadBtn.className = "pgc-pll-btn pgc-pll-btn-load"; loadBtn.textContent = "Load";
  const saveBtn = document.createElement("button"); saveBtn.className = "pgc-pll-btn pgc-pll-btn-save"; saveBtn.textContent = "💾 Save";
  const delBtn  = document.createElement("button"); delBtn.className  = "pgc-pll-btn pgc-pll-btn-del";  delBtn.textContent  = "🗑";

  container.append(label, select, loadBtn, input, saveBtn, delBtn);

  const mt = () => node._pllModelType || "Flux2Klein";

  async function refreshList() {
    const presets = await _fetchPresets(mt());
    if (!presets) return;
    select.innerHTML = "";
    const ph = document.createElement("option");
    ph.value = ""; ph.textContent = Object.keys(presets).length ? "— Select —" : "(none)";
    select.appendChild(ph);
    for (const name of Object.keys(presets)) {
      const opt = document.createElement("option");
      opt.value = name; opt.textContent = name;
      select.appendChild(opt);
    }
  }

  // Expose so model selector can trigger a refresh
  node._pllRefreshPresets = refreshList;

  loadBtn.addEventListener("click", async () => {
    const name = select.value; if (!name) return;
    const presets = await _fetchPresets(mt());
    if (!presets || !presets[name]) return;
    _applyPreset(presets[name], node);
  });

  saveBtn.addEventListener("click", async () => {
    const name = input.value.trim(); if (!name) { input.focus(); return; }
    const loraData = [];
    let wetPct = 100, capPct = 0;
    for (const w of node.widgets) {
      if (w.name?.startsWith("lora_")) loraData.push({ ...w.value });
      if (w.name === "wet") wetPct = w._pct ?? 100;
      if (w.name === "pgc_cap") capPct = w._pct ?? 0;
    }
    try {
      const r = await fetch("/crt-pll/api/presets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_type: mt(), name, data: { loras: loraData, wet: wetPct / 100, cap: capPct / 100 } }),
      });
      if (r.ok) { input.value = ""; await refreshList(); select.value = name; }
      else console.error("[pgc-pll] save error:", await r.text());
    } catch (e) { console.error("[pgc-pll] save:", e); }
  });

  delBtn.addEventListener("click", async () => {
    const name = select.value; if (!name) return;
    if (!confirm(`Delete preset "${name}"?`)) return;
    try {
      await fetch("/crt-pll/api/presets/" + encodeURIComponent(name) + "?model_type=" + encodeURIComponent(mt()), { method: "DELETE" });
      await refreshList();
    } catch (e) { console.error("[pgc-pll] delete:", e); }
  });

  refreshList();
  const widget = node.addDOMWidget("pgc_presets", "div", container, { serialize: false });
  widget.computeSize = (w) => [Math.max(120, (w ?? node.size?.[0] ?? 380) - 20), 22];
  return widget;
}

// ---------------------------------------------------------------------------
// PgcWetWidget — Dry/Wet multiplier 0–200%
// Serialized as { __pgc_wet: true, value: <float 0.0-2.0> }
// ---------------------------------------------------------------------------
class PgcWetWidget extends RgthreeBaseWidget {
  constructor(initialPct = 100) {
    super("wet");
    this.type = "custom";
    this._pct = Math.round(Math.max(0, Math.min(200, initialPct)));
    this.haveMouseMoved = false;
    this.hitAreas = {
      dec: { bounds: [0, 0], onClick: this.onDecClick },
      val: { bounds: [0, 0], onClick: this.onValClick },
      inc: { bounds: [0, 0], onClick: this.onIncClick },
      any: { bounds: [0, 0], onMove:  this.onAnyMove  },
    };
  }

  get value()  { return { __pgc_wet: true, value: this._pct / 100 }; }
  set value(v) {
    if (v != null && typeof v === "object" && v.__pgc_wet === true) {
      this._pct = Math.round(Math.max(0, Math.min(200, (v.value ?? 1) * 100)));
    } else if (typeof v === "number") {
      this._pct = Math.round(Math.max(0, Math.min(200, v * 100)));
    }
  }
  serializeValue() { return { __pgc_wet: true, value: this._pct / 100 }; }

  draw(ctx, node, w, posY, height) {
    const margin = 10, innerMargin = margin * 0.33;
    const lowQuality = isLowQuality();
    const midY = posY + height * 0.5;
    ctx.save();
    if (!lowQuality) {
      const fillFrac = this._pct / 200;
      const barLeft = margin, barW = node.size[0] - margin * 2;
      const barH = height - 4, barY = posY + 2;
      ctx.fillStyle = "rgba(0,0,0,0.25)";
      ctx.beginPath(); ctx.roundRect(barLeft, barY, barW, barH, 3); ctx.fill();
      const grad = ctx.createLinearGradient(barLeft, 0, barLeft + barW, 0);
      grad.addColorStop(0, ACCENT_DIM);
      grad.addColorStop(Math.min(1, fillFrac + 0.01), ACCENT_MED);
      grad.addColorStop(Math.min(1, fillFrac + 0.01), "transparent");
      grad.addColorStop(1, "transparent");
      ctx.fillStyle = grad;
      ctx.beginPath(); ctx.roundRect(barLeft, barY, barW, barH, 3); ctx.fill();

      ctx.globalAlpha = app.canvas.editor_alpha * 0.55;
      ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
      ctx.textBaseline = "middle"; ctx.textAlign = "left";
      ctx.fillText("Dry / Wet", margin + 8, midY);

      let posX = node.size[0] - margin - innerMargin - innerMargin - WET_WIDGET_TOTAL;
      ctx.globalAlpha = app.canvas.editor_alpha;
      ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
      ctx.fill(_arrowLeft(posX, midY));
      this.hitAreas.dec.bounds = [posX, _AW]; posX += _AW + _IM;
      ctx.textAlign = "center";
      ctx.fillStyle = this._pct === 100 ? LiteGraph.WIDGET_TEXT_COLOR : SIGMA_COLOR;
      ctx.fillText(this._pct + "%", posX + _NW / 2, midY);
      this.hitAreas.val.bounds = [posX, _NW]; posX += _NW + _IM;
      ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
      ctx.fill(_arrowRight(posX, midY));
      this.hitAreas.inc.bounds = [posX, _AW];
      this.hitAreas.any.bounds = [this.hitAreas.dec.bounds[0], this.hitAreas.inc.bounds[0] + _AW - this.hitAreas.dec.bounds[0]];
    }
    ctx.restore();
  }

  onDecClick(event, pos, node) { this._pct = Math.max(0,   this._pct - 5); node.setDirtyCanvas(true, false); }
  onIncClick(event, pos, node) { this._pct = Math.min(200, this._pct + 5); node.setDirtyCanvas(true, false); }
  onValClick(event, pos, node) {
    if (this.haveMouseMoved) return;
    app.canvas.prompt("Wet % (0 – 200)", this._pct, (v) => {
      const n = Number(v);
      if (!isNaN(n)) { this._pct = Math.round(Math.max(0, Math.min(200, n))); node.setDirtyCanvas(true, false); }
    }, event);
  }
  onAnyMove(event, pos, node) {
    if (event.deltaX) { this.haveMouseMoved = true; this._pct = Math.round(Math.max(0, Math.min(200, this._pct + event.deltaX))); node.setDirtyCanvas(true, false); }
  }
  onMouseUp(event, pos, node) { super.onMouseUp(event, pos, node); this.haveMouseMoved = false; }
}

// ---------------------------------------------------------------------------
// PgcCapWidget — per-node strength cap (0 = OFF, otherwise clamps |strength|)
// Serialized as { __pgc_cap: true, value: <float 0.0-2.0> }  (0 = disabled)
// ---------------------------------------------------------------------------
class PgcCapWidget extends RgthreeBaseWidget {
  constructor(initialPct = 0) {
    super("pgc_cap");
    this.type = "custom";
    this._pct = Math.round(Math.max(0, Math.min(200, initialPct)));
    this.haveMouseMoved = false;
    this.hitAreas = {
      dec: { bounds: [0, 0], onClick: this.onDecClick },
      val: { bounds: [0, 0], onClick: this.onValClick },
      inc: { bounds: [0, 0], onClick: this.onIncClick },
      any: { bounds: [0, 0], onMove:  this.onAnyMove  },
    };
  }

  get value()  { return { __pgc_cap: true, value: this._pct / 100 }; }
  set value(v) {
    if (v != null && typeof v === "object" && v.__pgc_cap === true) {
      this._pct = Math.round(Math.max(0, Math.min(200, (v.value ?? 0) * 100)));
    } else if (typeof v === "number") {
      this._pct = Math.round(Math.max(0, Math.min(200, v * 100)));
    }
  }
  serializeValue() { return { __pgc_cap: true, value: this._pct / 100 }; }

  draw(ctx, node, w, posY, height) {
    const margin = 10, innerMargin = margin * 0.33;
    const lowQuality = isLowQuality();
    const midY = posY + height * 0.5;
    ctx.save();
    if (!lowQuality) {
      const fillFrac = this._pct / 200;
      const barLeft = margin, barW = node.size[0] - margin * 2;
      const barH = height - 4, barY = posY + 2;
      ctx.fillStyle = "rgba(0,0,0,0.25)";
      ctx.beginPath(); ctx.roundRect(barLeft, barY, barW, barH, 3); ctx.fill();
      if (this._pct > 0) {
        const grad = ctx.createLinearGradient(barLeft, 0, barLeft + barW, 0);
        grad.addColorStop(0, CAP_DIM);
        grad.addColorStop(Math.min(1, fillFrac + 0.01), CAP_MED);
        grad.addColorStop(Math.min(1, fillFrac + 0.01), "transparent");
        grad.addColorStop(1, "transparent");
        ctx.fillStyle = grad;
        ctx.beginPath(); ctx.roundRect(barLeft, barY, barW, barH, 3); ctx.fill();
      }

      ctx.globalAlpha = app.canvas.editor_alpha * 0.55;
      ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
      ctx.textBaseline = "middle"; ctx.textAlign = "left";
      ctx.fillText("Strength Cap", margin + 8, midY);

      let posX = node.size[0] - margin - innerMargin - innerMargin - WET_WIDGET_TOTAL;
      ctx.globalAlpha = app.canvas.editor_alpha;
      ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
      ctx.fill(_arrowLeft(posX, midY));
      this.hitAreas.dec.bounds = [posX, _AW]; posX += _AW + _IM;
      ctx.textAlign = "center";
      ctx.fillStyle = this._pct === 0 ? "rgba(255,255,255,0.35)" : CAP_COLOR;
      ctx.fillText(this._pct === 0 ? "OFF" : this._pct + "%", posX + _NW / 2, midY);
      this.hitAreas.val.bounds = [posX, _NW]; posX += _NW + _IM;
      ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
      ctx.fill(_arrowRight(posX, midY));
      this.hitAreas.inc.bounds = [posX, _AW];
      this.hitAreas.any.bounds = [this.hitAreas.dec.bounds[0], this.hitAreas.inc.bounds[0] + _AW - this.hitAreas.dec.bounds[0]];
    }
    ctx.restore();
  }

  onDecClick(event, pos, node) { this._pct = Math.max(0,   this._pct - 5); node.setDirtyCanvas(true, false); }
  onIncClick(event, pos, node) { this._pct = Math.min(200, this._pct + 5); node.setDirtyCanvas(true, false); }
  onValClick(event, pos, node) {
    if (this.haveMouseMoved) return;
    app.canvas.prompt("Cap % (0 = OFF, 1–200 = max |strength|)", this._pct, (v) => {
      const n = Number(v);
      if (!isNaN(n)) { this._pct = Math.round(Math.max(0, Math.min(200, n))); node.setDirtyCanvas(true, false); }
    }, event);
  }
  onAnyMove(event, pos, node) {
    if (event.deltaX) { this.haveMouseMoved = true; this._pct = Math.round(Math.max(0, Math.min(200, this._pct + event.deltaX))); node.setDirtyCanvas(true, false); }
  }
  onMouseUp(event, pos, node) { super.onMouseUp(event, pos, node); this.haveMouseMoved = false; }
}

// ---------------------------------------------------------------------------
// PgcGlobalBlocksWidget — invisible zero-height widget that serializes global
// per-block multipliers so they're saved with the workflow
// ---------------------------------------------------------------------------
class PgcGlobalBlocksWidget extends RgthreeBaseWidget {
  constructor(modelType = "Flux2Klein") {
    super("pgc_global_blocks");
    this.type = "custom";
    this._blocks = _defaultBlocks(modelType);
  }
  get value() {
    const result = { __pgc_global_blocks: true };
    for (const [k, v] of Object.entries(this._blocks)) result[k] = [...v];
    return result;
  }
  set value(v) {
    if (v?.__pgc_global_blocks === true) {
      const blocks = {};
      for (const key of ["double", "single", "transformer", "layers", "blocks"]) {
        if (v[key]?.length) blocks[key] = Array.from(v[key]);
      }
      if (Object.keys(blocks).length) this._blocks = blocks;
    }
  }
  serializeValue() { return this.value; }
  draw() {}
  computeSize(w) { return [w ?? 0, -4]; }
}

// ---------------------------------------------------------------------------
// PgcModelTypeWidget — hidden zero-height widget that serializes model type
// ---------------------------------------------------------------------------
class PgcModelTypeWidget extends RgthreeBaseWidget {
  constructor(modelType = "Flux2Klein") {
    super("pgc_model_type");
    this.type = "custom";
    this._modelType = MODEL_TYPES.includes(modelType) ? modelType : "Flux2Klein";
  }
  get value()  { return { __pgc_model_type: true, value: this._modelType }; }
  set value(v) {
    if (v?.__pgc_model_type === true && MODEL_TYPES.includes(v.value)) this._modelType = v.value;
    else if (typeof v === "string" && MODEL_TYPES.includes(v)) this._modelType = v;
  }
  serializeValue() { return this.value; }
  draw() {}
  computeSize(w) { return [w ?? 0, -4]; }
}

// ---------------------------------------------------------------------------
// Model type selector DOM widget
// ---------------------------------------------------------------------------
function _buildModelSelectDomWidget(node, initialModelType = "Flux2Klein") {
  _injectStyles();
  const container = document.createElement("div");
  container.className = "pgc-pll-model-select-wrap";

  const label = document.createElement("span");
  label.className = "pgc-pll-presets-label";
  label.textContent = "Model";

  const select = document.createElement("select");
  select.className = "pgc-pll-select";
  for (const mt of MODEL_TYPES) {
    const opt = document.createElement("option");
    opt.value = mt; opt.textContent = mt;
    if (mt === initialModelType) opt.selected = true;
    select.appendChild(opt);
  }

  select.addEventListener("change", () => {
    const mt = select.value;
    if (node._pllModelTypeWidget) node._pllModelTypeWidget._modelType = mt;
    node._pllModelType = mt;
    // Reset global blocks to the default for the new model type
    const gbWidget = node._pllGlobalBlocksWidget;
    if (gbWidget) gbWidget._blocks = _defaultBlocks(mt);
    // Refresh preset list (only shows presets for this model type)
    node._pllRefreshPresets?.();
    // Close panels — they'll rebuild for the new model type on next open
    _blockEditorPanel.hide();
    _stackHeatmapPanel.hide();
    node.setDirtyCanvas(true, true);
  });
  select.addEventListener("click",    e => e.stopPropagation());
  select.addEventListener("mousedown", e => e.stopPropagation());

  container.append(label, select);
  const widget = node.addDOMWidget("pgc_model_select", "div", container, { serialize: false });
  widget.computeSize = (w) => [Math.max(120, (w ?? node.size?.[0] ?? 380) - 20), 22];
  return widget;
}

// ---------------------------------------------------------------------------
// MagicLoraLoaderHeaderWidget — Toggle-all + Σ effective strength
// ---------------------------------------------------------------------------
class MagicLoraLoaderHeaderWidget extends RgthreeBaseWidget {
  constructor(name = "MagicLoraLoaderHeaderWidget") {
    super(name);
    this.value = { type: "MagicLoraLoaderHeaderWidget" };
    this.type = "custom";
    this.options = { serialize: false };
    this.hitAreas = { toggle: { bounds: [0, 0], onDown: this.onToggleDown } };
    this.showModelAndClip = null;
  }

  draw(ctx, node, w, posY, height) {
    if (!node.hasLoraWidgets()) return;
    this.showModelAndClip = node.properties[PROP_LABEL_SHOW_STRENGTHS] === PROP_VALUE_SHOW_STRENGTHS_SEPARATE;
    const margin = 10, innerMargin = margin * 0.33;
    const lowQuality = isLowQuality();
    posY += 2;
    const midY = posY + height * 0.5;
    let posX = 10;

    ctx.save();
    if (!lowQuality) {
      ctx.strokeStyle = ACCENT_MED; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(margin * 2, posY); ctx.lineTo(node.size[0] - margin * 2, posY); ctx.stroke();
    }

    this.hitAreas.toggle.bounds = drawTogglePart(ctx, { posX, posY, height, value: node.allLorasState() });

    if (!lowQuality) {
      posX += this.hitAreas.toggle.bounds[1] + innerMargin;
      ctx.globalAlpha = app.canvas.editor_alpha * 0.55;
      ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
      ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText("Toggle All", posX, midY);

      let rawTotal = 0;
      for (const widget of node.widgets) {
        if (widget.name?.startsWith("lora_") && widget.value?.on === true) rawTotal += widget.value?.strength ?? 0;
      }
      const wetWidget = node.widgets.find((ww) => ww.name === "wet");
      const wet = (wetWidget?._pct ?? 100) / 100;
      posX += ctx.measureText("Toggle All").width + innerMargin * 4;
      ctx.globalAlpha = app.canvas.editor_alpha * 0.85;
      ctx.fillStyle = SIGMA_COLOR;
      ctx.fillText("Σ " + (rawTotal * wet).toFixed(2), posX, midY);

      let rposX = node.size[0] - margin - innerMargin - innerMargin;
      ctx.textAlign = "center"; ctx.globalAlpha = app.canvas.editor_alpha * 0.45;
      ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
      ctx.fillText(this.showModelAndClip ? "Clip" : "Strength", rposX - drawNumberWidgetPart.WIDTH_TOTAL / 2, midY);
      if (this.showModelAndClip) {
        rposX -= drawNumberWidgetPart.WIDTH_TOTAL + innerMargin * 2;
        ctx.fillText("Model", rposX - drawNumberWidgetPart.WIDTH_TOTAL / 2, midY);
      }
    }
    ctx.restore();
  }

  onToggleDown(event, pos, node) { node.toggleAllLoras(); this.cancelMouseDown(); return true; }
}

// ---------------------------------------------------------------------------
// MagicLoraLoaderWidget — per-LoRA row
// ---------------------------------------------------------------------------
const DEFAULT_LORA_WIDGET_DATA = {
  on: true, lora: null, strength: 1, strengthTwo: null, blocks: null, reduction: 1.0,
};

class MagicLoraLoaderWidget extends RgthreeBaseWidget {
  constructor(name) {
    super(name);
    this.type = "custom";
    this.haveMouseMovedStrength = false;
    this.loraInfoPromise = null;
    this.loraInfo = null;
    this.showModelAndClip = null;
    this.hitAreas = {
      toggle:         { bounds: [0, 0], onDown:  this.onToggleDown        },
      lora:           { bounds: [0, 0], onClick: this.onLoraClick          },
      blocks:         { bounds: [0, 0], onClick: this.onBlocksClick        },
      strengthDec:    { bounds: [0, 0], onClick: this.onStrengthDecDown    },
      strengthVal:    { bounds: [0, 0], onClick: this.onStrengthValUp      },
      strengthInc:    { bounds: [0, 0], onClick: this.onStrengthIncDown    },
      strengthAny:    { bounds: [0, 0], onMove:  this.onStrengthAnyMove    },
      strengthTwoDec: { bounds: [0, 0], onClick: this.onStrengthTwoDecDown },
      strengthTwoVal: { bounds: [0, 0], onClick: this.onStrengthTwoValUp   },
      strengthTwoInc: { bounds: [0, 0], onClick: this.onStrengthTwoIncDown },
      strengthTwoAny: { bounds: [0, 0], onMove:  this.onStrengthTwoAnyMove },
    };
    this._value = { ...DEFAULT_LORA_WIDGET_DATA };
  }

  set value(v) {
    this._value = (v && typeof v === "object") ? v : { ...DEFAULT_LORA_WIDGET_DATA };
    if (this.showModelAndClip && this._value.strengthTwo == null) this._value.strengthTwo = this._value.strength;
    this.getLoraInfo();
  }
  get value() { return this._value; }

  setLora(lora) { this._value.lora = lora; this.getLoraInfo(); }

  draw(ctx, node, w, posY, height) {
    const currentShowModelAndClip = node.properties[PROP_LABEL_SHOW_STRENGTHS] === PROP_VALUE_SHOW_STRENGTHS_SEPARATE;
    if (this.showModelAndClip !== currentShowModelAndClip) {
      const old = this.showModelAndClip;
      this.showModelAndClip = currentShowModelAndClip;
      if (this.showModelAndClip) { if (old != null) this.value.strengthTwo = this.value.strength ?? 1; }
      else {
        this.value.strengthTwo = null;
        this.hitAreas.strengthTwoDec.bounds = [0,-1]; this.hitAreas.strengthTwoVal.bounds = [0,-1];
        this.hitAreas.strengthTwoInc.bounds = [0,-1]; this.hitAreas.strengthTwoAny.bounds = [0,-1];
      }
    }

    ctx.save();
    const margin = 10, innerMargin = margin * 0.33;
    const lowQuality = isLowQuality();
    const midY = posY + height * 0.5;
    let posX = margin;

    if (!lowQuality && this._value.on) {
      ctx.fillStyle = ACCENT_DIM;
      ctx.beginPath(); ctx.roundRect(posX, posY + 1, node.size[0] - margin * 2, height - 2, 4); ctx.fill();
    }
    drawRoundedRectangle(ctx, { pos: [posX, posY], size: [node.size[0] - margin * 2, height] });

    this.hitAreas.toggle.bounds = drawTogglePart(ctx, { posX, posY, height, value: this.value.on });
    posX += this.hitAreas.toggle.bounds[1] + innerMargin;

    if (lowQuality) { ctx.restore(); return; }
    if (!this.value.on) ctx.globalAlpha = app.canvas.editor_alpha * 0.4;

    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    let rposX = node.size[0] - margin - innerMargin - innerMargin;

    // Strength controls (drawn right-to-left)
    const strengthValue = this.showModelAndClip ? (this.value.strengthTwo ?? 1) : (this.value.strength ?? 1);
    let textColor;
    if (this.loraInfo?.strengthMax != null && strengthValue > this.loraInfo.strengthMax) textColor = "#c66";
    else if (this.loraInfo?.strengthMin != null && strengthValue < this.loraInfo.strengthMin) textColor = "#c66";

    const [la, tx, ra] = drawNumberWidgetPart(ctx, { posX: rposX, posY, height, value: strengthValue, direction: -1, textColor });
    this.hitAreas.strengthDec.bounds = la; this.hitAreas.strengthVal.bounds = tx; this.hitAreas.strengthInc.bounds = ra;
    this.hitAreas.strengthAny.bounds = [la[0], ra[0] + ra[1] - la[0]];
    rposX = la[0] - innerMargin;

    if (this.showModelAndClip) {
      rposX -= innerMargin;
      this.hitAreas.strengthTwoDec.bounds = this.hitAreas.strengthDec.bounds;
      this.hitAreas.strengthTwoVal.bounds = this.hitAreas.strengthVal.bounds;
      this.hitAreas.strengthTwoInc.bounds = this.hitAreas.strengthInc.bounds;
      this.hitAreas.strengthTwoAny.bounds = this.hitAreas.strengthAny.bounds;
      let tc2;
      if (this.loraInfo?.strengthMax != null && this.value.strength > this.loraInfo.strengthMax) tc2 = "#c66";
      else if (this.loraInfo?.strengthMin != null && this.value.strength < this.loraInfo.strengthMin) tc2 = "#c66";
      const [la2, tx2, ra2] = drawNumberWidgetPart(ctx, { posX: rposX, posY, height, value: this.value.strength ?? 1, direction: -1, textColor: tc2 });
      this.hitAreas.strengthDec.bounds = la2; this.hitAreas.strengthVal.bounds = tx2; this.hitAreas.strengthInc.bounds = ra2;
      this.hitAreas.strengthAny.bounds = [la2[0], ra2[0] + ra2[1] - la2[0]];
      rposX = la2[0] - innerMargin;
    }

    // Info icon
    const infoIconSize = height * 0.66;
    if (this.hitAreas["info"]) {
      rposX -= innerMargin;
      drawInfoIcon(ctx, rposX - infoIconSize, posY + (height - infoIconSize) / 2, infoIconSize);
      this.hitAreas.info.bounds = [rposX - infoIconSize, infoIconSize + innerMargin * 2];
      rposX = rposX - infoIconSize - innerMargin;
    }

    // Blocks icon
    const iconSize = 10;
    rposX -= innerMargin * 2;
    const hasCustomBlocks = this.value.blocks != null;
    const iconColor = hasCustomBlocks ? ACCENT : "rgba(255,255,255,0.2)";
    ctx.globalAlpha = app.canvas.editor_alpha * (hasCustomBlocks ? 1 : 0.5);
    _drawBlocksIcon(ctx, rposX - iconSize, posY + (height - iconSize) / 2, iconSize, iconColor);
    this.hitAreas.blocks.bounds = [rposX - iconSize - innerMargin, iconSize + innerMargin * 2];
    rposX = rposX - iconSize - innerMargin * 2;

    // LoRA name
    ctx.globalAlpha = this.value.on ? app.canvas.editor_alpha : app.canvas.editor_alpha * 0.4;
    const loraWidth = rposX - posX;
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(fitString(ctx, String(this.value?.lora || "None"), loraWidth), posX, midY);
    this.hitAreas.lora.bounds = [posX, loraWidth];

    ctx.globalAlpha = app.canvas.editor_alpha;
    ctx.restore();
  }

  serializeValue(node, index) {
    const v = { ...this.value };
    if (!this.showModelAndClip) { delete v.strengthTwo; }
    else { this.value.strengthTwo = this.value.strengthTwo ?? 1; v.strengthTwo = this.value.strengthTwo; }
    // omit blocks:null to save space
    if (v.blocks == null) delete v.blocks;
    return v;
  }

  onToggleDown(event, pos, node) { this.value.on = !this.value.on; this.cancelMouseDown(); return true; }
  onLoraClick(event, pos, node) {
    rgthreeApi.getLoras().then((lorasDetails) => {
      const loras = lorasDetails.map((l) => l.file);
      showLoraChooser(event, (value) => {
        if (typeof value === "string") { this.value.lora = value; this.loraInfo = null; this.getLoraInfo(); }
        node.setDirtyCanvas(true, true);
      }, this.value.lora, loras);
    });
    this.cancelMouseDown();
  }
  onBlocksClick(event, pos, node) {
    const currentBlocks = this.value.blocks ? JSON.parse(JSON.stringify(this.value.blocks)) : _defaultBlocks(node._pllModelType);
    const loraShort = this.value.lora
      ? this.value.lora.split(/[/\\]/).pop().replace(/\.safetensors$/i, "").slice(0, 20)
      : "LoRA";
    _blockEditorPanel.show({
      blocks: currentBlocks,
      title: loraShort,
      isPerLora: true,
      loraName: this.value.lora || null,
      modelType: node._pllModelType || "Flux2Klein",
      reduction: this.value.reduction ?? 1.0,
      onChange: (b) => { this.value.blocks = b; node.setDirtyCanvas(true, false); },
      onReductionChange: (v) => { this.value.reduction = v; },
      onUseGlobal: () => { this.value.blocks = null; node.setDirtyCanvas(true, false); },
      clientX: event.clientX, clientY: event.clientY,
    });
    this.cancelMouseDown();
  }
  onStrengthDecDown(event, pos, node)     { this.stepStrength(-1, false); }
  onStrengthIncDown(event, pos, node)     { this.stepStrength( 1, false); }
  onStrengthTwoDecDown(event, pos, node)  { this.stepStrength(-1, true);  }
  onStrengthTwoIncDown(event, pos, node)  { this.stepStrength( 1, true);  }
  onStrengthAnyMove(event, pos, node)     { this._doStrengthMove(event, false); }
  onStrengthTwoAnyMove(event, pos, node)  { this._doStrengthMove(event, true);  }

  _doStrengthMove(event, isTwo) {
    if (event.deltaX) {
      this.haveMouseMovedStrength = true;
      const prop = isTwo ? "strengthTwo" : "strength";
      this.value[prop] = (this.value[prop] ?? 1) + event.deltaX * 0.05;
    }
  }
  onStrengthValUp(event, pos, node)    { this._doValUp(event, false); }
  onStrengthTwoValUp(event, pos, node) { this._doValUp(event, true); }
  _doValUp(event, isTwo) {
    if (this.haveMouseMovedStrength) return;
    const prop = isTwo ? "strengthTwo" : "strength";
    app.canvas.prompt("Value", this.value[prop], (v) => (this.value[prop] = Number(v)), event);
  }
  onMouseUp(event, pos, node) { super.onMouseUp(event, pos, node); this.haveMouseMovedStrength = false; }

  showLoraInfoDialog() {
    if (!this.value.lora || this.value.lora === "None") return;
    const dlg = new RgthreeLoraInfoDialog(this.value.lora).show();
    dlg.addEventListener("close", (e) => { if (e.detail.dirty) this.getLoraInfo(true); });
  }

  stepStrength(direction, isTwo) {
    const prop = isTwo ? "strengthTwo" : "strength";
    this.value[prop] = Math.round(((this.value[prop] ?? 1) + 0.05 * direction) * 100) / 100;
  }

  getLoraInfo(force = false) {
    if (!this.loraInfoPromise || force) {
      const promise = (this.value.lora && this.value.lora !== "None")
        ? LORA_INFO_SERVICE.getInfo(this.value.lora, force, true)
        : Promise.resolve(null);
      this.loraInfoPromise = promise.then((v) => (this.loraInfo = v));
    }
    return this.loraInfoPromise;
  }
}

// ---------------------------------------------------------------------------
// CrtMagicLoraLoaderNode — main node
// ---------------------------------------------------------------------------
var _a;
const CRT_PLL_MIN_WIDTH = 380;

class CrtMagicLoraLoaderNode extends RgthreeBaseServerNode {
  constructor(title = NODE_CLASS.title) {
    super(title);
    this.serialize_widgets = true;
    this.logger = rgthree.newLogSession(`[CRT Magic LoRA Loader]`);
    this.loraWidgetsCounter = 0;
    this.widgetButtonSpacer = null;
    this.min_size = [CRT_PLL_MIN_WIDTH, 200];
    this.size = this.size || [CRT_PLL_MIN_WIDTH, 200];
    this.size[0] = Math.max(this.size[0] || 0, CRT_PLL_MIN_WIDTH);
    this.properties[PROP_LABEL_SHOW_STRENGTHS] = PROP_VALUE_SHOW_STRENGTHS_SINGLE;
    rgthreeApi.getLoras();
    if (rgthree.loadingApiJson) {
      const fullApiJson = rgthree.loadingApiJson;
      setTimeout(() => this.configureFromApiJson(fullApiJson), 16);
    }
  }

  configureFromApiJson(fullApiJson) {
    if (this.id == null) return;
    const nodeData = fullApiJson[this.id] || fullApiJson[String(this.id)] || fullApiJson[Number(this.id)];
    if (nodeData == null) return;
    this.configure({
      widgets_values: Object.values(nodeData.inputs).filter((input) => typeof input?.lora === "string"),
    });
  }

  configure(info) {
    while (this.widgets?.length) this.removeWidget(0);
    this.widgetButtonSpacer = null;

    if (info.id != null) super.configure(info);

    this._tempWidth  = this.size[0];
    this._tempHeight = this.size[1];

    // Scan saved values for model type, wet pct, cap pct and global blocks
    let savedModelType = "Flux2Klein";
    let savedWetPct = 100;
    let savedCapPct = 0;
    let savedGlobalBlocks = null;
    for (const v of info.widgets_values || []) {
      if (v?.__pgc_model_type === true) {
        savedModelType = MODEL_TYPES.includes(v.value) ? v.value : "Flux2Klein";
      } else if (v?.__pgc_wet === true) {
        savedWetPct = Math.round(Math.max(0, Math.min(200, (v.value ?? 1) * 100)));
      } else if (v?.__pgc_cap === true) {
        savedCapPct = Math.round(Math.max(0, Math.min(200, (v.value ?? 0) * 100)));
      } else if (v?.__pgc_global_blocks === true) {
        const blocks = {};
        for (const key of ["double", "single", "transformer", "layers", "blocks"]) {
          if (v[key]?.length) blocks[key] = Array.from(v[key]);
        }
        if (Object.keys(blocks).length) savedGlobalBlocks = blocks;
      }
    }

    // Recreate lora widgets (skip sentinel objects)
    for (const widgetValue of info.widgets_values || []) {
      if (widgetValue?.lora !== undefined) {
        const widget = this.addNewLoraWidget();
        widget.value = { ...DEFAULT_LORA_WIDGET_DATA, ...widgetValue };
      }
    }

    this.addNonLoraWidgets(savedWetPct, savedGlobalBlocks, savedCapPct, savedModelType);

    this.size[0] = Math.max(this._tempWidth || 0, CRT_PLL_MIN_WIDTH);
    this.size[1] = Math.max(this._tempHeight, this.computeSize()[1]);
    this.enforceMinWidth();
  }

  onNodeCreated() {
    super.onNodeCreated?.call(this);
    if (this.widgets?.some((w) => w.name === "pgc_presets" || w.name === "wet" || w.name === "pgc_model_select")) {
      return;
    }
    this.addNonLoraWidgets();
    const computed = this.computeSize();
    this.size = this.size || [0, 0];
    this.size[0] = Math.max(this.size[0], computed[0], CRT_PLL_MIN_WIDTH);
    this.size[1] = Math.max(this.size[1], computed[1]);
    this.enforceMinWidth();
    this.setDirtyCanvas(true, true);
  }

  onResize(size) {
    this.enforceMinWidth();
    return super.onResize?.(size);
  }

  enforceMinWidth() {
    this.min_size = this.min_size || [CRT_PLL_MIN_WIDTH, 200];
    this.min_size[0] = Math.max(this.min_size[0] || 0, CRT_PLL_MIN_WIDTH);
    this.size = this.size || [CRT_PLL_MIN_WIDTH, 200];
    if ((this.size[0] || 0) < CRT_PLL_MIN_WIDTH) {
      this.size[0] = CRT_PLL_MIN_WIDTH;
    }
  }

  addNewLoraWidget(lora) {
    this.loraWidgetsCounter++;
    const widget = this.addCustomWidget(new MagicLoraLoaderWidget("lora_" + this.loraWidgetsCounter));
    if (lora) widget.setLora(lora);
    if (this.widgetButtonSpacer) moveArrayItem(this.widgets, widget, this.widgets.indexOf(this.widgetButtonSpacer));
    return widget;
  }

  addNonLoraWidgets(wetPct = 100, globalBlocks = null, capPct = 0, modelType = "Flux2Klein") {
    // Cache model type on the node for easy access
    this._pllModelType = MODEL_TYPES.includes(modelType) ? modelType : "Flux2Klein";

    // Top rows (requested order): Presets -> Model -> Auto Mix -> Automix controls
    moveArrayItem(this.widgets, _buildPresetDomWidget(this), 0);
    moveArrayItem(this.widgets, _buildModelSelectDomWidget(this, this._pllModelType), 1);

    moveArrayItem(this.widgets, _buildAutoMixDomWidget(this), 2);
    moveArrayItem(this.widgets, _buildAutomixControlsDomWidget(this), 3);

    // Small breathing space before header row
    moveArrayItem(
      this.widgets,
      this.addCustomWidget(new RgthreeDividerWidget({ marginTop: 2, marginBottom: 2, thickness: 0 })),
      4,
    );

    // Main rows under automix
    moveArrayItem(this.widgets, this.addCustomWidget(new MagicLoraLoaderHeaderWidget()), 5);
    moveArrayItem(this.widgets, this.addCustomWidget(new PgcWetWidget(wetPct)), 6);
    moveArrayItem(this.widgets, this.addCustomWidget(new PgcCapWidget(capPct)), 7);

    // Hidden serializable widget for model type
    const mtWidget = this.addCustomWidget(new PgcModelTypeWidget(this._pllModelType));
    this._pllModelTypeWidget = mtWidget;

    // Hidden serializable widget for global block multipliers
    const gbWidget = this.addCustomWidget(new PgcGlobalBlocksWidget(this._pllModelType));
    if (globalBlocks) gbWidget.value = { __pgc_global_blocks: true, ...globalBlocks };
    this._pllGlobalBlocksWidget = gbWidget;

    // Spacer before footer (lora widgets sit above this spacer)
    this.widgetButtonSpacer = this.addCustomWidget(new RgthreeDividerWidget({ marginTop: 4, marginBottom: 0, thickness: 0 }));

    // + Add Lora button
    this.addCustomWidget(new RgthreeBetterButtonWidget("➕ Add Lora", (event, pos, node) => {
      rgthreeApi.getLoras().then((lorasDetails) => {
        const loras = lorasDetails.map((l) => l.file);
        showLoraChooser(event, (value) => {
          if (typeof value === "string" && !value.includes("Magic LoRA Chooser") && value !== "NONE") {
            this.addNewLoraWidget(value);
            this.size[1] = Math.max(this._tempHeight ?? 15, this.computeSize()[1]);
            this.setDirtyCanvas(true, true);
          }
        }, null, [...loras]);
      });
      return true;
    }));
  }

  getSlotInPosition(canvasX, canvasY) {
    const slot = super.getSlotInPosition(canvasX, canvasY);
    if (!slot) {
      let lastWidget = null;
      for (const widget of this.widgets) {
        if (!widget.last_y) return;
        if (canvasY > this.pos[1] + widget.last_y) { lastWidget = widget; continue; }
        break;
      }
      if (lastWidget?.name?.startsWith("lora_")) return { widget: lastWidget, output: { type: "LORA WIDGET" } };
    }
    return slot;
  }

  getSlotMenuOptions(slot) {
    if (slot?.widget?.name?.startsWith("lora_")) {
      const widget = slot.widget;
      const index = this.widgets.indexOf(widget);
      const canMoveUp   = !!this.widgets[index - 1]?.name?.startsWith("lora_");
      const canMoveDown = !!this.widgets[index + 1]?.name?.startsWith("lora_");
      new LiteGraph.ContextMenu([
        { content: "ℹ️ Show Info",   callback: () => widget.showLoraInfoDialog() },
        null,
        { content: `${widget.value.on ? "⚫" : "🟢"} Toggle ${widget.value.on ? "Off" : "On"}`,
          callback: () => { widget.value.on = !widget.value.on; } },
        { content: "⬆️ Move Up",    disabled: !canMoveUp,   callback: () => moveArrayItem(this.widgets, widget, index - 1) },
        { content: "⬇️ Move Down",  disabled: !canMoveDown, callback: () => moveArrayItem(this.widgets, widget, index + 1) },
        { content: "🗑️ Remove",     callback: () => removeArrayItem(this.widgets, widget) },
      ], { title: "LORA WIDGET", event: rgthree.lastCanvasMouseEvent });
      return undefined;
    }
    return this.defaultGetSlotMenuOptions(slot);
  }

  refreshComboInNode(defs) { rgthreeApi.getLoras(true); }

  hasLoraWidgets() { return !!this.widgets?.find((w) => w.name?.startsWith("lora_")); }

  allLorasState() {
    let allOn = true, allOff = true;
    for (const widget of this.widgets) {
      if (widget.name?.startsWith("lora_")) {
        allOn  = allOn  && widget.value?.on === true;
        allOff = allOff && widget.value?.on === false;
        if (!allOn && !allOff) return null;
      }
    }
    return allOn && this.widgets?.length ? true : false;
  }

  toggleAllLoras() {
    const toggledTo = !this.allLorasState();
    for (const widget of this.widgets) {
      if (widget.name?.startsWith("lora_") && widget.value?.on != null) widget.value.on = toggledTo;
    }
  }

  static setUp(comfyClass, nodeData) {
    RgthreeBaseServerNode.registerForOverride(comfyClass, nodeData, NODE_CLASS);
  }

  static onRegisteredForOverride(comfyClass, ctxClass) {
    addConnectionLayoutSupport(NODE_CLASS, app, [["Left", "Right"], ["Right", "Left"]]);
    setTimeout(() => { NODE_CLASS.category = comfyClass.category; });
  }

  getHelp() {
    return `
      <p>The ${this.type.replace("(pgc)", "")} loads multiple LoRAs with advanced controls.</p>
      <ul>
        <li><p><strong>Dry / Wet</strong> — global multiplier (0–200%) applied to all strengths.</p></li>
        <li><p><strong>Strength Cap</strong> — clamps the absolute value of every LoRA strength to this limit after wet is applied. 0 = OFF (no cap).</p></li>
        <li><p><strong>Global Blocks</strong> — per-block strength multipliers for FLUX.2 Klein (double blocks 0–7, single blocks 0–23).
            Click ⚙ Configure to edit. Applies to all LoRAs unless overridden per-LoRA.</p></li>
        <li><p><strong>⊞ icon per LoRA</strong> — click to set per-LoRA block overrides.
            Grey = using global; accent = custom blocks set. "Use Global" clears the override.</p></li>
        <li><p><strong>Σ</strong> (header) — combined effective strength (sum × wet).</p></li>
        <li><p><strong>Presets</strong> — Load / Save / Delete via the dropdown.</p></li>
      </ul>`;
  }
}

_a = PROP_LABEL_SHOW_STRENGTHS_STATIC;
CrtMagicLoraLoaderNode.title      = PGC_NODE_TYPE;
CrtMagicLoraLoaderNode.type       = PGC_NODE_TYPE;
CrtMagicLoraLoaderNode.comfyClass = PGC_NODE_TYPE;
CrtMagicLoraLoaderNode.color      = "#111111";
CrtMagicLoraLoaderNode.bgcolor    = "#000000";
CrtMagicLoraLoaderNode[_a] = {
  type: "combo",
  values: [PROP_VALUE_SHOW_STRENGTHS_SINGLE, PROP_VALUE_SHOW_STRENGTHS_SEPARATE],
};

// ---------------------------------------------------------------------------
// Register extension
// ---------------------------------------------------------------------------
const NODE_CLASS = CrtMagicLoraLoaderNode;

app.registerExtension({
  name: "crt.MagicLoraLoader",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === NODE_CLASS.type) {
      NODE_CLASS.setUp(nodeType, nodeData);
    }
  },
});

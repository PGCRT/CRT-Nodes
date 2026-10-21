import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_NAME = "CRT_OpenCodeGoThinkingDisplay";
const EVENT_NAME = "crt_opencode_go_thinking";
const STYLE_ID = "crt-opencode-go-thinking-style";

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .crt-ocg-root {
      width: 100%; height: 100%;
      display: flex; flex-direction: column;
      background: #0b0b0d; color: #e4e4e7;
      font-family: 'Inter', -apple-system, 'Segoe UI', system-ui, sans-serif;
      font-size: 12px; box-sizing: border-box;
      border: 1px solid #1c1c1f; border-radius: 6px; overflow: hidden;
    }
    .crt-ocg-head {
      display: flex; align-items: center; gap: 8px;
      padding: 6px 10px; background: #111114;
      border-bottom: 1px solid #1c1c1f;
      font-weight: 600; letter-spacing: 0.02em;
      user-select: none; flex-shrink: 0;
    }
    .crt-ocg-dot { width: 8px; height: 8px; border-radius: 50%; background: #52525b; flex-shrink: 0; }
    .crt-ocg-dot.running { background: #f59e0b; animation: crt-ocg-pulse 1s infinite ease-in-out; }
    .crt-ocg-dot.done { background: #22c55e; animation: none; }
    .crt-ocg-dot.error { background: #ef4444; animation: none; }
    @keyframes crt-ocg-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
    .crt-ocg-stats { color: #a1a1aa; font-weight: 400; font-size: 11px; margin-left: auto; white-space: nowrap; font-variant-numeric: tabular-nums; }
    .crt-ocg-size { width: 70px; accent-color: #8b5cf6; cursor: pointer; flex-shrink: 0; }
    .crt-ocg-think {
      flex: 1; margin: 0; padding: 8px 10px; overflow-y: auto; min-height: 0;
      border-left: 2px solid #8b5cf6; background: rgba(139,92,246,0.07); color: #c4b5fd;
      font-family: ui-monospace, 'Consolas', 'Monaco', monospace; font-size: 11px;
      white-space: pre-wrap; word-break: break-word;
    }
  `;
  document.head.appendChild(style);
}

function collectDisplayNodes() {
  const graph = app.graph;
  if (!graph) return [];
  const list = graph._nodes || graph.nodes || [];
  return list.filter((n) => n && (n.type === NODE_NAME || n.comfyClass === NODE_NAME));
}

function setStatus(node, mode, text) {
  if (!node._crtOcgDot || !node._crtOcgStats) return;
  node._crtOcgDot.className = `crt-ocg-dot${mode ? ` ${mode}` : ""}`;
  node._crtOcgStats.textContent = text || "";
}

function formatCost(cost) {
  if (!(cost > 0)) return "$0";
  if (cost < 0.001) return `$${cost.toFixed(6)}`;
  if (cost < 0.01) return `$${cost.toFixed(5)}`;
  if (cost < 1) return `$${cost.toFixed(4)}`;
  return `$${cost.toFixed(2)}`;
}

function formatStats(stats, done) {
  if (!stats) return done ? "done" : "thinking...";
  const parts = [];
  if (Number.isFinite(stats.tps) && stats.tps > 0) parts.push(`${stats.tps.toFixed(1)} tok/s`);
  if (Number.isFinite(stats.completion_tokens) && stats.completion_tokens > 0) {
    parts.push(`${Math.round(stats.completion_tokens)} tok`);
  }
  if (done && Number.isFinite(stats.elapsed) && stats.elapsed > 0) {
    parts.push(`${stats.elapsed.toFixed(1)}s`);
  }
  if (done && Number.isFinite(stats.cost)) parts.push(formatCost(stats.cost));
  if (!parts.length) return done ? "done" : "thinking...";
  return done ? `done - ${parts.join(" - ")}` : parts.join(" - ");
}

function applySnapshot(node, thinking, done, error, stats) {
  if (!node._crtOcgThink) return;
  if (node._crtOcgFrame) cancelAnimationFrame(node._crtOcgFrame);
  node._crtOcgFrame = requestAnimationFrame(() => {
    node._crtOcgFrame = null;
    const el = node._crtOcgThink;
    const stick = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
    el.textContent = thinking || "";
    if (error) {
      setStatus(node, "error", String(error).slice(0, 140));
    } else if (done) {
      setStatus(node, "done", formatStats(stats, true));
    } else {
      setStatus(node, "running", formatStats(stats, false));
    }
    if (!done && stick) el.scrollTop = el.scrollHeight;
    if (done) el.scrollTop = el.scrollHeight;
    node.setDirtyCanvas?.(true, true);
  });
}

function clearView(node, text) {
  if (!node._crtOcgThink) return;
  if (node._crtOcgFrame) cancelAnimationFrame(node._crtOcgFrame);
  node._crtOcgFrame = null;
  node._crtOcgThink.textContent = "";
  setStatus(node, "running", text || "starting...");
  node.setDirtyCanvas?.(true, true);
}

function routeEvent(data) {
  if (!data) return;
  for (const node of collectDisplayNodes()) {
    if (typeof node._crtOcgApply !== "function") continue;
    node._crtOcgApply(data);
  }
}

app.registerExtension({
  name: "CRTOpenCodeGo.ThinkingDisplay",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;
    ensureStyles();

    const origCreated = nodeType.prototype.onNodeCreated;
    const origRemoved = nodeType.prototype.onRemoved;
    const origExecuted = nodeType.prototype.onExecuted;
    const origSerialize = nodeType.prototype.onSerialize;
    const origConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      origCreated?.apply(this, arguments);
      this.bgcolor = "#0b0b0d";
      this.color = "#1c1c1f";
      if (!this.size || this.size[0] < 380 || this.size[1] < 200) this.size = [420, 360];

      const root = document.createElement("div");
      root.className = "crt-ocg-root";

      const head = document.createElement("div");
      head.className = "crt-ocg-head";
      const dot = document.createElement("span");
      dot.className = "crt-ocg-dot";
      const title = document.createElement("span");
      title.textContent = "Thinking";
      const stats = document.createElement("span");
      stats.className = "crt-ocg-stats";
      stats.textContent = "idle";
      const sizeKnob = document.createElement("input");
      sizeKnob.type = "range";
      sizeKnob.min = "8";
      sizeKnob.max = "28";
      sizeKnob.step = "1";
      sizeKnob.className = "crt-ocg-size";
      sizeKnob.title = "Text size";
      head.append(dot, title, stats, sizeKnob);

      const thinkPre = document.createElement("pre");
      thinkPre.className = "crt-ocg-think";
      thinkPre.textContent = "";

      root.append(head, thinkPre);
      root.addEventListener("mousedown", (e) => e.stopPropagation(), true);

      this.properties = this.properties || {};
      const savedSize = Number(this.properties.thinking_font_size) || 11;
      sizeKnob.value = String(savedSize);
      thinkPre.style.fontSize = `${savedSize}px`;
      sizeKnob.addEventListener("input", () => {
        const next = Number(sizeKnob.value) || 11;
        thinkPre.style.fontSize = `${next}px`;
        this.properties.thinking_font_size = next;
      });

      this._crtOcgDot = dot;
      this._crtOcgStats = stats;
      this._crtOcgThink = thinkPre;
      this._crtOcgSizeKnob = sizeKnob;
      this._crtOcgApply = (data) =>
        applySnapshot(
          this,
          data.thinking || "",
          Boolean(data.done),
          data.error || null,
          data.stats || null
        );

      this.addDOMWidget("crt_opencode_go_view", "thinking", root, {
        serialize: false,
        computeSize: () => [this.size?.[0] || 420, this.size?.[1] || 360],
      });
      this.setDirtyCanvas?.(true, true);
    };

    nodeType.prototype.onExecuted = function (message) {
      origExecuted?.apply(this, arguments);
      try {
        const pick = (v) => (Array.isArray(v) ? String(v[0] ?? "") : v == null ? "" : String(v));
        const thinking = pick(message?.thinking);
        if (thinking) {
          applySnapshot(this, thinking, true, null, message?.stats || null);
        } else if (this._crtOcgDot?.classList.contains("running")) {
          setStatus(this, "done", this._crtOcgStats?.textContent || "done");
        }
      } catch {
        /* ignore malformed payloads */
      }
    };

    nodeType.prototype.onRemoved = function () {
      if (this._crtOcgFrame) cancelAnimationFrame(this._crtOcgFrame);
      this._crtOcgFrame = null;
      this._crtOcgApply = null;
      origRemoved?.apply(this, arguments);
    };

    nodeType.prototype.onSerialize = function (o) {
      origSerialize?.apply(this, arguments);
      o.properties = this.properties;
    };

    nodeType.prototype.onConfigure = function (info) {
      origConfigure?.apply(this, arguments);
      this.properties = info.properties || {};
      const size = Number(this.properties.thinking_font_size) || 11;
      if (this._crtOcgThink) this._crtOcgThink.style.fontSize = `${size}px`;
      if (this._crtOcgSizeKnob) this._crtOcgSizeKnob.value = String(size);
    };
  },

  async setup() {
    ensureStyles();
    api.addEventListener(EVENT_NAME, (e) => {
      try {
        routeEvent(e.detail);
      } catch (err) {
        console.warn("[CRT OpenCode Go display] event routing failed:", err);
      }
    });
    api.addEventListener("execution_start", () => {
      for (const node of collectDisplayNodes()) clearView(node, "starting...");
    });
  },
});

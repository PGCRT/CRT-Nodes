import { app } from "../../scripts/app.js";

const STYLE_ID = "crt-line-spot-selection-style";

function ensureSelectionStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.innerHTML = `
    .comfy-multiline-input textarea::selection {
      background: rgba(255, 255, 255, 0.2) !important;
      -webkit-text-fill-color: white !important;
    }
  `;
  document.head.appendChild(style);
}

app.registerExtension({
  name: "CRT.LineSpot",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "Text Box line spot") {

      const onNodeCreated = nodeType.prototype.onNodeCreated;
      const onRemoved = nodeType.prototype.onRemoved;
      nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated?.apply(this, arguments);
        const widget = this.widgets?.find((w) => w.name === "text");
        if (!widget || !widget.inputEl) return r;

        const textarea = widget.inputEl;
        const highlighter = document.createElement("div");
        const cleanup = [];

        const commonStyle = {
          fontFamily: "monospace",
          fontSize: "14px",
          lineHeight: "1.6",
          padding: "10px",
          boxSizing: "border-box",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        };

        Object.assign(textarea.style, commonStyle);
        textarea.style.setProperty("color", "transparent", "important");
        textarea.style.background = "transparent";
        textarea.style.caretColor = "white";
        textarea.style.position = "relative";
        textarea.style.zIndex = "2";

        const update = () => {
          if (!highlighter.parentNode && textarea.parentNode) {
            textarea.parentNode.appendChild(highlighter);
          }
          if (!highlighter.parentNode) return;

          const comp = window.getComputedStyle(textarea);
          Object.assign(highlighter.style, {
            position: "absolute",
            top: textarea.offsetTop + "px",
            left: textarea.offsetLeft + "px",
            width: textarea.offsetWidth + "px",
            height: textarea.offsetHeight + "px",
            padding: comp.padding,
            fontSize: comp.fontSize,
            fontFamily: comp.fontFamily,
            lineHeight: comp.lineHeight,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            pointerEvents: "none",
            zIndex: "1",
            overflow: "hidden",
            boxSizing: "border-box",
          });

          const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
          const paragraphs = textarea.value.split("\n");
          highlighter.innerHTML = paragraphs
            .map((para, i) => {
              const color = i % 2 === 0 ? "#00FF00" : "#FF00FF";
              const safe = para ? esc(para) : "&nbsp;";
              return `<div style="color: ${color}; width: 100%; min-height: ${comp.lineHeight};">${safe}</div>`;
            })
            .join("");

          highlighter.scrollTop = textarea.scrollTop;
        };

        this._crtUpdate = update;
        this._crtTextarea = textarea;
        this._crtHighlighter = highlighter;
        this._crtStickToBottom = true;

        const isAtBottom = () =>
          Math.abs(textarea.scrollHeight - textarea.clientHeight - textarea.scrollTop) < 4;

        const snapToBottom = () => {
          textarea.scrollTop = textarea.scrollHeight;
          highlighter.scrollTop = textarea.scrollTop;
        };

        const onInput = () => {
          update();
          if (this._crtStickToBottom) snapToBottom();
        };
        const onScroll = () => {
          highlighter.scrollTop = textarea.scrollTop;
          // If user scrolls up, release stick; if back at bottom, re-stick
          this._crtStickToBottom = isAtBottom();
        };

        textarea.addEventListener("input", onInput);
        textarea.addEventListener("scroll", onScroll);
        cleanup.push(() => textarea.removeEventListener("input", onInput));
        cleanup.push(() => textarea.removeEventListener("scroll", onScroll));

        const ro = new ResizeObserver(() => {
          update();
          if (this._crtStickToBottom && textarea.scrollHeight > textarea.clientHeight) {
            textarea.scrollTop = textarea.scrollHeight;
            highlighter.scrollTop = textarea.scrollTop;
          }
        });
        ro.observe(textarea);
        cleanup.push(() => ro.disconnect());

        // Only snap to bottom on run — listen for execution start
        const onExecStart = () => { this._crtStickToBottom = true; };
        if (app.api?.addEventListener) {
          app.api.addEventListener("execution_start", onExecStart);
          cleanup.push(() => app.api.removeEventListener("execution_start", onExecStart));
        }

        ensureSelectionStyle();

        this._crtLineSpotTimer = window.setTimeout(() => {
          update();
          if (this._crtStickToBottom) {
            textarea.scrollTop = textarea.scrollHeight;
            highlighter.scrollTop = textarea.scrollTop;
          }
        }, 100);
        this._crtLineSpotCleanup = () => {
          cleanup.forEach((fn) => fn());
          cleanup.length = 0;
          if (this._crtLineSpotTimer) {
            window.clearTimeout(this._crtLineSpotTimer);
            this._crtLineSpotTimer = null;
          }
          if (this._crtLineSpotFrame) {
            cancelAnimationFrame(this._crtLineSpotFrame);
            this._crtLineSpotFrame = null;
          }
          highlighter.remove();
          this._crtUpdate = null;
        };
        return r;
      };

      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        if (!message?.text && message?.text !== "") return;
        const raw = message.text;
        const val = Array.isArray(raw) ? (raw[0] ?? "") : (raw ?? "");
        this._crtStickToBottom = true;
        const widget = this.widgets?.find((w) => w.name === "text");
        if (!widget?.inputEl) return;
        // Keep widget.value and inputEl in sync for serialization
        widget.value = val;
        widget.inputEl.value = val;
        // Instant update - no rAF wipe, rebuild highlighter synchronously
        this._crtUpdate?.();
        const ta = this._crtTextarea || widget.inputEl;
        const hl = this._crtHighlighter;
        if (ta) {
          // Force layout before scroll to avoid jitter
          void ta.offsetHeight;
          ta.scrollTop = ta.scrollHeight;
          if (hl) hl.scrollTop = ta.scrollTop;
        }
      };

      nodeType.prototype.onRemoved = function () {
        this._crtLineSpotCleanup?.();
        this._crtLineSpotCleanup = null;
        onRemoved?.apply(this, arguments);
      };

    }
  },
});

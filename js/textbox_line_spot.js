import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "CRT.LineSpot",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "Text Box line spot") {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated?.apply(this, arguments);
        const widget = this.widgets.find((w) => w.name === "text");
        if (!widget || !widget.inputEl) return r;

        const textarea = widget.inputEl;
        const highlighter = document.createElement("div");

        // Force identical style to prevent shifting
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
          // SAFETY GUARD: Check if parent exists before appending
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

          // Logical line break logic: Split by \n
          const paragraphs = textarea.value.split("\n");
          highlighter.innerHTML = paragraphs
            .map((para, i) => {
              const color = i % 2 === 0 ? "#00FF00" : "#FF00FF";
              return `<div style="color: ${color}; width: 100%; min-height: ${comp.lineHeight};">${para || "&nbsp;"}</div>`;
            })
            .join("");

          highlighter.scrollTop = textarea.scrollTop;
        };

        // Listeners
        textarea.addEventListener("input", update);
        textarea.addEventListener("scroll", () => (highlighter.scrollTop = textarea.scrollTop));

        // Sync on resize
        const ro = new ResizeObserver(() => update());
        ro.observe(textarea);

        // Selection style fix
        if (!document.getElementById("crt-selection-style")) {
          const style = document.createElement("style");
          style.id = "crt-selection-style";
          style.innerHTML = `
            .comfy-multiline-input textarea::selection {
              background: rgba(255, 255, 255, 0.2) !important;
              -webkit-text-fill-color: white !important;
            }
          `;
          document.head.appendChild(style);
        }

        setTimeout(update, 100);
        return r;
      };
    }
  },
});
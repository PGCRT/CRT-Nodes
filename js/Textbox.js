import { app } from "../../scripts/app.js";

const NODE_NAME = "CRT_Textbox";

function executionText(message) {
  const payload = message?.text;
  if (Array.isArray(payload)) return payload.length ? String(payload[0] ?? "") : "";
  return payload == null ? null : String(payload);
}

function updateTextWidget(node, value) {
  const widget = node.widgets?.find((candidate) => candidate.name === "text");
  if (!widget) return;

  widget.value = value;
  if (widget.inputEl) {
    widget.inputEl.value = value;
  }
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "CRT.TextboxDisplay",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const originalOnExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      originalOnExecuted?.apply(this, arguments);
      const value = executionText(message);
      if (value === null) return;

      if (this._crtTextboxDisplayFrame) {
        cancelAnimationFrame(this._crtTextboxDisplayFrame);
      }
      this._crtTextboxDisplayFrame = requestAnimationFrame(() => {
        updateTextWidget(this, value);
        this._crtTextboxDisplayFrame = null;
      });
    };

    const originalOnRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (this._crtTextboxDisplayFrame) {
        cancelAnimationFrame(this._crtTextboxDisplayFrame);
        this._crtTextboxDisplayFrame = null;
      }
      originalOnRemoved?.apply(this, arguments);
    };
  },
});

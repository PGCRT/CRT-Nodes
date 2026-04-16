import { app } from "/scripts/app.js";

const CRT_NODE_COLOR = "#111111";
const CRT_NODE_BGCOLOR = "#000000";

function applyBlackTheme(node) {
  if (!node) {
    return;
  }

  const color = String(node.color || "").toLowerCase();
  const bgcolor = String(node.bgcolor || "").toLowerCase();

  if (
    color === "transparent" ||
    bgcolor === "transparent" ||
    color === "rgba(0,0,0,0)" ||
    bgcolor === "rgba(0,0,0,0)"
  ) {
    return;
  }
  node.color = CRT_NODE_COLOR;
  node.bgcolor = CRT_NODE_BGCOLOR;
}

app.registerExtension({
  name: "crt.NodeBlackDefault",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData?.category?.startsWith("CRT")) {
      return;
    }

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      applyBlackTheme(this);
      return result;
    };

    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = originalOnConfigure?.apply(this, arguments);

      const isCrtOrEmpty = (v) => {
        const s = String(v || "").toLowerCase();
        return !s || s === CRT_NODE_COLOR || s === CRT_NODE_BGCOLOR || s === "transparent" || s === "rgba(0,0,0,0)";
      };

      if (isCrtOrEmpty(this.color) && isCrtOrEmpty(this.bgcolor)) {
        applyBlackTheme(this);
      }

      return result;
    };
  },
});

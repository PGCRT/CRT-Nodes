import { app } from "../../scripts/app.js";

const FX_COLOR_INPUTS = {
  AdvancedBloomFX: {
    color_filter: "#FFFFFF",
    split_shadows: "#808080",
    split_highlights: "#808080",
    smh_shadow_color: "#FFFFFF",
    smh_midtone_color: "#FFFFFF",
    smh_highlight_color: "#FFFFFF",
  },
  ColorIsolationFX: {
    target_color: "#DE827B",
  },
  ContourFX: {
    line_color: "#FFFFFF",
    background_color: "#000000",
  },
};

const FX_COLOR_ALIASES = {
  "Advanced Bloom FX (CRT)": FX_COLOR_INPUTS.AdvancedBloomFX,
  "Color Isolation FX (CRT)": FX_COLOR_INPUTS.ColorIsolationFX,
  "Contour FX (CRT)": FX_COLOR_INPUTS.ContourFX,
};

function normalizeHex(value) {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (/^#[0-9a-fA-F]{6}$/.test(trimmed)) return trimmed.toUpperCase();
  return null;
}

function nodeClassName(node) {
  return node?.comfyClass || node?.constructor?.type || node?.type || "";
}

function colorInputsForNode(node) {
  const name = nodeClassName(node);
  return FX_COLOR_INPUTS[name] || FX_COLOR_ALIASES[name] || null;
}

function chooseColorValue(entries, keeper, defaultValue) {
  const fallback = normalizeHex(defaultValue) || "#FFFFFF";
  const keeperValue = normalizeHex(keeper?.value);
  const values = entries
    .map(({ widget }) => normalizeHex(widget.value))
    .filter(Boolean);
  const duplicateDefault = values.find((value) => value === fallback);
  const nonFallbackValue = values.find(
    (value) => value !== fallback && (fallback === "#000000" || value !== "#000000") && value !== "#FF0000"
  );

  if (keeperValue && keeperValue !== "#000000") return keeperValue;
  if (duplicateDefault) return duplicateDefault;
  if (nonFallbackValue) return nonFallbackValue;
  if (keeperValue && fallback === "#000000") return keeperValue;
  return fallback;
}

function cleanupNodeColorWidgets(node) {
  const colorInputs = colorInputsForNode(node);
  if (!colorInputs || !Array.isArray(node.widgets)) return;

  let changed = false;

  for (const [name, defaultValue] of Object.entries(colorInputs)) {
    const entries = node.widgets
      .map((widget, index) => ({ widget, index }))
      .filter(({ widget }) => widget?.name === name);

    if (entries.length === 1) {
      const fallback = normalizeHex(defaultValue);
      const widgetValue = normalizeHex(entries[0].widget.value);
      if (fallback && fallback !== "#000000" && widgetValue === "#000000") {
        entries[0].widget.value = fallback;
        changed = true;
      }
      continue;
    }

    if (entries.length === 0) continue;

    const keeperEntry = entries.find(({ widget }) => widget.type !== "COLOR") || entries[0];
    const keeper = keeperEntry.widget;
    keeper.value = chooseColorValue(entries, keeper, defaultValue);

    const removeIndexes = entries
      .filter((entry) => entry !== keeperEntry)
      .map(({ index }) => index)
      .sort((a, b) => b - a);

    for (const index of removeIndexes) {
      node.widgets.splice(index, 1);
      changed = true;
    }
  }

  if (changed) {
    node.setSize?.(node.computeSize?.() || node.size);
    node.setDirtyCanvas?.(true, true);
  }
}

function deferCleanup(node) {
  cleanupNodeColorWidgets(node);
  setTimeout(() => cleanupNodeColorWidgets(node), 0);
  setTimeout(() => cleanupNodeColorWidgets(node), 100);
}

app.registerExtension({
  name: "CRT.FX.ColorWidgets",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!FX_COLOR_INPUTS[nodeData?.name]) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      deferCleanup(this);
      return result;
    };

    const originalConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function () {
      const result = originalConfigure?.apply(this, arguments);
      deferCleanup(this);
      return result;
    };
  },

  async nodeCreated(node) {
    deferCleanup(node);
  },
});

import { app } from "/scripts/app.js";

app.registerExtension({
  name: "CRT.StringSplitter",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "CRT_StringSplitter") {
      return;
    }

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      originalOnNodeCreated?.apply(this, arguments);

      const syncSize = () => {
        this.size = this.computeSize();
        this.setDirtyCanvas(true, true);
      };

      const synchronizeOutputs = () => {
        const targetCount = Math.max(1, Math.round(this.split_count ?? 1));
        const currentCount = this.outputs?.length ?? 0;

        if (currentCount < targetCount) {
          for (let index = currentCount + 1; index <= targetCount; index += 1) {
            this.addOutput(`string_${index}`, "STRING");
          }
        } else if (currentCount > targetCount) {
          for (let index = currentCount - 1; index >= targetCount; index -= 1) {
            this.removeOutput(index);
          }
        }

        syncSize();
      };

      const splitCountWidget = this.widgets?.find((widget) => widget.name === "split_count");
      if (!splitCountWidget) {
        return;
      }

      const originalSplitCountCallback = splitCountWidget.callback;
      this.split_count = Math.max(1, Math.round(splitCountWidget.value ?? 2));
      splitCountWidget.value = this.split_count;
      splitCountWidget.callback = (value, ...args) => {
        const nextValue = Math.max(1, Math.round(value ?? 1));
        splitCountWidget.value = nextValue;
        originalSplitCountCallback?.call(splitCountWidget, nextValue, ...args);

        if (this.split_count === nextValue) {
          return;
        }

        this.split_count = nextValue;
        synchronizeOutputs();
      };

      synchronizeOutputs();
    };
  },
});

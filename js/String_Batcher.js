import { app } from "/scripts/app.js";

app.registerExtension({
  name: "CRT.StringBatcher",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "CRT_StringBatcher") {
      return;
    }

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      originalOnNodeCreated?.apply(this, arguments);

      const syncSize = () => {
        this.size = this.computeSize();
        this.setDirtyCanvas(true, true);
      };

      const hideWidget = (name) => {
        const widget = this.widgets?.find((entry) => entry.name === name);
        if (widget) {
          widget.hidden = true;
        }
      };

      const synchronizeInputs = () => {
        const targetCount = Math.max(1, Math.round(this.batch_count ?? 1));
        const currentInputs = this.inputs?.filter((input) => input.name.startsWith("string_")) || [];
        const currentCount = currentInputs.length;

        if (currentCount < targetCount) {
          for (let index = currentCount + 1; index <= targetCount; index += 1) {
            this.addInput(`string_${index}`, "STRING", { multiline: true, default: "" });
          }
        } else if (currentCount > targetCount) {
          for (let index = currentCount; index > targetCount; index -= 1) {
            const slotIndex = this.findInputSlot(`string_${index}`);
            if (slotIndex !== -1) {
              this.removeInput(slotIndex);
            }
          }
        }

        syncSize();
      };

      let batchCountWidget = this.widgets?.find((widget) => widget.name === "batch_count");
      if (!batchCountWidget) {
        batchCountWidget = this.addWidget("number", "batch_count", 1, () => {}, {
          min: 1,
          max: 256,
          step: 1,
          precision: 0,
        });
      }

      const initialStringCount = this.inputs?.filter((input) => input.name.startsWith("string_")).length || 1;
      const originalBatchCountCallback = batchCountWidget.callback;

      this.batch_count = Math.max(1, Math.round(batchCountWidget.value ?? initialStringCount));
      batchCountWidget.value = this.batch_count;
      batchCountWidget.callback = (value, ...args) => {
        const nextValue = Math.max(1, Math.round(value ?? 1));
        batchCountWidget.value = nextValue;
        originalBatchCountCallback?.call(batchCountWidget, nextValue, ...args);

        if (this.batch_count === nextValue) {
          return;
        }

        this.batch_count = nextValue;
        synchronizeInputs();
      };

      synchronizeInputs();
      hideWidget("seed");
      hideWidget("control_after_generate");
      syncSize();
    };
  },
});

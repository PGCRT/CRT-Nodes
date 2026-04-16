import { app } from "/scripts/app.js";

app.registerExtension({
  name: "CRT.FileLoaderCrawlBatch",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "FileLoaderCrawlBatch") {
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
        const targetCount = Math.max(1, Math.round(this.batch_count ?? 1));

        for (let index = (this.outputs?.length ?? 0) - 1; index >= 0; index -= 1) {
          const output = this.outputs[index];
          if (!output?.name) {
            continue;
          }

          if (output.name.startsWith("text_output_") || output.name.startsWith("file_name_")) {
            this.removeOutput(index);
          }
        }

        for (let index = 1; index <= targetCount; index += 1) {
          this.addOutput(`text_output_${index}`, "STRING");
        }

        for (let index = 1; index <= targetCount; index += 1) {
          this.addOutput(`file_name_${index}`, "STRING");
        }

        syncSize();
      };

      const batchCountWidget = this.widgets?.find((widget) => widget.name === "batch_count");
      const originalBatchCountCallback = batchCountWidget?.callback;

      this.batch_count = Math.max(1, Math.round(batchCountWidget?.value ?? 1));
      if (batchCountWidget) {
        batchCountWidget.value = this.batch_count;
        batchCountWidget.callback = (value, ...args) => {
          const nextValue = Math.max(1, Math.round(value ?? 1));
          batchCountWidget.value = nextValue;
          originalBatchCountCallback?.call(batchCountWidget, nextValue, ...args);

          if (this.batch_count === nextValue) {
            return;
          }

          this.batch_count = nextValue;
          synchronizeOutputs();
        };
      }

      synchronizeOutputs();
    };
  },
});

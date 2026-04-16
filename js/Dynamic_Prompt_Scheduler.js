import { app } from "/scripts/app.js";

app.registerExtension({
  name: "CRT.DynamicPromptScheduler",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "CRT_DynamicPromptScheduler") {
      return;
    }

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      originalOnNodeCreated?.apply(this, arguments);

      const syncSize = () => {
        this.size = this.computeSize();
        this.setDirtyCanvas(true, true);
      };

      const getWidget = (name, createWidget) => {
        const widget = this.widgets?.find((entry) => entry.name === name);
        return widget ?? createWidget();
      };

      const synchronizeInputs = () => {
        const targetCount = Math.max(1, Math.round(this.batch_count ?? 1));
        const imagesEnabled = Boolean(this.images_enabled);

        for (const input of [...(this.inputs ?? [])]) {
          if (!input?.name) {
            continue;
          }

          if (input.name.startsWith("prompt_") || input.name.startsWith("image_")) {
            const slotIndex = this.findInputSlot(input.name);
            if (slotIndex !== -1) {
              this.removeInput(slotIndex);
            }
          }
        }

        for (let index = 1; index <= targetCount; index += 1) {
          this.addInput(`prompt_${index}`, "STRING", { multiline: true, default: "" });
          if (imagesEnabled) {
            this.addInput(`image_${index}`, "IMAGE");
          }
        }

        syncSize();
      };

      const batchCountWidget = getWidget("batch_count", () =>
        this.addWidget("number", "batch_count", 1, () => {}, {
          min: 1,
          max: 128,
          step: 1,
          precision: 0,
        })
      );
      const imageToggleWidget = getWidget("batch_images", () =>
        this.addWidget("toggle", "batch_images", false, () => {}, {
          on: "Enabled",
          off: "Disabled",
        })
      );

      const initialPromptCount = this.inputs?.filter((input) => input.name.startsWith("prompt_")).length || 1;
      const initialImageState = this.inputs?.some((input) => input.name.startsWith("image_")) || false;
      const originalBatchCountCallback = batchCountWidget.callback;
      const originalImageToggleCallback = imageToggleWidget.callback;

      this.batch_count = Math.max(1, Math.round(batchCountWidget.value ?? initialPromptCount));
      this.images_enabled = Boolean(imageToggleWidget.value ?? initialImageState);
      batchCountWidget.value = this.batch_count;
      imageToggleWidget.value = this.images_enabled;

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

      imageToggleWidget.callback = (value, ...args) => {
        const nextValue = Boolean(value);
        imageToggleWidget.value = nextValue;
        originalImageToggleCallback?.call(imageToggleWidget, nextValue, ...args);

        if (this.images_enabled === nextValue) {
          return;
        }

        this.images_enabled = nextValue;
        synchronizeInputs();
      };

      synchronizeInputs();
    };
  },
});

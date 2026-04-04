import { app } from "/scripts/app.js";

app.registerExtension({
	name: "CRT.DynamicPromptScheduler",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "CRT_DynamicPromptScheduler") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);

                const synchronizeInputs = () => {
                    const targetCount = this.batch_count;
                    const imagesEnabled = this.images_enabled;

                    // --- Step 1: Remove all existing dynamic inputs ---
                    // This is the only guaranteed way to ensure a correct final order.
                    // We create a copy of the array to iterate over, as removing items while iterating can cause issues.
                    const inputsToRemove = [...(this.inputs || [])];
                    for (const input of inputsToRemove) {
                        if (input.name.startsWith("prompt_") || input.name.startsWith("image_")) {
                            this.removeInput(this.findInputSlot(input.name));
                        }
                    }

                    // --- Step 2: Re-add all inputs in the correct interleaved order ---
                    for (let i = 1; i <= targetCount; i++) {
                        // Add the prompt input first
                        this.addInput(`prompt_${i}`, "STRING", { multiline: true, default: "" });

                        // If images are enabled, add the corresponding image input immediately after
                        if (imagesEnabled) {
                            this.addInput(`image_${i}`, "IMAGE");
                        }
                    }

                    this.size = this.computeSize();
                    this.setDirtyCanvas(true, true);
                };

                // --- Find or Create Widgets ---
                let batchCountWidget = this.widgets?.find(w => w.name === "batch_count");
                if (!batchCountWidget) {
                     batchCountWidget = this.addWidget(
                        "number", "batch_count", 1, () => {},
                        { min: 1, max: 128, step: 1, precision: 0 }
                    );
                }

                let imageToggleWidget = this.widgets?.find(w => w.name === "batch_images");
                if (!imageToggleWidget) {
                    imageToggleWidget = this.addWidget(
                        "toggle", "batch_images", false, () => {},
                        { on: "Enabled", off: "Disabled" }
                    );
                }

                // --- Initialize State ---
                const initialPromptCount = this.inputs?.filter(i => i.name.startsWith("prompt_")).length || 1;
                const initialImageState = this.inputs?.some(i => i.name.startsWith("image_")) || false;

                this.batch_count = batchCountWidget.value ?? initialPromptCount;
                this.images_enabled = imageToggleWidget.value ?? initialImageState;

                batchCountWidget.value = this.batch_count;
                imageToggleWidget.value = this.images_enabled;

                // --- Assign Callbacks ---
                batchCountWidget.callback = (v) => {
                    const newCount = Math.max(1, Math.round(v));
                    if (this.batch_count !== newCount) {
                        this.batch_count = newCount;
                        batchCountWidget.value = this.batch_count;
                        synchronizeInputs();
                    }
                };

                imageToggleWidget.callback = (v) => {
                    if (this.images_enabled !== v) {
                        this.images_enabled = v;
                        synchronizeInputs();
                    }
                };

                // --- Initial Build ---
                synchronizeInputs();
			};
		}
	},
});
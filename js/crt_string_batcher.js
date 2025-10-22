import { app } from "/scripts/app.js";

app.registerExtension({
	name: "CRT.StringBatcher",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "CRT_StringBatcher") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);

                // --- Logic for dynamic string inputs ---
                const synchronizeInputs = () => {
                    const targetCount = this.batch_count;
                    const currentInputs = this.inputs?.filter(i => i.name.startsWith("string_")) || [];
                    const currentCount = currentInputs.length;

                    if (currentCount < targetCount) {
                        for (let i = currentCount + 1; i <= targetCount; i++) {
                            this.addInput(`string_${i}`, "STRING", { multiline: true, default: "" });
                        }
                    } else if (currentCount > targetCount) {
                        for (let i = currentCount; i > targetCount; i--) {
                            this.removeInput(this.findInputSlot(`string_${i}`));
                        }
                    }

                    this.size = this.computeSize();
                    this.setDirtyCanvas(true, true);
                };

                let batchCountWidget = this.widgets?.find(w => w.name === "batch_count");
                if (!batchCountWidget) {
                     batchCountWidget = this.addWidget(
                        "number", "batch_count", 1, () => {},
                        { min: 1, max: 256, step: 1, precision: 0 }
                    );
                }
                
                const initialStringCount = this.inputs?.filter(i => i.name.startsWith("string_")).length || 1;
                this.batch_count = batchCountWidget.value ?? initialStringCount;
                batchCountWidget.value = this.batch_count;

                batchCountWidget.callback = (v) => {
                    const newCount = Math.max(1, Math.round(v));
                    if (this.batch_count !== newCount) {
                        this.batch_count = newCount;
                        batchCountWidget.value = this.batch_count;
                        synchronizeInputs();
                    }
                };

                synchronizeInputs();
                
                const seedWidget = this.widgets.find(w => w.name === "seed");
                if (seedWidget) {
                    seedWidget.hidden = true;
                }
                
                const controlWidget = this.widgets.find(w => w.name === "control_after_generate");
                if (controlWidget) {
                    controlWidget.hidden = true;
                }

                this.computeSize();
                this.setDirtyCanvas(true, true);
			};
		}
	},
});
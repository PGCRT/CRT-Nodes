import { app } from "/scripts/app.js";

app.registerExtension({
	name: "CRT.FileLoaderCrawlBatch",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "FileLoaderCrawlBatch") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);

				this.batch_count = 1; // Initial value

				const updateOutputs = () => {
					// Remove existing dynamic outputs
					const currentOutputs = [...this.outputs]; // Clone to iterate
					for (let i = currentOutputs.length - 1; i >= 0; i--) {
						const output = currentOutputs[i];
						if (output.name.startsWith("text_output_") || output.name.startsWith("file_name_")) {
							this.removeOutput(i);
						}
					}

					// Add new outputs based on current batch_count
					for (let i = 0; i < this.batch_count; i++) {
						this.addOutput(`text_output_${i + 1}`, "STRING");
					}
					for (let i = 0; i < this.batch_count; i++) {
						this.addOutput(`file_name_${i + 1}`, "STRING");
					}

					this.setDirtyCanvas(true, true);
					this.size = this.computeSize();
				};

				// Find the 'batch_count' widget and attach a callback
				const batchCountWidget = this.widgets.find(w => w.name === "batch_count");
				if (batchCountWidget) {
					// Store the original callback if it exists
					const originalCallback = batchCountWidget.callback;
					batchCountWidget.callback = (v) => {
						this.batch_count = v;
						updateOutputs();
						originalCallback?.(v); // Call original callback if it existed
					};
				}

				// Initial update of outputs when the node is created
				this.batch_count = batchCountWidget ? batchCountWidget.value : 1;
				updateOutputs();
			};
		}
	},
});
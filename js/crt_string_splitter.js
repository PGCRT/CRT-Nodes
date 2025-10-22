import { app } from "/scripts/app.js";

app.registerExtension({
	name: "CRT.StringSplitter",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "CRT_StringSplitter") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);

                /**
                 * Synchronizes the number of output slots with the value of the 'split_count' widget.
                 */
                const synchronizeOutputs = () => {
                    const targetCount = this.split_count;
                    const currentOutputs = this.outputs || [];
                    const currentCount = currentOutputs.length;

                    // Add outputs if the current count is less than the target
                    if (currentCount < targetCount) {
                        for (let i = currentCount; i < targetCount; i++) {
                            // Output names are 1-based for user readability (e.g., string_1, string_2)
                            this.addOutput(`string_${i + 1}`, "STRING");
                        }
                    } 
                    // Remove outputs if the current count is greater than the target
                    else if (currentCount > targetCount) {
                        for (let i = currentCount - 1; i >= targetCount; i--) {
                            this.removeOutput(i);
                        }
                    }
                    
                    // Force the node to redraw
                    this.size = this.computeSize();
                    this.setDirtyCanvas(true, true);
                };

                // Find the 'split_count' widget on the node
                const splitCountWidget = this.widgets?.find(w => w.name === "split_count");
                
                if (!splitCountWidget) {
                    console.error("[CRT_StringSplitter] Could not find split_count widget!");
                    return;
                }
                
                // Store the split count value on the node object for easy access
                this.split_count = splitCountWidget.value ?? 2;

                // Hijack the widget's callback to trigger the synchronization
                splitCountWidget.callback = (v) => {
                    // Ensure the value is a valid integer >= 1
                    const newCount = Math.max(1, Math.round(v));
                    if (this.split_count !== newCount) {
                        this.split_count = newCount;
                        // Ensure the widget's displayed value is synced with our sanitized value
                        splitCountWidget.value = this.split_count; 
                        synchronizeOutputs();
                    }
                };

                // Run once on node creation to set the initial state
                synchronizeOutputs();
			};
		}
	},
});
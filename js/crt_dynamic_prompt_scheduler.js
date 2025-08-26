import { app } from "/scripts/app.js";

app.registerExtension({
	name: "CRT.DynamicPromptScheduler",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "CRT_DynamicPromptScheduler") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);

				// --- Defensive Cleanup ---
				// Keep only the essential "clip" input and remove any old widgets/inputs from cache.
                // This is a robust way to ensure old, broken states don't persist.
				if (this.inputs && this.inputs.length > 1) {
					for (let i = this.inputs.length - 1; i > 0; i--) {
						this.removeInput(i);
					}
				}
				this.widgets = []; // Clear all widgets

				this.prompt_count = 0;

				const addPrompt = () => {
					this.prompt_count++;
					
                    // --- THE KEY CHANGE ---
					// Create an INPUT, not a widget. ComfyUI will automatically provide a
                    // textbox for it when nothing is connected.
					this.addInput(
						`prompt_${this.prompt_count}`, // The unique name the backend expects
						"STRING",
						{
							multiline: true,
							default: "",
						}
					);

					this.size = this.computeSize();
					this.setDirtyCanvas(true, true);
				};

				const removePrompt = () => {
					// Enforce a minimum of 1 prompt
					if (this.prompt_count > 1) {
                        // The last input is always at the end of the array
						this.removeInput(this.inputs.length - 1);
						this.prompt_count--;
						
						this.size = this.computeSize();
						this.setDirtyCanvas(true, true);
					}
				};

				// --- Add Control Buttons ---
				this.addWidget("BUTTON", "Add Prompt", null, addPrompt);
				this.addWidget("BUTTON", "Remove Last Prompt", null, removePrompt);

				// Add the first prompt input automatically
				addPrompt();
			};
		}
	},
});
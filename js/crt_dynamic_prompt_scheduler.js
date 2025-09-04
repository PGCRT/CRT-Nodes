import { app } from "/scripts/app.js";

app.registerExtension({
	name: "CRT.DynamicPromptScheduler",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "CRT_DynamicPromptScheduler") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);

				// --- ROBUST INITIALIZATION ---
				// We will rebuild the widgets and initialize the internal state based on
				// the inputs that ComfyUI has already loaded from the workflow.

				// 1. Clear any old widgets to prevent button duplication on reload.
				this.widgets = [];

				// 2. Find the highest existing prompt number to correctly set our counter.
				let max_prompt_id = 0;
				if (this.inputs) {
					for (const input of this.inputs) {
						// Defensive check: Ensure the input and its name are valid before processing.
						if (input && typeof input.name === "string" && input.name.startsWith("prompt_")) {
							const id = parseInt(input.name.split('_')[1], 10);
							// Check if 'id' is a valid number before comparing.
							if (!isNaN(id) && id > max_prompt_id) {
								max_prompt_id = id;
							}
						}
					}
				}
				this.prompt_count = max_prompt_id;

				// --- DEFINE NODE ACTIONS ---
				const addPrompt = () => {
					this.prompt_count++;
					this.addInput(
						`prompt_${this.prompt_count}`,
						"STRING",
						{ multiline: true, default: "" }
					);
					this.size = this.computeSize();
					this.setDirtyCanvas(true, true);
				};

				const removePrompt = () => {
					// Count how many actual prompt inputs we currently have.
					const promptInputs = this.inputs ? this.inputs.filter(i => i && typeof i.name === "string" && i.name.startsWith("prompt_")) : [];

					if (promptInputs.length > 1) {
						// Find the input with the highest ID number to remove it. This is more reliable
						// than just removing the last one in the array, as order isn't guaranteed.
						let highest_id = -1;
						let input_to_remove_index = -1;

						for (let i = 0; i < this.inputs.length; i++) {
							const input = this.inputs[i];
							if (input && typeof input.name === "string" && input.name.startsWith("prompt_")) {
								const id = parseInt(input.name.split('_')[1], 10);
								if (!isNaN(id) && id > highest_id) {
									highest_id = id;
									input_to_remove_index = i;
								}
							}
						}

						if (input_to_remove_index > -1) {
							this.removeInput(input_to_remove_index);

							// After removing, we must recalculate the new highest ID for our counter.
							let new_max_id = 0;
							if (this.inputs) {
								for (const input of this.inputs) {
									if (input && typeof input.name === "string" && input.name.startsWith("prompt_")) {
										const id = parseInt(input.name.split('_')[1], 10);
										if (!isNaN(id) && id > new_max_id) {
											new_max_id = id;
										}
									}
								}
							}
							this.prompt_count = new_max_id;
						}

						this.size = this.computeSize();
						this.setDirtyCanvas(true, true);
					}
				};

				// --- ADD CONTROLS ---
				this.addWidget("BUTTON", "Add Prompt", null, addPrompt);
				this.addWidget("BUTTON", "Remove Last Prompt", null, removePrompt);

				// --- FINAL CHECK ---
				// If after initialization we have no prompts (i.e., it's a new node), add the first one.
				if (this.prompt_count === 0) {
					addPrompt();
				}
			};
		}
	},
});
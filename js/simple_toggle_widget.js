import { app } from "../../scripts/app.js";

const SimpleToggleExtension = {
    name: "CRT.SimpleToggle.UI",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SimpleToggleNode") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                // Make node background and title transparent
                this.bgcolor = "transparent";
                this.color = "transparent";
                this.title = "";

                // Find the original boolean widget so we can read/write its value
                const valueWidget = this.widgets.find(w => w.name === "value");

                // Hide the original widget. We are replacing it with our own UI.
                valueWidget.hidden = true;

                // Initialize properties on the node itself. This is how settings are saved.
                this.properties = this.properties || {};
                this.properties.title = this.properties.title || "Toggle";
                this.properties.color_on = this.properties.color_on || "#00ff88";
                this.properties.color_off = this.properties.color_off || "#ff4444";
                this.properties.style = this.properties.style || "switch"; // switch, button, checkbox
                this.properties.on_text = this.properties.on_text || "ON";
                this.properties.off_text = this.properties.off_text || "OFF";

                // --- UI Construction ---
                const wrapper = document.createElement("div");
                wrapper.className = "simple-toggle-wrapper";
                
                const style = document.createElement("style");
                style.textContent = `
                    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&display=swap');
                    
                    .simple-toggle-wrapper { 
                        display: flex; 
                        flex-direction: column; 
                        align-items: center; 
                        justify-content: center; 
                        width: 200px; 
                        height: 120px; 
                        position: relative; 
                        top: -85px;
						left: -5px;
                        background: transparent;
                        min-width: 200px;
                        min-height: 120px;
                        max-width: 200px;
                        max-height: 120px;
                        font-family: 'Orbitron', monospace;
                        user-select: none;
                        pointer-events: none;
                    }
                    
                    .simple-toggle-header { 
                        display: flex; 
                        justify-content: flex-end; 
                        align-items: center; 
                        width: 100%; 
                        position: absolute; 
                        top: 0; 
                        padding: 0px; 
                        box-sizing: border-box; 
                        pointer-events: none;
                    }
                    
                    .simple-toggle-settings-btn { 
                        position: absolute;
                        top: 45px; 
                        left: 5px;
                        background: rgba(255, 255, 255, 0.1); 
                        color: #888; 
                        border: 1px solid #888; 
                        border-radius: 6px; 
                        width: 24px; 
                        height: 24px; 
                        cursor: pointer; 
                        font-size: 12px; 
                        transition: all 0.3s ease;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        pointer-events: auto;
                    }
                    
                    .simple-toggle-settings-btn:hover { 
                        background: #888; 
                        color: white; 
                        transform: scale(1.1);
                    }
                    
                    .simple-toggle-title { 
                        color: var(--toggle-color, #888); 
                        font-family: 'Orbitron', monospace;
                        font-size: 16px; 
                        font-weight: bold; 
                        position: absolute; 
                        bottom: -5px; 
                        text-shadow: 0 0 8px var(--toggle-color, #888);
                        pointer-events: none;
                    }
                    
                    /* Switch Style */
                    .toggle-switch {
                        width: 120px;
                        height: 60px;
                        background: linear-gradient(145deg, #2a2a2a, #1a1a1a);
                        border-radius: 30px;
                        position: relative;
                        cursor: pointer;
                        border: 3px solid var(--toggle-color, #888);
                        transition: all 0.3s ease;
                        pointer-events: auto;
                        box-shadow: 
                            inset 0 0 20px rgba(0, 0, 0, 0.8),
                            0 8px 20px rgba(0, 0, 0, 0.6);
                    }
                    
                    .toggle-switch:hover {
                        transform: scale(1.05);
                        box-shadow: 
                            inset 0 0 20px rgba(0, 0, 0, 0.8),
                            0 12px 25px rgba(0, 0, 0, 0.7),
                            0 0 15px var(--toggle-color, #888);
                    }
                    
                    .toggle-switch-handle {
                        width: 48px;
                        height: 48px;
                        background: radial-gradient(circle at 30% 30%, #ffffff, #cccccc);
                        border-radius: 50%;
                        position: absolute;
                        top: 3px;
                        left: 3px;
                        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
                        border: 2px solid var(--toggle-color, #888);
                    }
                    
                    .toggle-switch.on .toggle-switch-handle {
                        left: 63px;
                        box-shadow: 0 0 15px var(--toggle-color, #888);
                    }
                    
                    .toggle-switch-text {
                        position: absolute;
                        top: 50%;
                        transform: translateY(-50%);
                        font-size: 12px;
                        font-weight: bold;
                        color: var(--toggle-color, #888);
                        text-shadow: 0 0 5px var(--toggle-color, #888);
                        transition: all 0.3s ease;
                        pointer-events: none;
                    }
                    
                    .toggle-switch-text.on-text {
                        left: 15px;
                        opacity: 0;
                    }
                    
                    .toggle-switch-text.off-text {
                        right: 10px;
                        opacity: 1;
                    }
                    
                    .toggle-switch.on .toggle-switch-text.on-text {
                        opacity: 1;
                    }
                    
                    .toggle-switch.on .toggle-switch-text.off-text {
                        opacity: 0;
                    }
                    
                    /* Button Style */
                    .toggle-button {
                        width: 120px;
                        height: 60px;
                        background: linear-gradient(145deg, #2a2a2a, #1a1a1a);
                        border: 3px solid var(--toggle-color, #888);
                        border-radius: 15px;
                        cursor: pointer;
                        transition: all 0.3s ease;
                        pointer-events: auto;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 16px;
                        font-weight: bold;
                        color: var(--toggle-color, #888);
                        text-shadow: 0 0 8px var(--toggle-color, #888);
                        box-shadow: 
                            inset 0 0 20px rgba(0, 0, 0, 0.8),
                            0 8px 20px rgba(0, 0, 0, 0.6);
                    }
                    
                    .toggle-button:hover {
                        transform: scale(1.05);
                        box-shadow: 
                            inset 0 0 20px rgba(0, 0, 0, 0.8),
                            0 12px 25px rgba(0, 0, 0, 0.7),
                            0 0 15px var(--toggle-color, #888);
                    }
                    
                    .toggle-button.on {
                        background: linear-gradient(145deg, var(--toggle-color, #888), var(--toggle-color-dark, #666));
                        color: white;
                        text-shadow: 0 0 10px rgba(255, 255, 255, 0.8);
                    }
                    
                    /* Checkbox Style */
                    .toggle-checkbox {
                        width: 60px;
                        height: 60px;
                        background: linear-gradient(145deg, #2a2a2a, #1a1a1a);
                        border: 3px solid var(--toggle-color, #888);
                        border-radius: 10px;
                        cursor: pointer;
                        transition: all 0.3s ease;
                        pointer-events: auto;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        position: relative;
                        box-shadow: 
                            inset 0 0 20px rgba(0, 0, 0, 0.8),
                            0 8px 20px rgba(0, 0, 0, 0.6);
                    }
                    
                    .toggle-checkbox:hover {
                        transform: scale(1.05);
                        box-shadow: 
                            inset 0 0 20px rgba(0, 0, 0, 0.8),
                            0 12px 25px rgba(0, 0, 0, 0.7),
                            0 0 15px var(--toggle-color, #888);
                    }
                    
                    .toggle-checkbox-check {
                        font-size: 30px;
                        color: var(--toggle-color, #888);
                        opacity: 0;
                        transition: all 0.3s ease;
                        text-shadow: 0 0 10px var(--toggle-color, #888);
                    }
                    
                    .toggle-checkbox.on .toggle-checkbox-check {
                        opacity: 1;
                        transform: scale(1.2);
                    }
                    
                    /* Modal Styles */
                    .simple-toggle-modal-overlay { 
                        position: fixed; 
                        top: 0; 
                        left: 0; 
                        width: 100%; 
                        height: 100%; 
                        background: rgba(0,0,0,0.8); 
                        z-index: 10000; 
                        display: none; 
                        align-items: center; 
                        justify-content: center; 
                        backdrop-filter: blur(5px);
                    }
                    
                    .simple-toggle-modal { 
                        background: linear-gradient(145deg, #2a2a2a, #1a1a1a); 
                        border: 2px solid #888; 
                        border-radius: 15px; 
                        padding: 25px; 
                        display: flex; 
                        flex-direction: column; 
                        gap: 15px; 
                        min-width: 350px;
                        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.8);
                        font-family: 'Orbitron', monospace;
                    }
                    
                    .simple-toggle-modal-row { 
                        display: grid; 
                        grid-template-columns: 100px 1fr; 
                        align-items: center; 
                        gap: 15px; 
                    }
                    
                    .simple-toggle-modal-row label { 
                        color: rgba(255, 255, 255, 0.8); 
                        font-size: 12px; 
                        font-weight: 500;
                    }
                    
                    .simple-toggle-modal-row input, .simple-toggle-modal-row select { 
                        background: rgba(0, 0, 0, 0.5); 
                        border: 1px solid #888; 
                        color: white; 
                        padding: 8px 12px; 
                        border-radius: 6px; 
                        font-size: 12px;
                        font-family: 'Orbitron', monospace;
                        transition: all 0.3s ease;
                    }
                    
                    .simple-toggle-modal-row input:focus, .simple-toggle-modal-row select:focus {
                        outline: none;
                        border-color: #ffffff;
                        box-shadow: 0 0 10px #888;
                    }
                    
                    .simple-toggle-modal-buttons { 
                        display: flex; 
                        justify-content: flex-end; 
                        gap: 10px; 
                        margin-top: 10px; 
                    }
                    
                    .simple-toggle-modal-button {
                        background: rgba(255, 255, 255, 0.1);
                        border: 1px solid #888;
                        border-radius: 6px;
                        color: #888;
                        padding: 8px 16px;
                        cursor: pointer;
                        font-size: 12px;
                        font-family: 'Orbitron', monospace;
                        font-weight: 600;
                        transition: all 0.3s ease;
                        text-transform: uppercase;
                    }
                    
                    .simple-toggle-modal-button:hover {
                        background: #888;
                        color: white;
                    }
                    
                    .simple-toggle-modal-button.save {
                        background: #888;
                        color: white;
                    }
                    
                    .simple-toggle-modal-title {
                        color: #888;
                        font-size: 16px;
                        font-weight: 700;
                        text-align: center;
                        margin-bottom: 10px;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                    }
                `;
                wrapper.appendChild(style);

                const header = document.createElement("div");
                header.className = "simple-toggle-header";
                
                const settingsBtn = document.createElement("button");
                settingsBtn.className = "simple-toggle-settings-btn";
                settingsBtn.innerHTML = "⚙"; 
                
                header.appendChild(settingsBtn);

                // Create toggle element based on style
                const toggleElement = document.createElement("div");
                this.updateToggleStyle(toggleElement);
                
                const titleDisplay = document.createElement("div");
                titleDisplay.className = "simple-toggle-title";

                wrapper.appendChild(header);
                wrapper.appendChild(toggleElement);
                wrapper.appendChild(titleDisplay);
                
                // Add the custom UI as a DOMWidget
                const domWidget = this.addDOMWidget("simple_toggle_ui", "div", wrapper, {
                    serialize: false,
                    computeSize: () => [120, 50]
                });
                domWidget.wrapper = wrapper;

                // Force fixed size
                this.size = [120, 50];
                this.resizable = false;

                // --- Modal for Settings ---
                const modalOverlay = document.createElement("div");
                modalOverlay.className = "simple-toggle-modal-overlay";
                const modal = this.buildModal(modalOverlay);
                document.body.appendChild(modalOverlay);

                // --- Logic and Event Handling ---
                const updateToggleVisuals = () => {
                    const isOn = valueWidget.value;
                    const color = isOn ? this.properties.color_on : this.properties.color_off;
                    
                    wrapper.style.setProperty("--toggle-color", color);
                    
                    // Create darker variant for gradients
                    const darkColor = this.darkenColor(color, 0.3);
                    wrapper.style.setProperty("--toggle-color-dark", darkColor);
                    
                    // Update toggle state
                    if (this.properties.style === "switch") {
                        toggleElement.classList.toggle("on", isOn);
                    } else if (this.properties.style === "button") {
                        toggleElement.classList.toggle("on", isOn);
                        toggleElement.textContent = isOn ? this.properties.on_text : this.properties.off_text;
                    } else if (this.properties.style === "checkbox") {
                        toggleElement.classList.toggle("on", isOn);
                    }
                };

                const toggleValue = () => {
                    valueWidget.value = !valueWidget.value;
                    valueWidget.callback?.(valueWidget.value);
                    updateToggleVisuals();
                };

                const updateAllVisuals = () => {
                    titleDisplay.textContent = this.properties.title;
                    this.updateToggleStyle(toggleElement);
                    updateToggleVisuals();
                };
                
                // Toggle click handler
                toggleElement.addEventListener("click", (e) => {
                    toggleValue();
                    e.preventDefault();
                    e.stopPropagation();
                });

                settingsBtn.addEventListener("click", (e) => {
                    modal.show();
                    modalOverlay.style.display = "flex";
                    e.preventDefault();
                    e.stopPropagation();
                });

                updateAllVisuals(); // Initial setup

                // Store references
                this.syncToggleUI = updateAllVisuals;
                this.toggleElement = toggleElement;
            };
            
            // Helper method to update toggle style
            nodeType.prototype.updateToggleStyle = function(toggleElement) {
                // Clear existing classes
                toggleElement.className = "";
                toggleElement.innerHTML = "";
                
                if (this.properties.style === "switch") {
                    toggleElement.className = "toggle-switch";
                    
                    const handle = document.createElement("div");
                    handle.className = "toggle-switch-handle";
                    
                    const onText = document.createElement("div");
                    onText.className = "toggle-switch-text on-text";
                    onText.textContent = this.properties.on_text;
                    
                    const offText = document.createElement("div");
                    offText.className = "toggle-switch-text off-text";
                    offText.textContent = this.properties.off_text;
                    
                    toggleElement.appendChild(handle);
                    toggleElement.appendChild(onText);
                    toggleElement.appendChild(offText);
                    
                } else if (this.properties.style === "button") {
                    toggleElement.className = "toggle-button";
                    toggleElement.textContent = this.properties.off_text;
                    
                } else if (this.properties.style === "checkbox") {
                    toggleElement.className = "toggle-checkbox";
                    
                    const check = document.createElement("div");
                    check.className = "toggle-checkbox-check";
                    check.innerHTML = "✓";
                    
                    toggleElement.appendChild(check);
                }
            };
            
            // Helper method to darken colors
            nodeType.prototype.darkenColor = function(color, factor) {
                const hex = color.replace('#', '');
                const r = Math.max(0, parseInt(hex.substr(0, 2), 16) * (1 - factor));
                const g = Math.max(0, parseInt(hex.substr(2, 2), 16) * (1 - factor));
                const b = Math.max(0, parseInt(hex.substr(4, 2), 16) * (1 - factor));
                return `#${Math.round(r).toString(16).padStart(2, '0')}${Math.round(g).toString(16).padStart(2, '0')}${Math.round(b).toString(16).padStart(2, '0')}`;
            };
            
            // --- Helper to build the modal ---
            nodeType.prototype.buildModal = function(overlay) {
                const modal = document.createElement("div");
                modal.className = "simple-toggle-modal";

                const title = document.createElement("div");
                title.className = "simple-toggle-modal-title";
                title.textContent = "Toggle Settings";
                modal.appendChild(title);

                const createRow = (labelText, input) => {
                    const row = document.createElement("div"); 
                    row.className = "simple-toggle-modal-row";
                    const label = document.createElement("label"); 
                    label.textContent = labelText;
                    row.appendChild(label); 
                    row.appendChild(input); 
                    return row;
                };
                
                const titleInput = document.createElement("input");
                const onTextInput = document.createElement("input");
                const offTextInput = document.createElement("input");
                const colorOnInput = document.createElement("input"); 
                colorOnInput.type = "color";
                const colorOffInput = document.createElement("input"); 
                colorOffInput.type = "color";
                
                const styleSelect = document.createElement("select");
                ["switch", "button", "checkbox"].forEach(style => {
                    const option = document.createElement("option");
                    option.value = style;
                    option.textContent = style.charAt(0).toUpperCase() + style.slice(1);
                    styleSelect.appendChild(option);
                });

                modal.appendChild(createRow("Title:", titleInput));
                modal.appendChild(createRow("Style:", styleSelect));
                modal.appendChild(createRow("ON Text:", onTextInput));
                modal.appendChild(createRow("OFF Text:", offTextInput));
                modal.appendChild(createRow("ON Color:", colorOnInput));
                modal.appendChild(createRow("OFF Color:", colorOffInput));
                
                const buttons = document.createElement("div"); 
                buttons.className = "simple-toggle-modal-buttons";
                const saveBtn = document.createElement("button"); 
                saveBtn.className = "simple-toggle-modal-button save"; 
                saveBtn.textContent = "Save";
                const cancelBtn = document.createElement("button"); 
                cancelBtn.className = "simple-toggle-modal-button"; 
                cancelBtn.textContent = "Cancel";

                buttons.appendChild(cancelBtn); 
                buttons.appendChild(saveBtn);
                modal.appendChild(buttons);
                overlay.appendChild(modal);

                const hide = () => overlay.style.display = "none";
                const show = () => {
                    titleInput.value = this.properties.title;
                    styleSelect.value = this.properties.style;
                    onTextInput.value = this.properties.on_text;
                    offTextInput.value = this.properties.off_text;
                    colorOnInput.value = this.properties.color_on;
                    colorOffInput.value = this.properties.color_off;
                };

                cancelBtn.addEventListener("click", hide);
                overlay.addEventListener("click", (e) => { if (e.target === overlay) hide(); });
                saveBtn.addEventListener("click", () => {
                    this.properties.title = titleInput.value;
                    this.properties.style = styleSelect.value;
                    this.properties.on_text = onTextInput.value;
                    this.properties.off_text = offTextInput.value;
                    this.properties.color_on = colorOnInput.value;
                    this.properties.color_off = colorOffInput.value;
                    this.syncToggleUI();
                    hide();
                });

                return { show, hide };
            };
            
            // --- Serialization Hooks ---
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                onConfigure?.apply(this, arguments);
                if (this.syncToggleUI) {
                    this.syncToggleUI();
                }
            };
        }
    }
};

app.registerExtension(SimpleToggleExtension);
import { app } from "../../scripts/app.js";

const SimpleKnobExtension = {
    name: "CRT.SimpleKnob.UI",

    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SimpleKnobNode") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                
                this.bgcolor = "transparent";
                this.color = "transparent";
                this.title = "";

                
                const valueWidget = this.widgets.find(w => w.name === "value");

                
                valueWidget.hidden = true;

                
				this.properties = this.properties || {};
				this.properties.title = this.properties.title || "";
				this.properties.color = this.properties.color || "#7700ff";
				this.properties.min = this.properties.min ?? valueWidget.options.min ?? 0.0;
				this.properties.max = this.properties.max ?? valueWidget.options.max ?? 1.0;
				this.properties.step = 0.01;
				this.properties.precision = this.properties.precision ?? 2;
                
                const wrapper = document.createElement("div");
                wrapper.className = "simple-knob-wrapper";
                
                const style = document.createElement("style");
                style.textContent = `
                    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&display=swap');
                    
                    .simple-knob-wrapper { 
                        display: flex; 
                        flex-direction: column; 
                        align-items: center; 
                        justify-content: center; 
                        width: 240px; 
                        height: 200px; 
                        position: absolute;
                        top: -105px;
						left: -17px;
                        background: transparent;
                        min-width: 240px;
                        min-height: 200px;
                        max-width: 240px;
                        max-height: 200px;
                        font-family: 'Orbitron', monospace;
                        user-select: none;
                        pointer-events: none; 
                    }
                    .simple-knob-header { 
                        display: flex; 
                        justify-content: space-between; 
                        align-items: center; 
                        width: 100%; 
                        position: absolute; 
                        top: 0; 
                        padding: 0px; 
                        box-sizing: border-box; 
                        pointer-events: none; 
                    }
                    .simple-knob-btn { 
						position: relative;
						top: 85px;
						left: 10px;
                        background: rgba(255, 255, 255, 0.1); 
                        color: var(--knob-color, #7700ff); 
                        border: 1px solid var(--knob-color, #7700ff); 
                        border-radius: 8px; 
                        width: 28px; 
                        height: 28px; 
                        cursor: pointer; 
                        font-size: 16px; 
                        transition: all 0.3s ease;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        pointer-events: auto; 
                    }
                    .simple-knob-btn:hover { 
                        background: var(--knob-color, #7700ff); 
                        color: white; 
                        transform: scale(1.1);
                        box-shadow: 0 0 15px var(--knob-color, #7700ff);
                    }
                    .simple-knob-title { 
                        color: var(--knob-color, #7700ff); 
                        font-family: 'Orbitron', monospace;
                        align-text: center;						
                        font-size: 20px; 
                        font-weight: bold; 
                        position: absolute; 
                        bottom: -5px; 
                        text-shadow: 0 0 8px var(--knob-color, #7700ff);
                        pointer-events: none; 
                    }
                    .simple-knob-body { 
                        width: 140px; 
                        height: 140px; 
                        border-radius: 50%; 
                        background: radial-gradient(circle at 30% 30%, #2a2a2a, #0f0f0f); 
                        position: relative; 
                        cursor: grab; 
                        box-shadow: 
                            inset 0 0 20px rgba(0, 0, 0, 0.8),
                            0 8px 20px rgba(0, 0, 0, 0.6),
                            0 0 0 3px rgba(255, 255, 255, 0.1);
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        transition: all 0.3s ease;
                        overflow: hidden;
                        pointer-events: auto; 
                    }
                    .simple-knob-body:hover {
                        transform: scale(1.02);
                        box-shadow: 
                            inset 0 0 20px rgba(0, 0, 0, 0.8),
                            0 12px 25px rgba(0, 0, 0, 0.7),
                            0 0 0 3px var(--knob-color, #7700ff);
                    }
                    .simple-knob-body.dragging {
                        cursor: ns-resize !important;
                        transform: scale(1.08);
                    }
                    .simple-knob-body::before {
                        content: '';
                        position: absolute;
                        top: 10px;
                        left: 10px;
                        right: 10px;
                        bottom: 10px;
                        border-radius: 50%;
                        background: conic-gradient(
                            from 225deg,
                            transparent 0deg,
                            var(--knob-color, #7700ff) calc(var(--knob-progress, 30) * 2.7deg),
                            transparent calc(var(--knob-progress, 30) * 2.7deg + 5deg)
                        );
                        opacity: 0.8;
                    }
                    .simple-knob-indicator { 
                        width: 6px; 
                        height: 35px; 
                        background: linear-gradient(180deg, var(--knob-color, #7700ff), #ffffff); 
                        position: absolute; 
                        left: calc(50% - 3px); 
                        top: 15px; 
                        transform-origin: center 55px; 
                        border-radius: 3px; 
                        box-shadow: 0 0 10px var(--knob-color, #7700ff);
                        z-index: 2;
                        pointer-events: none; 
                    }
                    .simple-knob-center-display {
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        width: 100px;
                        height: 100px;
                        background: radial-gradient(circle, #1a1a1a, #000000);
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        border: 2px solid var(--knob-color, #7700ff);
                        z-index: 3;
                        pointer-events: none; 
                    }
                    .simple-knob-value { 
                        color: var(--knob-color, #7700ff); 
                        font-size: 14px; 
                        font-weight: 700; 
                        text-align: center;
                        text-shadow: 0 0 8px var(--knob-color, #7700ff);
                        font-family: 'Orbitron', monospace;
                        user-select: none; 
                        pointer-events: none; 
                    }
                    .simple-knob-modal-overlay { 
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
                    .simple-knob-modal { 
                        background: linear-gradient(145deg, #2a2a2a, #1a1a1a); 
                        border: 2px solid var(--knob-color, #7700ff); 
                        border-radius: 15px; 
                        padding: 25px; 
                        display: flex; 
                        flex-direction: column; 
                        gap: 15px; 
                        min-width: 350px;
                        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.8);
                        font-family: 'Orbitron', monospace;
                    }
                    .simple-knob-modal-row { 
                        display: grid; 
                        grid-template-columns: 100px 1fr; 
                        align-items: center; 
                        gap: 15px; 
                    }
                    .simple-knob-modal-row label { 
                        color: rgba(255, 255, 255, 0.8); 
                        font-size: 12px; 
                        font-weight: 500;
                    }
                    .simple-knob-modal-row input { 
                        background: rgba(0, 0, 0, 0.5); 
                        border: 1px solid var(--knob-color, #7700ff); 
                        color: white; 
                        padding: 8px 12px; 
                        border-radius: 6px; 
                        font-size: 12px;
                        font-family: 'Orbitron', monospace;
                        transition: all 0.3s ease;
                    }
                    .simple-knob-modal-row input:focus {
                        outline: none;
                        border-color: #ffffff;
                        box-shadow: 0 0 10px var(--knob-color, #7700ff);
                    }
                    .simple-knob-modal-buttons { 
                        display: flex; 
                        justify-content: flex-end; 
                        gap: 10px; 
                        margin-top: 10px; 
                    }
                    .simple-knob-modal-button {
                        background: rgba(255, 255, 255, 0.1);
                        border: 1px solid var(--knob-color, #7700ff);
                        border-radius: 6px;
                        color: var(--knob-color, #7700ff);
                        padding: 8px 16px;
                        cursor: pointer;
                        font-size: 12px;
                        font-family: 'Orbitron', monospace;
                        font-weight: 600;
                        transition: all 0.3s ease;
                        text-transform: uppercase;
                    }
                    .simple-knob-modal-button:hover {
                        background: var(--knob-color, #7700ff);
                        color: white;
                    }
                    .simple-knob-modal-button.save {
                        background: var(--knob-color, #7700ff);
                        color: white;
                    }
                    .simple-knob-modal-title {
                        color: var(--knob-color, #7700ff);
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
                header.className = "simple-knob-header";
                
                const settingsBtn = document.createElement("button");
                settingsBtn.className = "simple-knob-btn";
                settingsBtn.innerHTML = "⚙"; 
                
                header.appendChild(settingsBtn);

                const knobBody = document.createElement("div");
                knobBody.className = "simple-knob-body";
                
                const indicator = document.createElement("div");
                indicator.className = "simple-knob-indicator";
                
                const centerDisplay = document.createElement("div");
                centerDisplay.className = "simple-knob-center-display";
                
                const valueDisplay = document.createElement("div");
                valueDisplay.className = "simple-knob-value";
                
                centerDisplay.appendChild(valueDisplay);
                knobBody.appendChild(indicator);
                knobBody.appendChild(centerDisplay);
                
                const titleDisplay = document.createElement("div");
                titleDisplay.className = "simple-knob-title";

                wrapper.appendChild(header);
                wrapper.appendChild(knobBody);
                wrapper.appendChild(titleDisplay);
                
                
                const domWidget = this.addDOMWidget("simple_knob_ui", "div", wrapper, {
                    serialize: false,
                    computeSize: () => [220, 60] 
                });
                domWidget.wrapper = wrapper; 

                
                this.size = [220, 60];
                this.resizable = false;

                
                const modalOverlay = document.createElement("div");
                modalOverlay.className = "simple-knob-modal-overlay";
                const modal = this.buildModal(modalOverlay);
                document.body.appendChild(modalOverlay);

                
                let isDragging = false;
                let lastY = 0;

                const updateKnobVisuals = () => {
                    const props = this.properties;
                    const range = props.max - props.min;
                    const valuePct = (valueWidget.value - props.min) / (range || 1);
                    const deg = -135 + (valuePct * 270);
                    const progress = valuePct * 100;
                    
                    indicator.style.transform = `rotate(${deg}deg)`;
                    valueDisplay.textContent = valueWidget.value.toFixed(props.precision);
                    knobBody.style.setProperty('--knob-progress', progress);
                };

                const applyValue = (newValue) => {
                    const props = this.properties;
                    let clampedValue = Math.max(props.min, Math.min(props.max, newValue));
                    clampedValue = Math.round(clampedValue / props.step) * props.step;
                    valueWidget.value = clampedValue;
                    
                    valueWidget.callback?.(clampedValue);
                    updateKnobVisuals();
                };

                const updateAllVisuals = () => {
                    titleDisplay.textContent = this.properties.title;
                    wrapper.style.setProperty("--knob-color", this.properties.color);
                    updateKnobVisuals();
                };
                
                
                knobBody.addEventListener("mousedown", (e) => {
                    if (e.button === 0) { 
                        isDragging = true;
                        lastY = e.clientY;
                        knobBody.classList.add('dragging');
                        e.preventDefault();
                        e.stopPropagation();
                    }
                    
                });
                
                document.addEventListener("mousemove", (e) => {
                    if (!isDragging) return;
                    
                    const deltaY = lastY - e.clientY; 
                    lastY = e.clientY; 
                    
                    const range = this.properties.max - this.properties.min;
                    const sensitivity = range / 300; 
                    
                    const newValue = valueWidget.value + (deltaY * sensitivity);
                    applyValue(newValue);
                    
                    e.preventDefault();
                });
                
                document.addEventListener("mouseup", (e) => { 
                    if (isDragging) {
                        isDragging = false; 
                        knobBody.classList.remove('dragging');
                        e.preventDefault();
                    }
                });

                
                knobBody.addEventListener("wheel", (e) => {
                    
                    e.preventDefault();
                    e.stopPropagation();
                    
                    
                    const range = this.properties.max - this.properties.min;
                    const wheelSensitivity = range / 100; 
                    
                    
                    
                    const deltaValue = -e.deltaY > 0 ? wheelSensitivity : -wheelSensitivity;
                    
                    const newValue = valueWidget.value + deltaValue;
                    applyValue(newValue);
                });

                settingsBtn.addEventListener("click", (e) => {
                    modal.show();
                    modalOverlay.style.display = "flex";
                    e.preventDefault();
                    e.stopPropagation();
                });

                
                knobBody.addEventListener("contextmenu", (e) => {
                    e.preventDefault();
                });

                updateAllVisuals(); 

                
                this.syncKnobUI = updateAllVisuals; 
            };
            
            
            nodeType.prototype.buildModal = function(overlay) {
                const modal = document.createElement("div");
                modal.className = "simple-knob-modal";

                const title = document.createElement("div");
                title.className = "simple-knob-modal-title";
                title.textContent = "Knob Settings";
                modal.appendChild(title);

                const createRow = (labelText, input) => {
                    const row = document.createElement("div"); row.className = "simple-knob-modal-row";
                    const label = document.createElement("label"); label.textContent = labelText;
                    row.appendChild(label); row.appendChild(input); return row;
                };
                
                const titleInput = document.createElement("input");
                const minInput = document.createElement("input"); minInput.type = "number";
                const maxInput = document.createElement("input"); maxInput.type = "number";
                const stepInput = document.createElement("input"); stepInput.type = "number";
                const precisionInput = document.createElement("input"); precisionInput.type = "number";
                const colorInput = document.createElement("input"); colorInput.type = "color";

                modal.appendChild(createRow("Title:", titleInput));
                modal.appendChild(createRow("Min:", minInput));
                modal.appendChild(createRow("Max:", maxInput));
                modal.appendChild(createRow("Step:", stepInput));
                modal.appendChild(createRow("Precision:", precisionInput));
                modal.appendChild(createRow("Color:", colorInput));
                
                const buttons = document.createElement("div"); buttons.className = "simple-knob-modal-buttons";
                const saveBtn = document.createElement("button"); saveBtn.className = "simple-knob-modal-button save"; saveBtn.textContent = "Save";
                const cancelBtn = document.createElement("button"); cancelBtn.className = "simple-knob-modal-button"; cancelBtn.textContent = "Cancel";

                buttons.appendChild(cancelBtn); buttons.appendChild(saveBtn);
                modal.appendChild(buttons);
                overlay.appendChild(modal);

                const hide = () => overlay.style.display = "none";
                const show = () => {
                    titleInput.value = this.properties.title;
                    minInput.value = this.properties.min;
                    maxInput.value = this.properties.max;
                    stepInput.value = this.properties.step;
                    precisionInput.value = this.properties.precision;
                    colorInput.value = this.properties.color;
                };

                cancelBtn.addEventListener("click", hide);
                overlay.addEventListener("click", (e) => { if (e.target === overlay) hide(); });
                saveBtn.addEventListener("click", () => {
                    this.properties.title = titleInput.value;
                    this.properties.min = parseFloat(minInput.value);
                    this.properties.max = parseFloat(maxInput.value);
                    this.properties.step = parseFloat(stepInput.value);
                    this.properties.precision = parseInt(precisionInput.value);
                    this.properties.color = colorInput.value;
                    this.syncKnobUI();
                    hide();
                });

                return { show, hide };
            };
            
            
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                onConfigure?.apply(this, arguments);
                if (this.syncKnobUI) {
                    this.syncKnobUI();
                }
            };
        }
    }
};

app.registerExtension(SimpleKnobExtension);
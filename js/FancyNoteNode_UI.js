import { app } from "/scripts/app.js";

const FancyNoteNodeExtension = {
    name: "FancyNoteNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FancyNoteNode") {
            const style = document.createElement("style");
            style.innerText = `
                @keyframes fancy-text-glow-pulse {
                    0%, 100% {
                        text-shadow: 0 0 25px var(--glow-color, #7300ff);
                    }
                    50% {
                        text-shadow: 0 0 35px var(--glow-color, #7300ff),
                    }
                }

                /* Keyframes for controls entrance (if desired - using class transition below for simplicity) */
                /*
                @keyframes fancy-controls-enter {
                    from {
                        opacity: 0;
                        transform: translateY(-10px) translateX(-50%);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0) translateX(-50%);
                    }
                }
                */

                .fancy-note-textarea {
                    text-align: center;
                    width: 100%;
                    height: 200%;
                    position: fixed;
                    top: -100%;
                    left: 0%;
                    resize: none;
                    background: transparent ;
                    border-style: none;
                    color: var(--text-color, #7300ff);
                    font-family: 'Orbitron', monospace;
                    /* text-shadow: 0 0 25px var(--glow-color, #7300ff); Applied by animation */
                    padding: 0px;
                    box-sizing: border-box;
                    border-radius: 0px;
                    outline: none;
                    margin: 0px;
                    overflow-y: auto;
                    transform: translateY(0%);

                    /* ANIMATION & TRANSITION ADDITIONS */
                    animation: fancy-text-glow-pulse 10s infinite ease-in-out;
                    transition: font-size 0.3s ease-out, 
                                color 10s ease-in-out, 
                                text-shadow 4s ease-in-out; /* For smooth color changes affecting shadow */
                }
                
                .fancy-note-controls {
                    display: flex;
                    gap: 4px;
                    align-items: center;
                    width: 0%; /* Span full width of parent - Preserved from original */
                    height: auto; /* Let content and padding define height */
                    padding: 0px; /* Preserved from original */
                    flex-shrink: 0; 
                    justify-content: center;
                    position: absolute; 
                    top: calc(-100% + -60px);
                    left: 50%; /* Preserved from original, works with width: 0% and justify-content */
                    background: rgba(0, 0, 0, 0.1);
                    border-radius: 0px; /* Preserved from original */
                    z-index: 1003;
                    box-sizing: border-box;

                    /* ANIMATION & TRANSITION ADDITIONS */
                    opacity: 1; /* Default visible state */
                    /* transform: translateX(-50%); If centered with width:auto, but with width:0 it's not strictly needed */
                    /* The translateX(-50%) is applied to ensure centering if width were auto. With width:0, it has little effect on the element itself. */
                    transition: opacity 0.4s ease-out, transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                }

                .fancy-note-controls.fnc-hidden-initial { /* Class for initial hidden state */
                    opacity: 0;
                    transform: translateY(-15px) translateX(-50%); /* translateX to keep consistent with left:50% */
                }
                
                .fancy-note-slider {
                    width: 120px;
                    height: 3px;
                    background: linear-gradient(90deg, rgba(170, 0, 255, 0.3), rgba(170, 0, 255, 0.1));
                    border-radius: 3px;
                    box-shadow: 0 0 10px var(--glow-color, #7300ff);
                    -webkit-appearance: none;
                    outline: none;
                    flex-shrink: 0;

                    /* TRANSITION ADDITIONS */
                    transition: box-shadow 0.2s ease-in-out;
                }
                .fancy-note-slider:hover {
                    box-shadow: 0 0 15px var(--glow-color, #7300ff), 0 0 5px var(--glow-color, #7300ff);
                }
                
                .fancy-note-slider::-webkit-slider-thumb {
                    -webkit-appearance: none;
                    width: 10px; /* Preserved from original */
                    height: 10px; /* Preserved from original */
                    background: var(--accent-color, #7300ff);
                    border-radius: 50%;
                    box-shadow: 0 0 10px var(--glow-color, #7300ff);
                    cursor: pointer;
                    border: none;

                    /* TRANSITION ADDITIONS */
                    transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.3s ease;
                }
                .fancy-note-slider::-webkit-slider-thumb:hover {
                    transform: scale(1.2); /* Kept from previous suggestion - subtle */
                    box-shadow: 0 0 15px var(--glow-color, #7300ff);
                }
                
                .fancy-note-color-button {
                    width: 20px; /* Preserved from original */
                    height: 20px; /* Preserved from original */
                    background: var(--accent-color, #7300ff);
                    border: none;
                    border-radius: 50%;
                    box-shadow: 0 0 10px var(--glow-color, #7300ff);
                    cursor: pointer;
                    /* transition: transform 0.2s ease; Original had this */
                    flex-shrink: 0;

                    /* TRANSITION ADDITIONS (enhancing original) */
                    transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.3s ease;
                }
                
                .fancy-note-color-button:hover {
                    transform: scale(1.1); /* Preserved from original */
                    box-shadow: 0 0 12px var(--glow-color, #7300ff); /* Slightly enhanced original hover */
                }
                
                .fancy-note-color-input {
                    display: none;
                }
                
                .fancy-note-textarea::-webkit-scrollbar { width: 8px; }
                .fancy-note-textarea::-webkit-scrollbar-track { background: transparent; }
                .fancy-note-textarea::-webkit-scrollbar-thumb {
                    background: var(--accent-color, #7300ff);
                    border-radius: 0px;
                    box-shadow: 0 0 20px var(--glow-color, #7300ff);
                    /* TRANSITION ADDITION */
                    transition: background-color 0.3s ease;
                }
                
                /* Preserved original node structure CSS */
                .litegraph .graph-node[data-type="FancyNoteNode"] {
                    display: flex; flex-direction: column ; min-height: 0px; overflow: hidden;
                }
                .litegraph .graph-node[data-type="FancyNoteNode"] .node-content {
                    display: flex; flex-direction: row; height: 0%; flex: 1; padding: 0;
                }
            `;
            document.head.appendChild(style);

            nodeType.prototype.onNodeCreated = function () {
                const node = this;
               node.bgcolor = "#000000";
                node.color = "#000000";
                node.title_style = "color: #000000 ;";

                const textWidget = node.widgets?.find((w) => w.name === "text");
                if (!textWidget) {
                    console.error("Text widget not found");
                    node.widgets = node.widgets || [];
                    return;
                }

                textWidget.serialize = true;

                node.properties = node.properties || {};
                node.properties.ui_font_size = node.properties.ui_font_size || 80;
                node.properties.ui_text_color = node.properties.ui_text_color || "#7300ff";
                node.properties.ui_glow_color = node.properties.ui_glow_color || "#7300ff";
                node.properties.ui_accent_color = node.properties.ui_accent_color || "#7300ff";
                node.properties.ui_glow_color_intensified = LightenDarkenColor(node.properties.ui_glow_color, 30);

                if (node.widgets) {
                    node.widgets.forEach((w) => (w.hidden = true));
                }

                const container = document.createElement("div");
                container.className = "fancy-note-container";

                const controls = document.createElement("div");
                controls.className = "fancy-note-controls fnc-hidden-initial";

                const fontSizeSlider = document.createElement("input");
                fontSizeSlider.type = "range";
                fontSizeSlider.min = "8";
                fontSizeSlider.max = "250";
                fontSizeSlider.value = node.properties.ui_font_size;
                fontSizeSlider.className = "fancy-note-slider";

                const colorButton = document.createElement("button");
                colorButton.className = "fancy-note-color-button";
                colorButton.title = "Change color";

                const colorInput = document.createElement("input");
                colorInput.type = "color";
                colorInput.className = "fancy-note-color-input";

                const textarea = document.createElement("textarea");
                textarea.className = "fancy-note-textarea";
                textarea.value = textWidget.value || "";
                textarea.style.fontSize = `${node.properties.ui_font_size}px`;
                textarea.spellcheck = false;
                textarea.setAttribute("autocorrect", "off");
                textarea.setAttribute("autocapitalize", "off");
                textarea.placeholder = "XXX";

                function LightenDarkenColor(col, amt) {
                    col = col.startsWith('#') ? col.substring(1) : col;
                    let usePound = col.length === 6 || col.length === 3;
                    if (col.length === 3) { col = col[0]+col[0]+col[1]+col[1]+col[2]+col[2]; }
                    const num = parseInt(col,16);
                    let r = (num >> 16) + amt; if (r > 255) r = 255; else if  (r < 0) r = 0;
                    let b = ((num >> 8) & 0x00FF) + amt; if (b > 255) b = 255; else if (b < 0) b = 0;
                    let g = (num & 0x0000FF) + amt; if (g > 255) g = 255; else if (g < 0) g = 0;
                    const newColor = (g | (b << 8) | (r << 16)).toString(16).padStart(6, '0');
                    return (usePound?"#":"") + newColor;
                }

                const updateTheme = (color) => {
                    const intensifiedGlow = LightenDarkenColor(color, 40);

                    container.style.setProperty("--text-color", color);
                    container.style.setProperty("--glow-color", color);
                    container.style.setProperty("--glow-color-intensified", intensifiedGlow);
                    container.style.setProperty("--accent-color", color);
                    
                    node.properties.ui_text_color = color;
                    node.properties.ui_glow_color = color;
                    node.properties.ui_glow_color_intensified = intensifiedGlow;
                    node.properties.ui_accent_color = color;
                    
                    colorButton.style.background = color;
                    colorInput.value = color;
                    node.setDirtyCanvas(true);
                };
                updateTheme(node.properties.ui_text_color);

                fontSizeSlider.addEventListener("input", () => {
                    textarea.style.fontSize = `${fontSizeSlider.value}px`;
                    node.properties.ui_font_size = parseInt(fontSizeSlider.value);
                });

                colorButton.addEventListener("click", (e) => {
                    e.preventDefault(); e.stopPropagation(); colorInput.click();
                });

                colorInput.addEventListener("input", () => {
                    updateTheme(colorInput.value);
                });

                textarea.addEventListener("input", () => {
                    textWidget.value = textarea.value;
                    node.properties.text = textarea.value;
                });

                textarea.addEventListener("mousedown", (e) => e.stopPropagation());

                controls.appendChild(fontSizeSlider);
                controls.appendChild(colorButton);
                controls.appendChild(colorInput);
                container.appendChild(controls);
                container.appendChild(textarea);
                node.container = container;

                node.addDOMWidget("fancyNote", "Fancy Note", container, {
                    serialize: false,
                    computeSize: () => {
                        const width = Math.max(node.size[0] || 300, 200);
                        const height = Math.max(node.size[1] || 150, 200);
                        return [width, height];
                    },
                });

                if (!document.querySelector('link[href*="Orbitron"]')) {
                    const fontLink = document.createElement("link");
                    fontLink.href = "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap";
                    fontLink.rel = "stylesheet";
                    document.head.appendChild(fontLink);
                }
                
                setTimeout(() => {
                    controls.classList.remove("fnc-hidden-initial");
                }, 50);

                node.syncUIWithState();
            };

            nodeType.prototype.syncUIWithState = function () {
                if (!this.container) {
                    console.warn("Container not found for UI sync");
                    return;
                }

                const textWidget = this.widgets?.find((w) => w.name === "text");
                if (!textWidget) {
                    console.error("Text widget not found for UI sync");
                    return;
                }

                const textarea = this.container.querySelector(".fancy-note-textarea");
                const fontSizeSlider = this.container.querySelector(".fancy-note-slider");
                const colorInput = this.container.querySelector(".fancy-note-color-input");
                const colorButton = this.container.querySelector(".fancy-note-color-button");

                if (textarea) {
                    textarea.value = textWidget.value || this.properties.text || "";
                }

                if (fontSizeSlider) {
                    fontSizeSlider.value = this.properties.ui_font_size || 80;
                    if (textarea) {
                        textarea.style.fontSize = `${this.properties.ui_font_size}px`;
                    }
                }

                if (colorInput && colorButton) {
                    const color = this.properties.ui_text_color || "#7300ff";
                    const intensifiedGlow = this.properties.ui_glow_color_intensified || LightenDarkenColor(color, 40);
                    this.container.style.setProperty("--text-color", color);
                    this.container.style.setProperty("--glow-color", this.properties.ui_glow_color || color);
                    this.container.style.setProperty("--glow-color-intensified", intensifiedGlow);
                    this.container.style.setProperty("--accent-color", this.properties.ui_accent_color || color);
                    colorInput.value = color;
                    colorButton.style.background = color;
                }

                this.setDirtyCanvas(true);
            };

            nodeType.prototype.onSerialize = function (info) {
                info.properties = {
                    ui_font_size: this.properties.ui_font_size || 80,
                    ui_text_color: this.properties.ui_text_color || "#7300ff",
                    ui_glow_color: this.properties.ui_glow_color || "#7300ff",
                    ui_accent_color: this.properties.ui_accent_color || "#7300ff",
                    ui_glow_color_intensified: this.properties.ui_glow_color_intensified || LightenDarkenColor(this.properties.ui_glow_color || "#7300ff", 40),
                    text: this.properties.text || ""
                };

                const textWidget = this.widgets?.find((w) => w.name === "text");
                if (textWidget) {
                    textWidget.value = this.properties.text || "";
                    info.widgets_values = [textWidget.value];
                } else {
                    info.widgets_values = [this.properties.text || ""];
                }
            };

            nodeType.prototype.onConfigure = function (info) {
                this.properties = this.properties || {};

                if (info.properties) {
                    this.properties.ui_font_size = info.properties.ui_font_size || 80;
                    this.properties.ui_text_color = info.properties.ui_text_color || "#7300ff";
                    this.properties.ui_glow_color = info.properties.ui_glow_color || "#7300ff";
                    this.properties.ui_accent_color = info.properties.ui_accent_color || "#7300ff";
                    this.properties.ui_glow_color_intensified = info.properties.ui_glow_color_intensified || LightenDarkenColor(this.properties.ui_glow_color || "#7300ff", 40);
                    this.properties.text = info.properties.text || "";
                }

                const textWidget = this.widgets?.find((w) => w.name === "text");
                if (textWidget) {
                    textWidget.value = info.widgets_values?.[0] || this.properties.text || "";
                    this.properties.text = textWidget.value;
                }

                if (this.container) {
                    this.syncUIWithState();
                }
            };
        }
    },
};

app.registerExtension(FancyNoteNodeExtension);
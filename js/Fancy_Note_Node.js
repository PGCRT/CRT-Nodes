import { app } from "/scripts/app.js";

const STYLE_ID = "crt-fancy-note-style";
const FONT_ID = "crt-fancy-note-font";

function lightenDarkenColor(col, amt) {
    col = col.startsWith("#") ? col.substring(1) : col;
    const usePound = col.length === 6 || col.length === 3;
    if (col.length === 3) {
        col = `${col[0]}${col[0]}${col[1]}${col[1]}${col[2]}${col[2]}`;
    }
    const num = parseInt(col, 16);
    let r = (num >> 16) + amt;
    let b = ((num >> 8) & 0x00FF) + amt;
    let g = (num & 0x0000FF) + amt;
    r = Math.max(0, Math.min(255, r));
    b = Math.max(0, Math.min(255, b));
    g = Math.max(0, Math.min(255, g));
    const newColor = (g | (b << 8) | (r << 16)).toString(16).padStart(6, "0");
    return `${usePound ? "#" : ""}${newColor}`;
}

function ensureAssets() {
    if (!document.getElementById(STYLE_ID)) {
        const style = document.createElement("style");
        style.id = STYLE_ID;
        style.innerText = `
            @keyframes fancy-text-glow-pulse {
                0%, 100% {
                    text-shadow: 0 0 25px var(--glow-color, #7300ff);
                }
                50% {
                    text-shadow: 0 0 35px var(--glow-color, #7300ff);
                }
            }

            .fancy-note-textarea {
                text-align: center;
                width: 100%;
                height: 100%;
                resize: none;
                background: transparent;
                border-style: none;
                color: var(--text-color, #7300ff);
                font-family: 'Orbitron', monospace;
                padding: 0px;
                box-sizing: border-box;
                border-radius: 0px;
                outline: none;
                margin: 0px;
                overflow-y: auto;
                animation: fancy-text-glow-pulse 10s infinite ease-in-out;
                transition: font-size 0.3s ease-out,
                            color 10s ease-in-out,
                            text-shadow 4s ease-in-out;
            }

            .fancy-note-controls {
                display: flex;
                gap: 4px;
                align-items: center;
                justify-content: center;
                padding: 0px;
                position: absolute;
                top: -35px;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(0, 0, 0, 0.1);
                border-radius: 0px;
                z-index: 1003;
                box-sizing: border-box;
                opacity: 1;
                transition: opacity 0.4s ease-out, transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            }

            .fancy-note-controls.fnc-hidden-initial {
                opacity: 0;
                transform: translateY(-15px) translateX(-50%);
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
                transition: box-shadow 0.2s ease-in-out;
            }
            .fancy-note-slider:hover {
                box-shadow: 0 0 15px var(--glow-color, #7300ff), 0 0 5px var(--glow-color, #7300ff);
            }

            .fancy-note-slider::-webkit-slider-thumb {
                -webkit-appearance: none;
                width: 10px;
                height: 10px;
                background: var(--accent-color, #7300ff);
                border-radius: 50%;
                box-shadow: 0 0 10px var(--glow-color, #7300ff);
                cursor: pointer;
                border: none;
                transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.3s ease;
            }
            .fancy-note-slider::-webkit-slider-thumb:hover {
                transform: scale(1.2);
                box-shadow: 0 0 15px var(--glow-color, #7300ff);
            }

            .fancy-note-color-button {
                width: 20px;
                height: 20px;
                background: var(--accent-color, #7300ff);
                border: none;
                border-radius: 50%;
                box-shadow: 0 0 10px var(--glow-color, #7300ff);
                cursor: pointer;
                flex-shrink: 0;
                transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.3s ease;
            }

            .fancy-note-color-button:hover {
                transform: scale(1.1);
                box-shadow: 0 0 12px var(--glow-color, #7300ff);
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
                transition: background-color 0.3s ease;
            }

            .litegraph .graph-node[data-type="FancyNoteNode"] {
                display: flex; flex-direction: column; min-height: 0px; overflow: hidden;
            }
            .litegraph .graph-node[data-type="FancyNoteNode"] .node-content {
                display: flex; flex-direction: row; height: 0%; flex: 1; padding: 0;
            }
        `;
        document.head.appendChild(style);
    }

    if (!document.getElementById(FONT_ID)) {
        const fontLink = document.createElement("link");
        fontLink.id = FONT_ID;
        fontLink.href = "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap";
        fontLink.rel = "stylesheet";
        document.head.appendChild(fontLink);
    }
}

const FancyNoteNodeExtension = {
    name: "FancyNoteNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FancyNoteNode") {
            ensureAssets();
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            const originalOnRemoved = nodeType.prototype.onRemoved;
            const originalOnSerialize = nodeType.prototype.onSerialize;
            const originalOnConfigure = nodeType.prototype.onConfigure;

            nodeType.prototype.onNodeCreated = function () {
                originalOnNodeCreated?.apply(this, arguments);

                const node = this;
                node.bgcolor = "#000000";
                node.color = "#000000";
                node.title_style = "color: #000000;";

                const textWidget = node.widgets?.find((w) => w.name === "text");
                if (!textWidget) {
                    return;
                }

                textWidget.serialize = true;

                node.properties = node.properties || {};
                node.properties.ui_font_size = node.properties.ui_font_size || 80;
                node.properties.ui_text_color = node.properties.ui_text_color || "#7300ff";
                node.properties.ui_glow_color = node.properties.ui_glow_color || "#7300ff";
                node.properties.ui_accent_color = node.properties.ui_accent_color || "#7300ff";
                node.properties.ui_glow_color_intensified = lightenDarkenColor(node.properties.ui_glow_color, 30);

                if (node.widgets) {
                    node.widgets.forEach((w) => {
                        w.hidden = true;
                        w.computeSize = () => [0, -4];
                    });
                }

                const container = document.createElement("div");
                container.className = "fancy-note-container";
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.position = "relative";
                container.style.display = "flex";
                container.style.flexDirection = "column";

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

                const updateTheme = (color) => {
                    const intensifiedGlow = lightenDarkenColor(color, 40);

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
                        const width = Math.max(node.size[0] || 200, 200);
                        const height = Math.max(node.size[1] || 50, 50);
                        return [width, height];
                    },
                });

                node._fancyNoteRevealTimeout = window.setTimeout(() => {
                    controls.classList.remove("fnc-hidden-initial");
                }, 50);

                node.syncUIWithState();
            };

            nodeType.prototype.syncUIWithState = function () {
                if (!this.container) {
                    return;
                }

                const textWidget = this.widgets?.find((w) => w.name === "text");
                if (!textWidget) {
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
                    const intensifiedGlow = this.properties.ui_glow_color_intensified || lightenDarkenColor(color, 40);
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
                originalOnSerialize?.apply(this, arguments);

                info.properties = {
                    ui_font_size: this.properties.ui_font_size || 80,
                    ui_text_color: this.properties.ui_text_color || "#7300ff",
                    ui_glow_color: this.properties.ui_glow_color || "#7300ff",
                    ui_accent_color: this.properties.ui_accent_color || "#7300ff",
                    ui_glow_color_intensified: this.properties.ui_glow_color_intensified || lightenDarkenColor(this.properties.ui_glow_color || "#7300ff", 40),
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
                originalOnConfigure?.apply(this, arguments);

                this.properties = this.properties || {};

                if (info.properties) {
                    this.properties.ui_font_size = info.properties.ui_font_size || 80;
                    this.properties.ui_text_color = info.properties.ui_text_color || "#7300ff";
                    this.properties.ui_glow_color = info.properties.ui_glow_color || "#7300ff";
                    this.properties.ui_accent_color = info.properties.ui_accent_color || "#7300ff";
                    this.properties.ui_glow_color_intensified = info.properties.ui_glow_color_intensified || lightenDarkenColor(this.properties.ui_glow_color || "#7300ff", 40);
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

            nodeType.prototype.onRemoved = function () {
                if (this._fancyNoteRevealTimeout) {
                    window.clearTimeout(this._fancyNoteRevealTimeout);
                    this._fancyNoteRevealTimeout = null;
                }
                this.container = null;
                originalOnRemoved?.apply(this, arguments);
            };
        }
    },
};

app.registerExtension(FancyNoteNodeExtension);

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const STYLE_ID = "crt-fancy-timer-style";
const FONT_ID = "crt-fancy-timer-font";
let eventsBound = false;

// --- The Global Timer Manager ---
const GlobalTimer = {
    startTime: 0,
    intervalId: null,
    isRunning: false,
    activeNodes: new Set(),

    formatTime(ms) {
        if (ms < 0) ms = 0;
        const minutes     = String(Math.floor(ms / 60000)).padStart(2, '0');
        const seconds     = String(Math.floor((ms % 60000) / 1000)).padStart(2, '0');
        const milliseconds = String(ms % 1000).padStart(3, '0');
        return { str: `${minutes}:${seconds}:${milliseconds}`, minutes, seconds, milliseconds };
    },

    setDisplay(node, time) {
        if (!node.timerDisplay) return;
        if (node._segMin) {
            node._segMin.textContent = time.minutes;
            node._segSec.textContent = time.seconds;
            node._segMs.textContent  = time.milliseconds;
        } else {
            node.timerDisplay.textContent = time.str;
        }
    },
    
    start() {
        if (this.isRunning) return;
        this.isRunning = true;
        this.startTime = Date.now();
        
        this.activeNodes.forEach(node => {
            if (node.timerDisplay) node.timerDisplay.style.color = '#aaffaa';
        });

        this.intervalId = setInterval(() => {
            const elapsed = Date.now() - this.startTime;
            const time = this.formatTime(elapsed);
            this.activeNodes.forEach(node => this.setDisplay(node, time));
        }, 33);
    },

    stop() {
        if (!this.isRunning) return;
        this.isRunning = false;
        clearInterval(this.intervalId);
        
        const finalTime = this.formatTime(Date.now() - this.startTime);

        this.activeNodes.forEach(node => {
            if (node.timerDisplay) {
                this.setDisplay(node, finalTime);
                node.timerDisplay.style.color = 'var(--text-color, #7300ff)';
            }
            node.properties.elapsed_time_str = finalTime.str;
        });
    },

    registerNode(node) { this.activeNodes.add(node); },
    unregisterNode(node) { this.activeNodes.delete(node); },
};

// --- ComfyUI Extension Definition ---
const FancyTimerNodeExtension = {
    name: "FancyTimerNode",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FancyTimerNode") {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            const originalOnRemoved = nodeType.prototype.onRemoved;
            const originalOnSerialize = nodeType.prototype.onSerialize;
            const originalOnConfigure = nodeType.prototype.onConfigure;
            
            nodeType.prototype.onNodeCreated = function () {
                originalOnNodeCreated?.apply(this, arguments);
                this.bgcolor = "#000000";
                this.color = "#000000";
                this.title = "Execution Timer";
                this.properties = this.properties || {};
                this.size = [420, 130];

                const container = document.createElement("div");
                container.style.cssText = `width: 100%; height: 100%; position: relative; --text-color: #7300ff; --glow-color: #7300ff;`;

                this.timerDisplay = document.createElement("div");
                this.timerDisplay.className = "fancy-timer-display";

                // Fixed-width segments so colons never shift
                this._segMin   = document.createElement("span");
                this._segMin.className = "fancy-timer-seg";
                const _col1    = document.createElement("span");
                _col1.className = "fancy-timer-sep";
                _col1.textContent = ":";
                this._segSec   = document.createElement("span");
                this._segSec.className = "fancy-timer-seg";
                const _col2    = document.createElement("span");
                _col2.className = "fancy-timer-sep";
                _col2.textContent = ":";
                this._segMs    = document.createElement("span");
                this._segMs.className = "fancy-timer-seg fancy-timer-seg-ms";
                this.timerDisplay.append(this._segMin, _col1, this._segSec, _col2, this._segMs);

                const saved = (this.properties.elapsed_time_str || "00:00:000").split(":");
                this._segMin.textContent = saved[0] || "00";
                this._segSec.textContent = saved[1] || "00";
                this._segMs.textContent  = saved[2] || "000";
                
                container.appendChild(this.timerDisplay);
                this.addDOMWidget("fancyTimer", "Fancy Timer", container, { serialize: false });
                
                GlobalTimer.registerNode(this);
            };
            
            nodeType.prototype.onRemoved = function() {
                GlobalTimer.unregisterNode(this);
                this.timerDisplay = null;
                originalOnRemoved?.apply(this, arguments);
            };
            nodeType.prototype.onSerialize = function(o) {
                originalOnSerialize?.apply(this, arguments);
                o.properties = this.properties;
            };
            nodeType.prototype.onConfigure = function(info) {
                originalOnConfigure?.apply(this, arguments);
                this.properties = info.properties || {};
                if (this._segMin) {
                    const parts = (this.properties.elapsed_time_str || "00:00:000").split(":");
                    this._segMin.textContent = parts[0] || "00";
                    this._segSec.textContent = parts[1] || "00";
                    this._segMs.textContent  = parts[2] || "000";
                }
            };
        }
    },
    
    setup() {
        if (!document.getElementById(STYLE_ID)) {
            const style = document.createElement("style");
            style.id = STYLE_ID;
            style.innerText = `
                @keyframes fancy-text-glow-pulse {
                    0%, 100% { text-shadow: 0 0 25px var(--glow-color); }
                    50% { text-shadow: 0 0 35px var(--glow-color); }
                }
                .fancy-timer-display {
                    text-align: center; width: 100%; height: 100%; position: absolute;
                    top: 0; left: 0; background: transparent; border: none;
                    color: var(--text-color);
                    font-family: 'Orbitron', 'Courier New', 'Consolas', 'Monaco', monospace;
                    box-sizing: border-box; outline: none; margin: 0;
                    overflow: hidden; display: flex; justify-content: center;
                    align-items: center; font-size: 50px;
                    animation: fancy-text-glow-pulse 10s infinite ease-in-out;
                    transition: color 0.5s ease-in-out;
                    font-variant-numeric: tabular-nums;
                    letter-spacing: 0;
                    white-space: nowrap;
                    font-weight: bold;
                }
                .fancy-timer-seg {
                    display: inline-block;
                    width: 2ch;
                    text-align: center;
                }
                .fancy-timer-seg-ms {
                    width: 3ch;
                }
                .fancy-timer-sep {
                    display: inline-block;
                    width: 0.5ch;
                    text-align: center;
                    opacity: 0.5;
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

        if (eventsBound) return;
        eventsBound = true;

        api.addEventListener("execution_start", () => GlobalTimer.start());
        api.addEventListener("executing", ({ detail }) => {
            if (detail === null) GlobalTimer.stop();
        });
        api.addEventListener("execution_error", () => GlobalTimer.stop());
        api.addEventListener("execution_interrupted", () => GlobalTimer.stop());
    }
};

app.registerExtension(FancyTimerNodeExtension);

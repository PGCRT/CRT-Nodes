import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// --- The Global Timer Manager ---
const GlobalTimer = {
    startTime: 0,
    intervalId: null,
    isRunning: false,
    activeNodes: new Set(),

    formatTime(ms) {
        if (ms < 0) ms = 0;
        const minutes = String(Math.floor(ms / 60000)).padStart(2, '0');
        const seconds = String(Math.floor((ms % 60000) / 1000)).padStart(2, '0');
        const milliseconds = String(ms % 1000).padStart(3, '0');
        return `${minutes}:${seconds}:${milliseconds}`;
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
            const timeString = this.formatTime(elapsed);
            this.activeNodes.forEach(node => {
                if (node.timerDisplay) node.timerDisplay.textContent = timeString;
            });
        }, 33);
    },

    stop() {
        if (!this.isRunning) return;
        this.isRunning = false;
        clearInterval(this.intervalId);
        
        const finalTime = Date.now() - this.startTime;
        const finalTimeString = this.formatTime(finalTime);
        
        this.activeNodes.forEach(node => {
            if (node.timerDisplay) {
                node.timerDisplay.textContent = finalTimeString;
                node.timerDisplay.style.color = 'var(--text-color, #7300ff)';
            }
            node.properties.elapsed_time_str = finalTimeString;
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
            
            nodeType.prototype.onNodeCreated = function () {
                this.bgcolor = "#000000";
                this.color = "#000000";
                this.title = "Execution Timer";
                this.properties = this.properties || {};
                this.size = [420, 130];

                const container = document.createElement("div");
                container.style.cssText = `width: 100%; height: 100%; position: relative; --text-color: #7300ff; --glow-color: #7300ff;`;

                this.timerDisplay = document.createElement("div");
                this.timerDisplay.className = "fancy-timer-display";
                this.timerDisplay.textContent = this.properties.elapsed_time_str || "00:00:000";
                
                container.appendChild(this.timerDisplay);
                this.addDOMWidget("fancyTimer", "Fancy Timer", container, { serialize: false });
                
                GlobalTimer.registerNode(this);
            };
            
            nodeType.prototype.onRemoved = function() { GlobalTimer.unregisterNode(this); };
            nodeType.prototype.onSerialize = function(o) { o.properties = this.properties; };
            nodeType.prototype.onConfigure = function(info) {
                this.properties = info.properties || {};
                if (this.timerDisplay) {
                    this.timerDisplay.textContent = this.properties.elapsed_time_str || "00:00:000";
                }
            };
        }
    },
    
    setup() {
        const style = document.createElement("style");
        style.innerText = `
            @keyframes fancy-text-glow-pulse {
                0%, 100% { text-shadow: 0 0 25px var(--glow-color); }
                50% { text-shadow: 0 0 35px var(--glow-color); }
            }
            .fancy-timer-display {
                text-align: center; width: 100%; height: 100%; position: absolute;
                top: 0; left: 0; background: transparent; border: none;
                color: var(--text-color);
                /* --- FONT CORRECTION: Restored your preferred font stack --- */
                font-family: 'Courier New', 'Consolas', 'Monaco', monospace;
                box-sizing: border-box; outline: none; margin: 0;
                overflow: hidden; display: flex; justify-content: center;
                align-items: center; font-size: 50px;
                animation: fancy-text-glow-pulse 10s infinite ease-in-out;
                transition: color 0.5s ease-in-out;
                font-variant-numeric: tabular-nums;
                letter-spacing: 0.1em;
                white-space: nowrap;
                font-weight: bold;
            }
        `;
        document.head.appendChild(style);

        // This line can optionally be removed if you don't want to load the Orbitron font at all,
        // but leaving it in doesn't hurt.
        if (!document.querySelector('link[href*="Orbitron"]')) {
            const fontLink = document.createElement("link");
            fontLink.href = "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap";
            fontLink.rel = "stylesheet";
            document.head.appendChild(fontLink);
        }
        
        // Your robust WebSocket event handling solution
        if (api.socket) {
            const originalOnMessage = api.socket.onmessage;
            api.socket.onmessage = function(event) {
                try {
                    const msg = JSON.parse(event.data);
                    
                    if (msg.type === "execution_start" || (msg.type === "executing" && msg.data?.node !== null)) {
                        GlobalTimer.start();
                    } else if (msg.type === "execution_cached" || msg.type === "execution_error" || (msg.type === "executing" && msg.data?.node === null)) {
                        GlobalTimer.stop();
                    }
                } catch (e) { /* Ignore non-JSON messages */ }
                
                if (originalOnMessage) {
                    originalOnMessage.call(this, event);
                }
            };
        } else {
            console.warn("FancyTimerNode: WebSocket not found on api object. Timer may not function.");
        }
    }
};

app.registerExtension(FancyTimerNodeExtension);
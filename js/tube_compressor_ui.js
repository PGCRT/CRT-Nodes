import { app } from "/scripts/app.js";

const CSS = `
@font-face {
    font-family: 'Orbitron';
    src: url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&display=swap') format('woff2'),
         local('Orbitron');
}

:root {
    --comp-primary: #ff8c00;
    --comp-primary-light: #ffaa40;
    --comp-secondary: #ffc973;
    --comp-bg: #000000;
    --comp-surface: #1a1a1a;
    --comp-dark: #111;
    --comp-text: #ffffff;
    --comp-text-dim: #888888;
    --comp-border: #333333;
    --comp-active: #00ff88;
    --comp-inactive: #ff4444;
}

/* Make all text non-selectable */
.compressor-container * {
    user-select: none !important;
    -webkit-user-select: none !important;
    -moz-user-select: none !important;
    -ms-user-select: none !important;
}

/* Main container - enhanced with modern styling */
.compressor-node-widget {
    position: relative !important;
    box-sizing: border-box !important;
    width: 100% !important;
    min-height: 420px !important;
    padding: 0px !important;
    margin: 0 !important;
    overflow: visible !important;
    display: block !important;
    visibility: visible !important;
    z-index: 1 !important;
    top: -17px !important;
}

.compressor-container {
    font-family: 'Orbitron', monospace;
    background: var(--comp-bg);
    border: 2px solid var(--comp-primary);
    border-radius: 20px;
    padding: 20px;
    width: 100%;
    max-width: 520px;
    box-sizing: border-box;
    min-height: 400px;
    box-shadow: 
        0 12px 40px rgba(255, 140, 0, 0.4),
        inset 0 2px 0 rgba(255, 255, 255, 0.1),
        0 0 60px rgba(255, 140, 0, 0.2);
    display: flex;
    flex-direction: column;
    gap: 16px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(10px);
}

.compressor-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(circle at 30% 20%, rgba(255, 140, 0, 0.1), transparent 50%),
                radial-gradient(circle at 70% 80%, rgba(255, 199, 115, 0.05), transparent 50%);
    pointer-events: none;
    z-index: 0;
}

.compressor-container > * {
    position: relative;
    z-index: 1;
}

/* Enhanced header */
.compressor-header {
    color: var(--comp-primary);
    font-size: 22px;
    font-weight: 700;
    text-align: center;
    margin: 0 0 16px 0;
    padding: 12px 0;
    text-shadow: 
        1px 1px 2px rgba(0, 0, 0, 0.8), 
        0 0 30px var(--comp-primary), 
        0 0 8px var(--comp-primary-light);
    background: linear-gradient(90deg, transparent, rgba(255, 140, 0, 0.15), transparent);
    border-radius: 12px;
    animation: compressorTitleGlow 4s ease-in-out infinite;
    position: relative;
}

.compressor-header::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    width: 60%;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--comp-primary), transparent);
    transform: translateX(-50%);
    border-radius: 1px;
}

@keyframes compressorTitleGlow {
    0%, 100% {
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8), 0 0 30px var(--comp-primary), 0 0 8px var(--comp-primary-light);
        transform: scale(1);
    }
    50% {
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8), 0 0 40px var(--comp-primary-light), 0 0 15px var(--comp-primary);
        transform: scale(1.02);
        color: var(--comp-primary-light);
    }
}

/* Status indicator */
.compressor-status-indicator {
    position: absolute;
    top: 16px;
    right: 16px;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--comp-active);
    box-shadow: 0 0 8px var(--comp-active);
    animation: pulseIndicator 2s ease-in-out infinite;
}

@keyframes pulseIndicator {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.2); }
}

/* Main layout improvements */
.compressor-main-layout {
    display: flex;
    flex-direction: column;
    gap: 20px;
    align-items: center;
}

/* Enhanced meter section */
.compressor-meter-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    width: 100%;
}

.compressor-gr-meter-bg {
    width: 160px;
    height: 160px;
    background: radial-gradient(circle, #333, #111);
    border-radius: 16px;
    border: 2px solid var(--comp-border);
    box-shadow: 
        inset 0 0 20px #000,
        0 4px 16px rgba(255, 140, 0, 0.25);
    padding: 15px;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.3s ease;
}

.compressor-gr-meter-bg:hover {
    border-color: var(--comp-primary);
    box-shadow: 
        inset 0 0 20px #000,
        0 6px 20px rgba(255, 140, 0, 0.35);
}

.compressor-gr-meter-canvas {
    width: 100%;
    height: 100%;
    background: transparent;
}

/* Enhanced preset select */
.preset-select {
    background: var(--comp-bg) !important;
    color: var(--comp-primary) !important;
    border: 2px solid var(--comp-border) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    width: 100% !important;
    margin-bottom: 12px !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}

.preset-select:hover {
    border-color: var(--comp-primary) !important;
    box-shadow: 0 0 8px rgba(255, 140, 0, 0.3) !important;
    background: rgba(255, 140, 0, 0.05) !important;
}

.preset-select:focus {
    border-color: var(--comp-primary) !important;
    box-shadow: 0 0 12px rgba(255, 140, 0, 0.5) !important;
    outline: none !important;
}

.preset-select option {
    background: var(--comp-bg) !important;
    color: var(--comp-text) !important;
}

/* Enhanced controls grid */
.compressor-controls-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 24px;
    width: 100%;
    padding: 16px;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 16px;
    border: 1px solid rgba(255, 140, 0, 0.2);
}

/* Enhanced knob groups */
.compressor-knob-group {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    padding: 12px;
    background: rgba(255, 255, 255, 0.02);
    border-radius: 12px;
    border: 1px solid rgba(255, 140, 0, 0.1);
    transition: all 0.3s ease;
}

.compressor-knob-group:hover {
    background: rgba(255, 140, 0, 0.05);
    border-color: rgba(255, 140, 0, 0.3);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(255, 140, 0, 0.2);
}

.compressor-knob-label {
    font-size: 12px;
    font-weight: 700;
    color: var(--comp-primary);
    text-shadow: 0 0 8px var(--comp-primary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Enhanced knobs */
.compressor-knob {
    width: 70px;
    height: 70px;
    position: relative;
    cursor: ns-resize;
    transition: all 0.3s ease;
}

.compressor-knob:hover {
    transform: scale(1.05);
}

.compressor-knob-bg {
    width: 100%;
    height: 100%;
    background: radial-gradient(circle at 30% 30%, #666, #222);
    border-radius: 50%;
    box-shadow: 
        0 6px 15px rgba(0, 0, 0, 0.6),
        inset 0 2px 4px #888,
        inset 0 -2px 4px #111;
    border: 1px solid #444;
}

.compressor-knob-indicator {
    width: 4px;
    height: 16px;
    background: linear-gradient(180deg, var(--comp-primary), var(--comp-primary-light));
    border-radius: 2px;
    position: absolute;
    top: 10px;
    left: calc(50% - 2px);
    transform-origin: 50% 25px;
    box-shadow: 
        0 0 8px var(--comp-primary),
        0 0 4px rgba(255, 140, 0, 0.5);
    transition: all 0.2s ease;
}

.compressor-knob:hover .compressor-knob-indicator {
    box-shadow: 
        0 0 12px var(--comp-primary),
        0 0 6px rgba(255, 140, 0, 0.8);
}

.compressor-knob-value {
    color: var(--comp-secondary);
    font-size: 11px;
    font-weight: 600;
    text-shadow: 0 0 6px var(--comp-secondary);
    background: rgba(255, 199, 115, 0.1);
    padding: 4px 8px;
    border-radius: 6px;
    border: 1px solid rgba(255, 199, 115, 0.2);
}

/* Enhanced footer */
.compressor-footer {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-around;
    align-items: center;
    padding: 16px;
    border-top: 1px solid rgba(255, 140, 0, 0.3);
    gap: 20px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 12px;
    margin-top: 8px;
}

/* Enhanced switch groups */
.compressor-switch-group {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 8px;
    background: rgba(255, 255, 255, 0.02);
    border-radius: 8px;
    border: 1px solid rgba(255, 140, 0, 0.1);
    transition: all 0.3s ease;
}

.compressor-switch-group:hover {
    background: rgba(255, 140, 0, 0.05);
    border-color: rgba(255, 140, 0, 0.3);
}

/* Enhanced toggle switches */
.compressor-toggle-switch {
    position: relative;
    display: inline-block;
    width: 50px;
    height: 24px;
}

.compressor-toggle-switch input {
    opacity: 0;
    width: 0;
    height: 0;
}

.switch-slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #333;
    border-radius: 24px;
    transition: .4s;
    border: 2px solid #111;
    box-shadow: 
        inset 0 2px 4px rgba(0, 0, 0, 0.6),
        0 2px 8px rgba(0, 0, 0, 0.3);
}

.switch-slider:before {
    position: absolute;
    content: "";
    height: 16px;
    width: 16px;
    left: 4px;
    bottom: 2px;
    background: linear-gradient(145deg, #ccc, #888);
    border-radius: 50%;
    transition: .4s;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
}

input:checked + .switch-slider {
    background-color: var(--comp-primary);
    box-shadow: 
        0 0 12px var(--comp-primary),
        inset 0 2px 4px rgba(0, 0, 0, 0.6);
}

input:checked + .switch-slider:before {
    background: linear-gradient(145deg, #fff, #ddd);
    transform: translateX(24px);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.6);
}

/* Enhanced soft clipper section */
.soft-clipper-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    background: rgba(0, 0, 0, 0.3);
    padding: 16px;
    border-radius: 12px;
    border: 1px solid rgba(255, 140, 0, 0.2);
    width: 100%;
    transition: all 0.3s ease;
}

.soft-clipper-section:hover {
    background: rgba(255, 140, 0, 0.05);
    border-color: rgba(255, 140, 0, 0.4);
}

.soft-clipper-header {
    font-size: 11px;
    color: var(--comp-primary);
    display: flex;
    gap: 8px;
    width: 100%;
    align-items: center;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.soft-clipper-header .clipper-label {
    flex-grow: 1;
    text-align: center;
    text-shadow: 0 0 6px var(--comp-primary);
}

.soft-clipper-controls {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
}

.clipper-knob {
    width: 55px !important;
    height: 55px !important;
}

.clipper-knob .compressor-knob-bg {
    background: radial-gradient(circle at 30% 30%, #555, #1a1a1a);
}

.clipper-knob .compressor-knob-indicator {
    height: 12px;
    transform-origin: 50% 19.5px;
    background: linear-gradient(180deg, #ff6b6b, #ff4444);
    box-shadow: 
        0 0 8px #ff6b6b,
        0 0 4px rgba(255, 107, 107, 0.5);
}

/* Enhanced makeup gain knob in footer */
.compressor-footer .compressor-knob-group {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 140, 0, 0.15);
    padding: 12px;
    border-radius: 8px;
}

.compressor-footer .compressor-knob {
    width: 50px;
    height: 50px;
}

.compressor-footer .compressor-knob-indicator {
    height: 10px;
    transform-origin: 50% 17px;
}

/* Responsive adjustments */
@media (max-width: 380px) {
    .compressor-controls-grid {
        grid-template-columns: 1fr;
        gap: 16px;
    }
    
    .compressor-footer {
        flex-direction: column;
        gap: 12px;
    }
}
`;

class CompressorUI {
    constructor(node) {
        this.node = node;
        this.knobs = {};
        this.container = null;
        this.isInitialized = false;
        
        // Set node appearance
        this.node.title = "";
        this.node.bgcolor = "transparent";
        this.node.color = "transparent";
        
        this.initializeUI();
    }

    initializeUI() {
        if (this.isInitialized) return;
        this.isInitialized = true;
        this.createUI();
        this.hideOriginalWidgets();
    }

    createUI() {
        // Inject enhanced styles
        if (!document.getElementById('compressor-ui-styles-enhanced')) {
            const style = document.createElement('style');
            style.id = 'compressor-ui-styles-enhanced';
            style.textContent = CSS;
            document.head.appendChild(style);
        }

        // Ensure Orbitron font is loaded
        if (!document.querySelector('link[href*="Orbitron"]')) {
            const fontLink = document.createElement("link");
            fontLink.href = "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;&display=swap";
            fontLink.rel = "stylesheet";
            document.head.appendChild(fontLink);
        }

        this.container = document.createElement("div");
        this.container.className = "compressor-container";
        
        // Enhanced header with status indicator
        const header = document.createElement("div");
        header.className = "compressor-header";
        header.textContent = "Tube Compressor";
        
        const statusIndicator = document.createElement("div");
        statusIndicator.className = "compressor-status-indicator";
        this.container.appendChild(statusIndicator);
        
        const mainLayout = document.createElement("div");
        mainLayout.className = "compressor-main-layout";
        
        const controlsGrid = this.createControlsGrid();
        const meterSection = this.createMeterSection();
        mainLayout.append(meterSection, controlsGrid);
        this.container.append(header, mainLayout, this.createFooter());
        
        // Wrap in enhanced widget container
        const wrapper = document.createElement("div");
        wrapper.className = "compressor-node-widget";
        wrapper.appendChild(this.container);
        
        this.node.addDOMWidget("compressor_ui", "div", wrapper, { serialize: false });
        
        console.log("[CompressorUI] Enhanced UI created successfully");
    }

    createMeterSection() {
        const section = document.createElement("div");
        section.className = "compressor-meter-section";
        
        // Enhanced presets with better organization
        const presets = [
            { name: "--- General Purpose ---" },
            { name: "Vocal Tamer", s: { threshold_db: -18, ratio: 3, attack_ms: 3, release_ms: 150, warmth: 0.2, mix_wet: 1, clipper_threshold: 1.0 } },
            { name: "Gentle Leveler", s: { threshold_db: -12, ratio: 1.8, attack_ms: 50, release_ms: 500, warmth: 0.1, mix_wet: 1, clipper_threshold: 1.0 } },
            { name: "Transparent Glue", s: { threshold_db: -15, ratio: 2.2, attack_ms: 25, release_ms: 200, warmth: 0.15, mix_wet: 1, clipper_threshold: 1.0 } },
            { name: "--- Drums & Percussion ---" },
            { name: "Drum Bus Glue", s: { threshold_db: -10, ratio: 2.5, attack_ms: 10, release_ms: 80, warmth: 0.25, mix_wet: 1, clipper_threshold: 1.0 } },
            { name: "Snare Punch", s: { threshold_db: -8, ratio: 4, attack_ms: 1, release_ms: 40, warmth: 0.3, mix_wet: 1, clipper_threshold: 0.95 } },
            { name: "Bass Smasher", s: { threshold_db: -25, ratio: 8, attack_ms: 20, release_ms: 60, warmth: 0.5, mix_wet: 1, clipper_threshold: 0.9 } },
            { name: "Kick Thump", s: { threshold_db: -12, ratio: 6, attack_ms: 5, release_ms: 100, warmth: 0.4, mix_wet: 1, clipper_threshold: 0.9 } },
            { name: "--- Limiting & Master ---" },
            { name: "Brickwall Limiter", s: { threshold_db: -2, ratio: 20, attack_ms: 0.1, release_ms: 50, warmth: 0, mix_wet: 1, clipper_threshold: 0.8 } },
            { name: "Mastering Glue", s: { threshold_db: -8, ratio: 1.5, attack_ms: 30, release_ms: 100, warmth: 0.05, mix_wet: 1, clipper_threshold: 1.0 } },
            { name: "--- Special Effects ---" },
            { name: "Parallel Warmth", s: { threshold_db: -25, ratio: 4, attack_ms: 5, release_ms: 100, warmth: 0.7, mix_wet: 0.5, clipper_threshold: 1.0 } },
            { name: "Pumping Effect", s: { threshold_db: -20, ratio: 10, attack_ms: 1, release_ms: 500, warmth: 0.3, mix_wet: 1, clipper_threshold: 0.85 } },
            { name: "Vintage Tube", s: { threshold_db: -16, ratio: 3.5, attack_ms: 15, release_ms: 250, warmth: 0.6, mix_wet: 1, clipper_threshold: 0.95 } }
        ];
        
        const select = document.createElement("select");
        select.className = "preset-select";
        select.innerHTML = `<option value="">-- Select Preset --</option>`;
        
        presets.forEach(p => {
            const option = document.createElement("option");
            option.value = p.name;
            option.textContent = p.name;
            if (!p.s) option.disabled = true;
            select.appendChild(option);
        });
        
        select.onchange = e => {
            const p = presets.find(pr => pr.name === e.target.value);
            if (!p || !p.s) return;
            
            for (const [key, value] of Object.entries(p.s)) {
                if (this.knobs[key]) {
                    this.knobs[key].widget.value = value;
                    this.knobs[key].updateVisuals(value);
                }
            }
            this.drawCompressionCurve();
            this.node.setDirtyCanvas(true, true);
            console.log(`[CompressorUI] Loaded preset: ${p.name}`);
        };
        
        // Enhanced meter background
        const meterBg = document.createElement("div");
        meterBg.className = "compressor-gr-meter-bg";
        
        this.grCanvas = document.createElement("canvas");
        this.grCanvas.className = "compressor-gr-meter-canvas";
        this.grCanvas.width = 130;
        this.grCanvas.height = 130;
        this.grCtx = this.grCanvas.getContext("2d");
        
        this.drawCompressionCurve();
        meterBg.appendChild(this.grCanvas);
        section.append(select, meterBg);
        
        return section;
    }

    createControlsGrid() {
        const grid = document.createElement("div");
        grid.className = "compressor-controls-grid";
        
        grid.append(
            this.createKnob("threshold_db", "Threshold", "dB"),
            this.createKnob("ratio", "Ratio", ":1"),
            this.createKnob("attack_ms", "Attack", "ms"),
            this.createKnob("release_ms", "Release", "ms"),
            this.createKnob("warmth", "Warmth", "%"),
            this.createKnob("mix_wet", "Mix", "%")
        );
        
        return grid;
    }
    
    createFooter() {
        const footer = document.createElement("div");
        footer.className = "compressor-footer";
        
        footer.append(
            this.createSwitch("match_input_peak", "Match Peak", true),
            this.createKnob("makeup_gain_db", "Makeup", "dB"),
            this.createSoftClipper()
        );
        
        return footer;
    }

    createSoftClipper() {
        const section = document.createElement("div");
        section.className = "soft-clipper-section";
        
        const header = document.createElement("div");
        header.className = "soft-clipper-header";
        
        const label = document.createElement("div");
        label.className = "clipper-label";
        label.textContent = "SOFT CLIPPER";
        
        header.append(this.createSwitch("soft_clipper_toggle", ""), label);
        
        const controls = document.createElement("div");
        controls.className = "soft-clipper-controls";
        
        const knob = this.createKnob("clipper_threshold", "Threshold", "%");
        knob.querySelector('.compressor-knob').classList.add('clipper-knob');
        controls.appendChild(knob);
        
        section.append(header, controls);
        return section;
    }

    createKnob(widgetName, label, unit = "") {
        const widget = this.node.widgets.find(w => w.name === widgetName);
        
        const group = document.createElement("div");
        group.className = "compressor-knob-group";
        
        const labelEl = document.createElement("div");
        labelEl.className = "compressor-knob-label";
        labelEl.textContent = label;
        
        const knob = document.createElement("div");
        knob.className = "compressor-knob";
        knob.innerHTML = `
            <div class="compressor-knob-bg"></div>
            <div class="compressor-knob-indicator"></div>
        `;
        
        const valueEl = document.createElement("div");
        valueEl.className = "compressor-knob-value";
        
        if (widget) {
            const indicator = knob.querySelector('.compressor-knob-indicator');
            
            const updateVisuals = (value) => {
                const { min = 0, max = 1 } = widget.options || {};
                const percent = (value - min) / (max - min);
                indicator.style.transform = `rotate(${-135 + (percent * 270)}deg)`;
                
                let displayVal = (unit === "%") ? value * 100 : value;
                const precision = (unit === "%" || unit === "dB") ? 1 : ((unit === ":1") ? 1 : 0);
                valueEl.textContent = `${displayVal.toFixed(precision)}${unit}`;
                
                if (['threshold_db', 'ratio'].includes(widgetName)) {
                    this.drawCompressionCurve();
                }
            };
            
            this.knobs[widgetName] = { widget, updateVisuals };
            updateVisuals(widget.value);
            
            let isDragging = false, lastY = 0;
            
            knob.onmousedown = e => {
                isDragging = true;
                lastY = e.clientY;
                document.body.style.cursor = 'ns-resize';
                e.preventDefault();
            };
            
            document.addEventListener("mousemove", e => {
                if (!isDragging) return;
                
                const { min = 0, max = 1 } = widget.options || {};
                let newValue = widget.value + ((lastY - e.clientY) * ((max - min) / 200));
                widget.value = Math.max(min, Math.min(max, newValue));
                
                if (widget.callback) widget.callback(widget.value);
                updateVisuals(widget.value);
                this.node.setDirtyCanvas(true, true);
                lastY = e.clientY;
            });
            
            document.addEventListener("mouseup", () => {
                isDragging = false;
                document.body.style.cursor = 'default';
            });
            
            knob.addEventListener('wheel', e => {
                e.preventDefault();
                const { min = 0, max = 1 } = widget.options || {};
                const sensitivity = (max - min) / 1500;
                let newValue = widget.value - (e.deltaY * sensitivity);
                widget.value = Math.max(min, Math.min(max, newValue));
                
                if (widget.callback) widget.callback(widget.value);
                updateVisuals(widget.value);
                this.node.setDirtyCanvas(true, true);
            });
        }
        
        group.append(labelEl, knob, valueEl);
        return group;
    }

    createSwitch(widgetName, label, isStandalone = false) {
        const widget = this.node.widgets.find(w => w.name === widgetName);
        
        const group = document.createElement("div");
        group.className = "compressor-switch-group";
        
        if (label) {
            const labelEl = document.createElement("div");
            labelEl.className = "compressor-knob-label";
            labelEl.textContent = label;
            group.appendChild(labelEl);
        }
        
        const switchLabel = document.createElement("label");
        switchLabel.className = "compressor-toggle-switch";
        
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        
        const slider = document.createElement("span");
        slider.className = "switch-slider";
        
        switchLabel.append(checkbox, slider);
        
        if (widget) {
            checkbox.checked = widget.value === "ON";
            checkbox.onchange = () => {
                widget.value = checkbox.checked ? "ON" : "OFF";
                if (widget.callback) widget.callback(widget.value);
                this.node.setDirtyCanvas(true, true);
                console.log(`[CompressorUI] ${widgetName} set to: ${widget.value}`);
            };
        }
        
        group.appendChild(switchLabel);
        return group;
    }

    hideOriginalWidgets() {
        this.node.widgets?.forEach(w => {
            if (!['audio', 'gr_meter_data', 'compressor_ui'].includes(w.name)) {
                w.computeSize = () => [0, -4];
                if (w.element) w.element.style.display = 'none';
            }
        });
        this.node.setDirtyCanvas(true, true);
    }

    drawCompressionCurve() {
        if (!this.grCtx) return;
        
        const { width: w, height: h } = this.grCanvas;
        const ctx = this.grCtx;
        
        // Clear canvas
        ctx.clearRect(0, 0, w, h);
        
        // Get current values
        const threshold = this.node.widgets.find(w => w.name === 'threshold_db')?.value ?? -20;
        const ratio = this.node.widgets.find(w => w.name === 'ratio')?.value ?? 4;
        
        const minDb = -30.0;
        const maxDb = 30.0;
        const range = maxDb - minDb;
        
        const dbToPixel = (db, dim, invert = false) => {
            const n = (db - minDb) / range;
            return invert ? dim - (n * dim) : n * dim;
        };
        
        // Enhanced grid
        ctx.strokeStyle = 'rgba(255, 140, 0, 0.3)';
        ctx.lineWidth = 1;
        
        // Center lines
        const center = dbToPixel(0, w);
        ctx.beginPath();
        ctx.moveTo(center, 0);
        ctx.lineTo(center, h);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(0, center);
        ctx.lineTo(w, center);
        ctx.stroke();
        
        // Grid lines
        ctx.strokeStyle = 'rgba(255, 140, 0, 0.15)';
        ctx.lineWidth = 0.5;
        
        for (let db = -24; db <= 24; db += 6) {
            if (db !== 0) {
                const y = dbToPixel(db, h, true);
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();
            }
        }
        
        // Unity gain reference line
        ctx.strokeStyle = 'rgba(100, 100, 100, 0.4)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(dbToPixel(minDb, w), dbToPixel(minDb, h, true));
        ctx.lineTo(dbToPixel(maxDb, w), dbToPixel(maxDb, h, true));
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Enhanced compression curve
        ctx.strokeStyle = '#ff8c00';
        ctx.lineWidth = 3;
        ctx.shadowBlur = 8;
        ctx.shadowColor = 'rgba(255, 140, 0, 0.8)';
        
        ctx.beginPath();
        
        for (let i = 0; i <= 200; i++) {
            const inputDb = minDb + (i / 200) * range;
            const outputDb = (inputDb <= threshold) 
                ? inputDb 
                : threshold + (inputDb - threshold) / ratio;
            
            const x = dbToPixel(inputDb, w);
            const y = dbToPixel(outputDb, h, true);
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
        ctx.shadowBlur = 0;
        
        // Threshold indicator
        const thresholdX = dbToPixel(threshold, w);
        const thresholdY = dbToPixel(threshold, h, true);
        
        ctx.beginPath();
        ctx.arc(thresholdX, thresholdY, 4, 0, 2 * Math.PI);
        ctx.fillStyle = '#ffaa40';
        ctx.fill();
        
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Labels
        ctx.font = 'bold 9px Orbitron, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillStyle = '#ffaa40';
        ctx.shadowBlur = 2;
        ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
        
        // Input/Output labels
        ctx.fillText('INPUT', w / 2, h - 4);
        
        ctx.save();
        ctx.translate(8, h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('OUTPUT', 0, 0);
        ctx.restore();
        
        // Ratio and threshold info
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        ctx.font = 'bold 8px Orbitron, monospace';
        ctx.fillStyle = '#ffc973';
        ctx.fillText(`${ratio.toFixed(1)}:1`, w - 4, 4);
        ctx.fillText(`${threshold.toFixed(1)}dB`, w - 4, 16);
        
        ctx.shadowBlur = 0;
    }

    destroy() {
        if (this.container) {
            this.container.remove();
        }
        console.log(`[CompressorUI] UI destroyed for node ${this.node.id}`);
    }
}

// Register the extension
app.registerExtension({
    name: "Comfy.TubeCompressor.UI.Enhanced",
    nodeCreated(node) {
        if (node.comfyClass === "AudioCompressor") {
            // Clean up any existing instance
            if (node.compressorUIInstance) {
                node.compressorUIInstance.destroy();
            }
            
            // Create new enhanced instance
            setTimeout(() => {
                node.compressorUIInstance = new CompressorUI(node);
				node.setSize([520, 10]);
                console.log(`[CompressorUI] Enhanced compressor UI created for node ${node.id}`);
            }, 100);
            
            // Handle node removal
            const originalOnRemove = node.onRemove;
            node.onRemove = function() {
                if (this.compressorUIInstance) {
                    this.compressorUIInstance.destroy();
                }
                if (originalOnRemove) {
                    originalOnRemove.call(this);
                }
            };
        }
    }
});
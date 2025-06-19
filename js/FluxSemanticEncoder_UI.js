import { app } from "../../scripts/app.js";

const MAX_EQ_BANDS = 4;

console.log("FluxSemanticEncoder_UI.js (Enhanced V7): Script loading...");

const CSS_STYLES_SEMANTIC_EQ = `
@font-face {
    font-family: 'Orbitron';
    src: url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&display=swap') format('woff2'),
         local('Orbitron');
}

:root {
    --primary-accent: #7700ff;
    --primary-accent-light: #9a70ff;
    --active-green: #2ecc71;
    --inactive-red: #d11d0a;
    --bright-orange: #ff8c00;
    --text-white: #ffffff;
    --text-gray: #555555;
    --background-black: #000000;
    --translucent-light: rgba(255, 255, 255, 0.1);
    --translucent-lighter: rgba(255, 255, 255, 0.2);
    --shift-blue: #3498db;
    --inject-purple: #9b59b6;
}

@layer properties {
  @property --knob-value-percent {
    syntax: "<number>";
    inherits: true;
    initial-value: 0;
  }
}

/* Make all text non-selectable */
.semantic-eq-container * {
    user-select: none !important;
    -webkit-user-select: none !important;
    -moz-user-select: none !important;
    -ms-user-select: none !important;
}

/* Main container with fixed positioning */
.semantic-eq-node-custom-widget {
    position: relative !important;
    box-sizing: border-box !important;
    width: 100% !important;
    min-height: 200px !important;
    padding: 0px !important;
    margin: 0 !important;
    overflow: visible !important;
    display: block !important;
    visibility: visible !important;
    z-index: 1 !important;
    top: -17px !important; /* Push the entire widget up by 20px */
}

.semantic-eq-container {
    background: #000000;
    border: 2px solid var(--primary-accent);
    border-radius: 16px;
    padding: 16px;
    margin: 0;
    width: 100%;
    max-width: 500px;
    box-sizing: border-box;
    min-height: 220px;
    box-shadow: 
        0 8px 32px rgba(119, 0, 255, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    display: flex;
    flex-direction: column;
    align-items: stretch;
    position: relative;
    overflow: hidden;
}

/* Enhanced title with better positioning */
.semantic-eq-title {
    position: sticky;
    top: 0;
    z-index: 10;
    color: var(--primary-accent);
    user-select: none;
    font-family: 'Orbitron', monospace;
    font-size: 16px;
    font-weight: 700;
    text-align: center;
    margin: 0 0 16px 0;
    padding: 8px 0;
    text-shadow: 
        1px 1px 2px rgba(0, 0, 0, 0.8), 
        0 0 25px var(--primary-accent), 
        0 0 5px var(--primary-accent-light);
    background: linear-gradient(90deg, transparent, rgba(119, 0, 255, 0.1), transparent);
    border-radius: 8px;
    animation: breathePurpleTitle 3s ease-in-out infinite;
}

@keyframes breathePurpleTitle {
    0%, 100% {
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8), 0 0 25px var(--primary-accent), 0 0 5px var(--primary-accent-light);
        transform: scale(1);
    }
    50% {
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8), 0 0 35px var(--primary-accent-light), 0 0 10px var(--primary-accent);
        transform: scale(1.02);
        color: var(--primary-accent-light);
    }
}

/* Fixed controls section */
.semantic-eq-controls-section {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 16px;
    align-items: center;
    position: relative;
    z-index: 5;
}

.controls-row {
    display: flex;
    justify-content: center;
    align-items: flex-start;
    gap: 16px;
    flex-wrap: wrap;
    width: 100%;
}

.control-group {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 120px;
    position: relative;
}

/* --- KNOB STYLES --- */
.flux-knob-label {
  color: var(--primary-accent);
  font-family: 'Orbitron', monospace;
  font-weight: 70;
  text-align: center;
  margin-bottom: 20px;
  font-size: 14px;
  text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8), 
               0 0 15px var(--primary-accent), 
               0 0 3px var(--primary-accent-light);
}

.control-group-flux_guidance .flux-knob-label {
  color: var(--bright-orange);
  text-shadow: 1px 1px 2px rgba(0,0,0,0.8), 
               0 0 20px var(--bright-orange), 
               0 0 5px var(--bright-orange);
}

.flux-knob {
  --knob-size: 110px;
  --knob-glow-color: var(--primary-accent);
  --knob-value-percent: 0;
  width: var(--knob-size);
  height: var(--knob-size);
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
}

.flux-knob-base {
  width: 100%;
  height: 100%;
  border-radius: 50%;
  background: #000000;
  position: absolute;
}

.flux-knob-dial {
  width: 100%;
  height: 100%;
  background: #000000;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  box-shadow: inset 0 0 10px rgb(0 0 0 / 80%), 
              inset 0 -2px 3px -1px rgba(255 255 255 / 10%);
  position: relative;
  z-index: 3;
}

/* Enhanced glow ring with better outward glow effect */
.flux-knob-glow-ring {
  width: 100%;
  height: 100%;
  position: absolute;
  border-radius: 50%;
  z-index: 1;
  
  /* Create the base glow effect */
  background: conic-gradient(
    from 180deg,
    transparent 0deg,
    var(--knob-glow-color) calc(var(--knob-value-percent) * 3.6deg),
    transparent calc(var(--knob-value-percent) * 3.6deg + 2deg)
  );
  
  /* Enhanced outward glow with multiple layers */
  filter: blur(2px) brightness(1.3);
  
  /* Add outer glow rings */
  box-shadow: 
    0 0 6px var(--knob-glow-color),
    0 0 12px var(--knob-glow-color),
    0 0 20px rgba(var(--knob-glow-color-rgb, 64, 224, 255), 0.4);
}

/* Additional glow layer for stronger outward effect */
.flux-knob-glow-ring::before {
  content: '';
  position: absolute;
  top: -8px;
  left: -8px;
  right: -8px;
  bottom: -8px;
  border-radius: 50%;
  z-index: -1;
  
  background: conic-gradient(
    from 180deg,
    transparent 0deg,
    rgba(var(--knob-glow-color-rgb, 64, 224, 255), 0.6) calc(var(--knob-value-percent) * 3.6deg),
    transparent calc(var(--knob-value-percent) * 3.6deg + 3deg)
  );
  
  filter: blur(5px);
  opacity: 0.8;
}

/* Subtle ambient glow layer */
.flux-knob-glow-ring::after {
  content: '';
  position: absolute;
  top: -12px;
  left: -12px;
  right: -12px;
  bottom: -12px;
  border-radius: 50%;
  z-index: -2;
  
  background: conic-gradient(
    from 180deg,
    transparent 0deg,
    rgba(var(--knob-glow-color-rgb, 64, 224, 255), 0.3) calc(var(--knob-value-percent) * 3.6deg),
    transparent calc(var(--knob-value-percent) * 3.6deg + 5deg)
  );
  
  filter: blur(8px);
  opacity: 0.6;
}

.flux-knob output {
  color: var(--text-white);
  font-variant-numeric: tabular-nums;
  font-family: 'Orbitron', monospace;
  font-size: 1.8em;
  font-weight: 500;
  user-select: none;
  pointer-events: none;
  z-index: 4;
  text-shadow: 0 0 8px var(--knob-glow-color), 
               0 0 3px black;
}

.flux-knob:active, 
.flux-knob:focus-within {
  cursor: ns-resize;
}

/* Optional: Enhanced glow for specific control groups */
.control-group-flux_guidance .flux-knob {
  --knob-glow-color: var(--bright-orange);
  --knob-glow-color-rgb: 255, 165, 0; /* Orange RGB values */
}

/* Animation for smooth glow transitions */
.flux-knob-glow-ring,
.flux-knob-glow-ring::before,
.flux-knob-glow-ring::after {
  transition: all 0.2s ease-out;
}

/* Hover effect for enhanced interactivity */
.flux-knob:hover .flux-knob-glow-ring {
  filter: blur(2px) brightness(1.5);
}

.flux-knob:hover .flux-knob-glow-ring::before {
  opacity: 1;
  filter: blur(5px) brightness(1.1);
}

.flux-knob:hover .flux-knob-glow-ring::after {
  opacity: 0.8;
}

/* Optional: Enhanced glow for specific control groups */
.control-group-flux_guidance .flux-knob {
  --knob-glow-color: var(--bright-orange);
  --knob-glow-color-rgb: 255, 165, 0; /* Orange RGB values */
}

/* Animation for smooth glow transitions */
.flux-knob-glow-ring,
.flux-knob-glow-ring::before,
.flux-knob-glow-ring::after {
  transition: all 0.2s ease-out;
}

/* Hover effect for enhanced interactivity */
.flux-knob:hover .flux-knob-glow-ring {
  filter: blur(3px) brightness(1.8);
}

.flux-knob:hover .flux-knob-glow-ring::before {
  opacity: 1;
  filter: blur(8px) brightness(1.2);
}

.flux-knob:hover .flux-knob-glow-ring::after {
  opacity: 0.8;
}

/* Enhanced tabs with better spacing and reduced size */
.semantic-eq-tabs {
    display: flex;
    flex-wrap: nowrap;
    gap: 4px;
    margin-bottom: 0; /* Adjusted for collapse bar */
    padding: 6px;
    justify-content: center;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    border: 1px solid rgba(119, 0, 255, 0.2);
    position: relative;
    z-index: 5;
}

.semantic-eq-tab {
    background: #000000;
    color: #cccccc;
    border: 1px solid transparent;
    padding: 6px 10px;
    border-radius: 16px;
    cursor: pointer;
    font-size: 9px;
    font-weight: 600;
    font-family: 'Orbitron', monospace;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    white-space: nowrap;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    position: relative;
    min-width: 50px;
    text-align: center;
    flex: 1;
    max-width: 80px;
}

.semantic-eq-tab:hover {
    background: linear-gradient(145deg, var(--primary-accent), var(--primary-accent-light));
    color: var(--text-white);
    border-color: var(--primary-accent-light);
    transform: translateY(-1px) scale(1.02);
    box-shadow: 0 4px 12px rgba(119, 0, 255, 0.3);
}

.semantic-eq-tab.active {
    background: linear-gradient(145deg, var(--primary-accent), var(--primary-accent-light));
    color: var(--text-white);
    border-color: var(--active-green);
    transform: translateY(-1px) scale(1.05);
    box-shadow: 
        0 4px 12px rgba(119, 0, 255, 0.4),
        0 0 15px rgba(46, 204, 113, 0.3);
    animation: tab-glow 2s infinite;
}

/* Tab-specific colors */
.semantic-eq-tab[data-type="shift"]:hover,
.semantic-eq-tab[data-type="shift"].active {
    background: linear-gradient(145deg, var(--shift-blue), #5dade2);
    border-color: var(--shift-blue);
}

.semantic-eq-tab[data-type="inject"]:hover,
.semantic-eq-tab[data-type="inject"].active {
    background: linear-gradient(145deg, var(--inject-purple), #bb7cbf);
    border-color: var(--inject-purple);
}

/* Collapse Bar */
.semantic-eq-collapse-bar {
    width: 100%;
    height: 10px;
    background-color: rgba(119, 0, 255, 0.2);
    border-radius: 5px;
    margin-top: 4px;
    margin-bottom: 8px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background-color 0.3s ease;
	box-shadow: 0 6px 20px rgba(119, 0, 255, 0.3);
}

.semantic-eq-collapse-bar:hover {
    background-color: rgba(119, 0, 255, 0.8);
}

.semantic-eq-collapse-handle {
    width: 30px;
    height: 4px;
    background-color: var(--primary-accent-light);
    border-radius: 2px;
}

/* Content Container */
.semantic-eq-content-container {
    overflow: hidden;
    transition: max-height 0.4s ease-in-out, opacity 0.4s ease-in-out;
    max-height: 800px; /* Large enough to not clip content */
    opacity: 1;
}

.semantic-eq-content-container.collapsed {
    max-height: 0;
    opacity: 0;
}

/* Enhanced canvas with better positioning */
.semantic-eq-eq-canvas {
    width: 100%;
    max-width: 460px;
    height: 180px;
    margin: 0 auto;
    background: #000000;
    border-radius: 12px;
    border: 2px solid rgba(119, 0, 255, 0.3);
    box-shadow: 
        0 4px 16px rgba(119, 0, 255, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    cursor: crosshair;
    transition: all 0.3s ease;
    position: relative;
    z-index: 1;
}

.semantic-eq-eq-canvas:hover {
    border-color: var(--active-green);
    box-shadow: 
        0 6px 20px rgba(119, 0, 255, 0.3),
        0 0 15px rgba(46, 204, 113, 0.2);
    transform: translateY(-1px);
}

/* Section containers with proper spacing */
.eq-section,
.magic-section,
.manual-section,
.shift-section,
.inject-section {
    margin: 0;
    padding: 16px 0;
    text-align: center;
    position: relative;
    z-index: 1;
    min-height: 40px;
    transition: all 0.3s ease;
}

.eq-section.hidden,
.magic-section.hidden,
.manual-section.hidden,
.shift-section.hidden,
.inject-section.hidden {
    display: none !important;
    opacity: 0;
    transform: translateY(-10px);
}

.eq-section.visible,
.magic-section.visible,
.manual-section.visible,
.shift-section.visible,
.inject-section.visible {
    display: block !important;
    opacity: 1;
    transform: translateY(0);
}

/* Enhanced form controls for shift and inject sections */
.param-container {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 12px;
    margin: 8px 0;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    border: 1px solid rgba(119, 0, 255, 0.2);
    transition: all 0.3s ease;
}

.param-toggle-row {
    cursor: pointer;
}

.param-row {
    display: flex;
    align-items: center;
    gap: 12px;
    width: 100%;
    justify-content: space-between;
}

.param-label {
    color: var(--primary-accent-light);
    font-size: 11px;
    font-weight: 600;
    font-family: 'Orbitron', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    text-shadow: 0 0 6px var(--primary-accent-light);
    min-width: 120px;
    text-align: left;
    transition: all 0.3s ease;
}

.param-select {
    background: #000000;
    border: 1px solid var(--active-green);
    border-radius: 6px;
    color: var(--active-green);
    padding: 4px 8px;
    font-size: 10px;
    width: 100px;
    font-family: 'Orbitron', monospace;
    font-weight: 500;
    transition: all 0.3s ease;
    cursor: pointer;
}

.param-led {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background-color: var(--inactive-red);
    box-shadow: 0 0 8px rgba(209, 29, 10, 0.5);
    transition: all 0.3s ease;
    border: 1px solid #ff4444;
}

.param-led.active {
    background-color: var(--active-green);
    box-shadow: 0 0 12px 2px rgba(46, 204, 113, 0.7);
    border: 1px solid #9affc8;
}

/* Magic slider enhancements */
.magic-slider-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 12px;
    margin: 8px 0;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    border: 1px solid rgba(119, 0, 255, 0.2);
    transition: all 0.3s ease;
}

.magic-slider-row {
    display: flex;
    align-items: center;
    gap: 12px;
    width: 100%;
    max-width: 400px;
}

.magic-slider {
    -webkit-appearance: none;
    appearance: none;
    flex: 1;
    height: 6px;
    background: linear-gradient(90deg, 
        rgba(119, 0, 255, 0.3), 
        rgba(119, 0, 255, 0.1));
    border-radius: 3px;
    outline: none;
    transition: all 0.3s ease;
    box-shadow: 
        0 2px 8px rgba(119, 0, 255, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.magic-slider:hover {
    background: linear-gradient(90deg, 
        rgba(119, 0, 255, 0.5), 
        rgba(119, 0, 255, 0.2));
}

.magic-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 16px;
    height: 16px;
    background: radial-gradient(circle, var(--active-green), #1e8449);
    border-radius: 50%;
    border: 2px solid #ffffff;
    box-shadow: 
        0 2px 8px rgba(46, 204, 113, 0.4),
        0 0 10px rgba(46, 204, 113, 0.2);
    cursor: grab;
    transition: all 0.2s ease;
}

.magic-value {
    color: var(--active-green);
    font-size: 12px;
    font-family: 'Orbitron', monospace;
    font-weight: 600;
    min-width: 50px;
    text-align: center;
    text-shadow: 0 0 6px var(--active-green);
    background: rgba(46, 204, 113, 0.1);
    padding: 4px 8px;
    border-radius: 6px;
    border: 1px solid rgba(46, 204, 113, 0.2);
    transition: all 0.3s ease;
}

/* Disabled state for sections */
.controls-disabled {
    opacity: 0.5;
    pointer-events: none;
}

.controls-disabled .param-label,
.controls-disabled .magic-value,
.controls-disabled .param-select {
    color: var(--text-gray);
    text-shadow: none;
}

.controls-disabled .magic-slider::-webkit-slider-thumb {
    background: var(--text-gray);
    box-shadow: none;
}

.controls-disabled .magic-slider {
    background: #222;
}

.controls-disabled .param-toggle-row {
    opacity: 1; /* Keep the toggle itself fully visible and interactive */
    pointer-events: auto;
}
`;

class FluxSemanticEncoderUI {
    constructor(node) {
        this.node = node;
        this.container = null;
        this.activeTab = "shift"; // Start with shift tab
        this.numberInputs = {};
        this.eqPoints = [];
        this.selectedPoint = null;
        this.hoveredPoint = null;
        this.canvas = null;
        this.ctx = null;
        this.isDragging = false;
        this.retryCount = 0;
        this.maxRetries = 5;
        this.isInitialized = false;
        this.resizeObserver = null;
        this.isCollapsed = true; // Set default state to collapsed
        this.collapseBar = null;
        this.contentContainer = null;
        this.animationFrameId = null;
        
        // Bind methods to preserve context
        this.handleMouseDown = this.handleMouseDown.bind(this);
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.handleMouseUp = this.handleMouseUp.bind(this);
        this.handleMouseLeave = this.handleMouseLeave.bind(this);
        this.handleWheel = this.handleWheel.bind(this);
        this.handleResize = this.handleResize.bind(this);
        
        // Initialize with proper timing
        this.initializeWithRetry();
    }

    initializeWithRetry() {
        if (this.isInitialized) return;
        
        console.log(`[FluxSemanticEncoderUI] Attempting to initialize UI for node ${this.node.id}, retry ${this.retryCount}`);
        
        // Check if DOM is ready and node widgets are available
        if (document.readyState === 'loading' || !this.node.widgets) {
            if (this.retryCount < this.maxRetries) {
                this.retryCount++;
                setTimeout(() => this.initializeWithRetry(), Math.min(500 * this.retryCount, 2000));
                return;
            } else {
                console.error(`[FluxSemanticEncoderUI] Failed to initialize after ${this.maxRetries} retries`);
                return;
            }
        }
        
        try {
            // IMPORTANT: Set default values FIRST before any UI creation
            this.setWidgetDefaults();
            this.initializeUI();
            this.isInitialized = true;
            console.log(`[FluxSemanticEncoderUI] UI initialized successfully for node ${this.node.id}`);
        } catch (err) {
            console.error(`[FluxSemanticEncoderUI] Error initializing for node ${this.node.id}:`, err);
            if (this.retryCount < this.maxRetries) {
                this.retryCount++;
                setTimeout(() => this.initializeWithRetry(), 1000);
            }
        }
    }

    setWidgetDefaults() {
        if (!this.node.widgets) return;
        
        console.log(`[FluxSemanticEncoderUI] Setting widget defaults for node ${this.node.id}`);
        
        // Set default values for main controls
        const controlDefaults = {
            'flux_guidance': 2.7,
            'multiplier': 1.0,
            'dry_wet_mix': 1.0,
            'magic_scale_min': 0.8,
            'magic_scale_max': 1.2,
            'enable_semantic_shift': false,
            'semantic_shift_strength': 0.5,
            'semantic_shift_distance': 0.2,
            'semantic_shift_randomness': 0.2,
            'semantic_shift_max_neighbors': 50,
            'enable_token_injection': false,
            'injection_mode': 'ADD',
            'num_tokens_to_inject': 5,
            'target_prompt': 'Both'
        };
        
        Object.entries(controlDefaults).forEach(([param, defaultValue]) => {
            const widget = this.node.widgets.find(w => w.name === param);
            if (widget) {
                // Always set the default value on initialization
                widget.value = defaultValue;
                console.log(`[FluxSemanticEncoderUI] Set ${param} = ${defaultValue}`);
            }
        });
        
        // Set default values for EQ bands
        const defaultCenters = [0.25, 0.5, 0.75, 1.0];
        for (let i = 1; i <= MAX_EQ_BANDS; i++) {
            const centerWidget = this.node.widgets.find(w => w.name === `band_${i}_center`);
            const gainWidget = this.node.widgets.find(w => w.name === `band_${i}_gain`);
            const qWidget = this.node.widgets.find(w => w.name === `band_${i}_q_factor`);
            
            if (centerWidget) {
                centerWidget.value = defaultCenters[i - 1];
            }
            if (gainWidget) {
                gainWidget.value = 1.0;
            }
            if (qWidget) {
                qWidget.value = 1.0;
            }
        }
        
        // Force canvas update
        this.node.setDirtyCanvas(true, true);
    }

    initializeUI() {
        this.injectStyles();
        this.hideOriginalWidgets();
        this.createCustomDOM();
        this.initializeEQPoints();
        this.syncInternalStateFromWidgets();
        this.setupResizeObserver();
        this.setNodeSizeInstantly();
        this.node.setDirtyCanvas(true, true);
    }

    injectStyles() {
        if (!document.getElementById('semantic-eq-styles-enhanced')) {
            const styleSheet = document.createElement('style');
            styleSheet.id = 'semantic-eq-styles-enhanced';
            styleSheet.textContent = CSS_STYLES_SEMANTIC_EQ;
            document.head.appendChild(styleSheet);
        }

        // Ensure Orbitron font is loaded
        if (!document.querySelector('link[href*="Orbitron"]')) {
            const fontLink = document.createElement("link");
            fontLink.href = "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;&display=swap";
            fontLink.rel = "stylesheet";
            document.head.appendChild(fontLink);
        }
    }

    hideOriginalWidgets() {
        if (!this.node.widgets) return;
        
        this.node.widgets.forEach(widget => {
            if (widget.name) {
                const keepVisible = ['clip', 'text', 'magic_seed'];
                if (!keepVisible.includes(widget.name)) {
                    widget.computeSize = () => [0, -4];
                    widget.type = "hidden_via_js";
                    if (widget.element) {
                        widget.element.style.display = 'none';
                    }
                }
            }
        });
    }

    createCustomDOM() {
        console.log(`[FluxSemanticEncoderUI] Creating custom DOM for node ${this.node.id}`);
        
        // Clean up any existing container
        if (this.container) {
            this.container.remove();
        }

        this.container = document.createElement('div');
        this.container.className = 'semantic-eq-container';

        // Create title
        const title = document.createElement('div');
        title.className = 'semantic-eq-title';
        title.textContent = 'Flux Semantic Encoder';
        this.container.appendChild(title);

        // Create sections
        this.createControlsSection();
        this.createTabs();

        // Create and add the collapse bar
        const collapseBar = document.createElement('div');
        collapseBar.className = 'semantic-eq-collapse-bar';
        collapseBar.title = 'Collapse/Expand';
        const collapseHandle = document.createElement('div');
        collapseHandle.className = 'semantic-eq-collapse-handle';
        collapseBar.appendChild(collapseHandle);
        collapseBar.addEventListener('click', () => this.toggleCollapse());
        this.container.appendChild(collapseBar);
        this.collapseBar = collapseBar;

        // Create a container for all tab content
        const contentContainer = document.createElement('div');
        contentContainer.className = 'semantic-eq-content-container';
        this.contentContainer = contentContainer;

        // Create sections and append to the content container
        this.createShiftSection(contentContainer);
        this.createInjectSection(contentContainer);
        this.createEQBandsSection(contentContainer);
        this.createMagicSection(contentContainer);
        this.createManualSection(contentContainer);
        
        this.container.appendChild(contentContainer);

        // Update visibility based on active tab
        this.updateTabVisibility();
        this.applyCollapseState(true); // Apply instantly on creation

        // Wrap in widget container
        const widgetWrapper = document.createElement('div');
        widgetWrapper.className = 'semantic-eq-node-custom-widget';
        widgetWrapper.appendChild(this.container);
        
        // Add DOM widget to node
        const domWidget = this.node.addDOMWidget('semantic_eq_ui', 'div', widgetWrapper, {
            serialize: false,
            draw: (ctx, node, widget_width, y, widget_height) => {
                // Ensure proper positioning
                if (widgetWrapper.style.position !== 'relative') {
                    widgetWrapper.style.position = 'relative';
                    widgetWrapper.style.width = '100%';
                }
            }
        });
        
        if (!domWidget) {
            console.error(`[FluxSemanticEncoderUI] addDOMWidget FAILED for node ${this.node.id}`);
        }
    }

    createKnob(param) {
        const controlGroup = document.createElement('div');
        controlGroup.className = 'control-group control-group-' + param.name;
    
        const label = document.createElement('div');
        label.className = 'flux-knob-label';
        label.textContent = param.label;
    
        const knob = document.createElement('div');
        knob.className = 'flux-knob';
        knob.tabIndex = 0; 
    
        if (param.color) {
            knob.style.setProperty('--knob-glow-color', `var(--${param.color})`);
        }
    
        const knobBase = document.createElement('div');
        knobBase.className = 'flux-knob-base';
        knob.appendChild(knobBase);
        
        const glowRing = document.createElement('div');
        glowRing.className = 'flux-knob-glow-ring';
        knob.appendChild(glowRing);
    
        const knobDial = document.createElement('div');
        knobDial.className = 'flux-knob-dial';
    
        const output = document.createElement('output');
        knobDial.appendChild(output);
        knob.appendChild(knobDial);
    
        const widget = this.node.widgets.find(w => w.name === param.name);
        let currentValue = widget ? widget.value : param.default;
    
        const updateKnobVisuals = (value) => {
            const percentage = ((value - param.min) / (param.max - param.min)) * 100;
            knob.style.setProperty('--knob-value-percent', Math.max(0, Math.min(100, percentage)));
            output.textContent = Number(value).toFixed(param.precision);
        };
    
        const updateWidgetValue = (newValue) => {
            newValue = Math.max(param.min, Math.min(param.max, newValue));
            newValue = Math.round(newValue / param.step) * param.step;
    
            if (currentValue.toFixed(param.precision) !== newValue.toFixed(param.precision)) {
                currentValue = newValue;
                if (widget) {
                    widget.value = newValue;
                }
                updateKnobVisuals(newValue);
                this.node.setDirtyCanvas(true, true);
            }
        };
        
        knob.addEventListener('mousedown', (e) => {
            e.preventDefault();
            const startY = e.clientY;
            let startValue = currentValue;
            let finalValue = currentValue;
            
            const onMouseMove = (moveEvent) => {
                const deltaY = startY - moveEvent.clientY;
                const sensitivity = (param.max - param.min) / 200; // Adjust sensitivity
                let newValue = startValue + deltaY * sensitivity;

                newValue = Math.max(param.min, Math.min(param.max, newValue));
                finalValue = newValue;
                updateKnobVisuals(finalValue);
            };
            
            const onMouseUp = () => {
                updateWidgetValue(finalValue);
                window.removeEventListener('mousemove', onMouseMove);
                window.removeEventListener('mouseup', onMouseUp);
            };
            
            window.addEventListener('mousemove', onMouseMove);
            window.addEventListener('mouseup', onMouseUp);
        });
    
        knob.addEventListener('wheel', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            const delta = e.deltaY > 0 ? -1 : 1;
            const newValue = currentValue + delta * param.step;
            updateWidgetValue(newValue);
        });
    
        controlGroup.appendChild(label);
        controlGroup.appendChild(knob);
        
        this.numberInputs[param.name] = { knob, update: updateKnobVisuals, getCurrentValue: () => currentValue };
        
        updateKnobVisuals(currentValue);
    
        return controlGroup;
    }

    createControlsSection() {
        const controlsSection = document.createElement('div');
        controlsSection.className = 'semantic-eq-controls-section';

        const controlsRow = document.createElement('div');
        controlsRow.className = 'controls-row';

        const knobParams = [
            { name: 'flux_guidance', label: 'Guidance', min: 0, max: 10, step: 0.1, precision: 1, default: 2.7, color: 'bright-orange' },
            { name: 'multiplier', label: 'Multiplier', min: -2, max: 2, step: 0.05, precision: 2, default: 1.0 },
            { name: 'dry_wet_mix', label: 'Dry/Wet', min: 0, max: 1, step: 0.01, precision: 2, default: 1.0 }
        ];

        knobParams.forEach(param => {
            const knobElement = this.createKnob(param);
            controlsRow.appendChild(knobElement);
        });

        controlsSection.appendChild(controlsRow);
        this.container.appendChild(controlsSection);
    }

    createTabs() {
        const tabsContainer = document.createElement('div');
        tabsContainer.className = 'semantic-eq-tabs';
        
        const tabs = [
            { id: 'shift', label: 'SHIFT' },
            { id: 'inject', label: 'INJECT' },
            { id: 'eq_bands', label: 'EQ' },
            { id: 'magic', label: 'MAGIC' },
            { id: 'manual', label: 'MAN' }
        ];
        
        tabs.forEach(({ id, label }) => {
            const tab = document.createElement('button');
            tab.className = 'semantic-eq-tab';
            tab.textContent = label;
            tab.dataset.type = id;
            
            if (id === this.activeTab) {
                tab.classList.add('active');
            }
            
            tab.addEventListener('click', () => {
                console.log(`[FluxSemanticEncoderUI] Switching to tab: ${id}`);
                this.switchTab(id);
            });
            
            tabsContainer.appendChild(tab);
        });
        
        this.container.appendChild(tabsContainer);
    }

    createToggleRow(parentSection, labelText, widgetName) {
        const toggleContainer = document.createElement('div');
        toggleContainer.className = 'param-container';
    
        const toggleRow = document.createElement('div');
        toggleRow.className = 'param-row param-toggle-row';
    
        const label = document.createElement('div');
        label.className = 'param-label';
        label.textContent = labelText;
    
        const led = document.createElement('div');
        led.className = 'param-led';
    
        toggleRow.appendChild(label);
        toggleRow.appendChild(led);
        toggleContainer.appendChild(toggleRow);
        parentSection.appendChild(toggleContainer);
    
        toggleRow.addEventListener('click', () => {
            const widget = this.node.widgets?.find(w => w.name === widgetName);
            if (widget) {
                widget.value = !widget.value;
                this.updateSectionVisualState(parentSection, led, widget.value);
                this.node.setDirtyCanvas(true, true);
            }
        });
    
        return led;
    }
    
    updateSectionVisualState(sectionElement, ledElement, isEnabled) {
        if (ledElement) {
            ledElement.classList.toggle('active', isEnabled);
        }
        sectionElement.classList.toggle('controls-disabled', !isEnabled);
    }

    createShiftSection(parent) {
        const shiftSection = document.createElement('div');
        shiftSection.id = 'shift-section';
        shiftSection.className = 'shift-section';
    
        const led = this.createToggleRow(shiftSection, 'Enable Semantic Shift', 'enable_semantic_shift');
        
        // Target Prompt selector
        const targetContainer = document.createElement('div');
        targetContainer.className = 'param-container';
        
        const targetRow = document.createElement('div');
        targetRow.className = 'param-row';
        
        const targetLabel = document.createElement('div');
        targetLabel.className = 'param-label';
        targetLabel.textContent = 'Target Prompt';
        
        const targetSelect = document.createElement('select');
        targetSelect.className = 'param-select';
        
        ['T5-XXL_Only', 'CLIP-L_Only', 'Both'].forEach(option => {
            const optionElement = document.createElement('option');
            optionElement.value = option;
            optionElement.textContent = option;
            targetSelect.appendChild(optionElement);
        });
        
        targetSelect.value = this.node.widgets?.find(w => w.name === 'target_prompt')?.value || 'Both';
        
        targetSelect.addEventListener('change', () => {
            const widget = this.node.widgets?.find(w => w.name === 'target_prompt');
            if (widget) {
                widget.value = targetSelect.value;
                this.node.setDirtyCanvas(true, true);
            }
        });
        
        this.numberInputs['target_prompt'] = targetSelect;
        targetRow.appendChild(targetLabel);
        targetRow.appendChild(targetSelect);
        targetContainer.appendChild(targetRow);
        shiftSection.appendChild(targetContainer);


        // Shift parameters
        const shiftParams = [
            { name: 'semantic_shift_strength', label: 'Strength', min: 0, max: 1, step: 0.01, default: 0.5 },
            { name: 'semantic_shift_distance', label: 'Distance', min: 0, max: 1, step: 0.01, default: 0.2 },
            { name: 'semantic_shift_randomness', label: 'Randomness', min: 0, max: 1, step: 0.01, default: 0.2 },
            { name: 'semantic_shift_max_neighbors', label: 'Max Neighbors', min: 1, max: 50, step: 1, default: 50 }
        ];

        shiftParams.forEach(param => {
            const container = document.createElement('div');
            container.className = 'magic-slider-container';

            const label = document.createElement('div');
            label.className = 'semantic-eq-band-label';
            label.textContent = param.label;
            container.appendChild(label);

            const sliderRow = document.createElement('div');
            sliderRow.className = 'magic-slider-row';

            const slider = document.createElement('input');
            slider.className = 'magic-slider';
            slider.type = 'range';
            slider.min = param.min;
            slider.max = param.max;
            slider.step = param.step;
            slider.value = this.node.widgets?.find(w => w.name === param.name)?.value || param.default;
            
            const valueDisplay = document.createElement('div');
            valueDisplay.className = 'magic-value';
            const precision = param.step < 1 ? 2 : 0;
            valueDisplay.textContent = parseFloat(slider.value).toFixed(precision);

            slider.addEventListener('input', () => {
                const widget = this.node.widgets?.find(w => w.name === param.name);
                if (widget) {
                    const value = param.step < 1 ? parseFloat(slider.value) : parseInt(slider.value, 10);
                    widget.value = value;
                    valueDisplay.textContent = value.toFixed(precision);
                    this.node.setDirtyCanvas(true, true);
                }
            });
            
            this.numberInputs[param.name] = slider;
            sliderRow.appendChild(slider);
            sliderRow.appendChild(valueDisplay);
            container.appendChild(sliderRow);
            shiftSection.appendChild(container);
        });

        parent.appendChild(shiftSection);
        
        // Set initial state
        const isEnabled = this.node.widgets?.find(w => w.name === 'enable_semantic_shift')?.value || false;
        this.updateSectionVisualState(shiftSection, led, isEnabled);
    }

    createInjectSection(parent) {
        const injectSection = document.createElement('div');
        injectSection.id = 'inject-section';
        injectSection.className = 'inject-section hidden';

        const led = this.createToggleRow(injectSection, 'Enable Token Injection', 'enable_token_injection');

        // Injection Mode selector
        const modeContainer = document.createElement('div');
        modeContainer.className = 'param-container';
        
        const modeRow = document.createElement('div');
        modeRow.className = 'param-row';
        
        const modeLabel = document.createElement('div');
        modeLabel.className = 'param-label';
        modeLabel.textContent = 'Injection Mode';
        
        const modeSelect = document.createElement('select');
        modeSelect.className = 'param-select';
        
        ['ADD', 'REPLACE', 'MIX'].forEach(option => {
            const optionElement = document.createElement('option');
            optionElement.value = option;
            optionElement.textContent = option;
            modeSelect.appendChild(optionElement);
        });
        
        modeSelect.value = this.node.widgets?.find(w => w.name === 'injection_mode')?.value || 'ADD';
        
        modeSelect.addEventListener('change', () => {
            const widget = this.node.widgets?.find(w => w.name === 'injection_mode');
            if (widget) {
                widget.value = modeSelect.value;
                this.node.setDirtyCanvas(true, true);
            }
        });
        
        this.numberInputs['injection_mode'] = modeSelect;
        modeRow.appendChild(modeLabel);
        modeRow.appendChild(modeSelect);
        modeContainer.appendChild(modeRow);
        injectSection.appendChild(modeContainer);

        // Number of tokens to inject
        const param = { name: 'num_tokens_to_inject', label: 'Tokens to Inject', min: 0, max: 100, step: 1, default: 5 };
        
        const container = document.createElement('div');
        container.className = 'magic-slider-container';

        const label = document.createElement('div');
        label.className = 'semantic-eq-band-label';
        label.textContent = param.label;
        container.appendChild(label);

        const sliderRow = document.createElement('div');
        sliderRow.className = 'magic-slider-row';

        const slider = document.createElement('input');
        slider.className = 'magic-slider';
        slider.type = 'range';
        slider.min = param.min;
        slider.max = param.max;
        slider.step = param.step;
        slider.value = this.node.widgets?.find(w => w.name === param.name)?.value || param.default;
        
        const valueDisplay = document.createElement('div');
        valueDisplay.className = 'magic-value';
        valueDisplay.textContent = slider.value;

        slider.addEventListener('input', () => {
            const widget = this.node.widgets?.find(w => w.name === param.name);
            if (widget) {
                widget.value = parseInt(slider.value, 10);
                valueDisplay.textContent = slider.value;
                this.node.setDirtyCanvas(true, true);
            }
        });
        
        this.numberInputs[param.name] = slider;
        sliderRow.appendChild(slider);
        sliderRow.appendChild(valueDisplay);
        container.appendChild(sliderRow);
        injectSection.appendChild(container);

        parent.appendChild(injectSection);
        
        // Set initial state
        const isEnabled = this.node.widgets?.find(w => w.name === 'enable_token_injection')?.value || false;
        this.updateSectionVisualState(injectSection, led, isEnabled);
    }

    createEQBandsSection(parent) {
        const eqSection = document.createElement('div');
        eqSection.id = 'eq-bands-section';
        eqSection.className = 'eq-section hidden';

        this.canvas = document.createElement('canvas');
        this.canvas.className = 'semantic-eq-eq-canvas';
        this.canvas.width = 460;
        this.canvas.height = 180;
        this.ctx = this.canvas.getContext('2d');

        // Add event listeners
        this.canvas.addEventListener('mousedown', this.handleMouseDown);
        this.canvas.addEventListener('mousemove', this.handleMouseMove);
        this.canvas.addEventListener('mouseup', this.handleMouseUp);
        this.canvas.addEventListener('mouseleave', this.handleMouseLeave);
        this.canvas.addEventListener('wheel', this.handleWheel, { passive: false });

        eqSection.appendChild(this.canvas);
        parent.appendChild(eqSection);
    }

    createMagicSection(parent) {
        const magicSection = document.createElement('div');
        magicSection.id = 'magic-section';
        magicSection.className = 'magic-section hidden';

        ['magic_scale_min', 'magic_scale_max'].forEach(param => {
            const container = document.createElement('div');
            container.className = 'magic-slider-container';

            const label = document.createElement('div');
            label.className = 'semantic-eq-band-label';
            label.textContent = param.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            container.appendChild(label);

            const sliderRow = document.createElement('div');
            sliderRow.className = 'magic-slider-row';

            const slider = document.createElement('input');
            slider.className = 'magic-slider';
            slider.type = 'range';
            slider.min = -2;
            slider.max = 3;
            slider.step = 0.05;
            slider.value = this.node.widgets?.find(w => w.name === param)?.value || 
                          (param === 'magic_scale_min' ? 0.8 : 1.2);
            slider.title = `Set ${param.replace(/_/g, ' ')}`;
            
            const valueDisplay = document.createElement('div');
            valueDisplay.className = 'magic-value';
            valueDisplay.textContent = parseFloat(slider.value).toFixed(2);

            slider.addEventListener('input', () => {
                const widget = this.node.widgets?.find(w => w.name === param);
                if (widget) {
                    widget.value = parseFloat(slider.value);
                    valueDisplay.textContent = parseFloat(slider.value).toFixed(2);
                    this.node.setDirtyCanvas(true, true);
                }
            });
            
            this.numberInputs[param] = slider;
            sliderRow.appendChild(slider);
            sliderRow.appendChild(valueDisplay);
            container.appendChild(sliderRow);
            magicSection.appendChild(container);
        });

        parent.appendChild(magicSection);
    }

    createManualSection(parent) {
        const manualSection = document.createElement('div');
        manualSection.id = 'manual-section';
        manualSection.className = 'manual-section hidden';
        
        const helpText = document.createElement('div');
        helpText.style.cssText = `
            color: var(--text-gray);
            font-family: 'Orbitron', monospace;
            font-size: 12px;
            text-align: center;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            border: 1px solid rgba(119, 0, 255, 0.2);
        `;
        helpText.innerHTML = `
            <strong>Manual Keywords Mode</strong><br>
            Use syntax: (word:SCALE) in T5-XXL prompt<br>
            Example: "a beautiful (landscape:1.3) with (vibrant:0.8) colors"
        `;
        
        manualSection.appendChild(helpText);
        parent.appendChild(manualSection);
    }

    setupResizeObserver() {
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }
        
        this.resizeObserver = new ResizeObserver(this.handleResize);
        this.resizeObserver.observe(this.container);
    }

    handleResize() {
        if (this.canvas && this.activeTab === 'eq_bands') {
            // Maintain aspect ratio and redraw
            requestAnimationFrame(() => {
                this.drawEQCanvas();
            });
        }
    }

    initializeEQPoints() {
        const defaultCenters = [0.25, 0.5, 0.75, 1.0];
        this.eqPoints = [];
        
        for (let i = 0; i < MAX_EQ_BANDS; i++) {
            this.eqPoints[i] = {
                center: defaultCenters[i],
                gain: 1.0,
                q_factor: 1.0,
                isDragging: false
            };
        }
        this.updateWidgetsFromCanvas();
    }

    drawEQCanvas() {
        if (!this.ctx || !this.canvas) {
            console.error('[FluxSemanticEncoderUI] Canvas context or element not available');
            return;
        }
        
        const { width, height } = this.canvas;
        this.ctx.clearRect(0, 0, width, height);

        // Enhanced grid with better styling
        this.ctx.strokeStyle = 'rgba(119, 0, 255, 0.2)';
        this.ctx.lineWidth = 0.5;
        
        // Vertical grid lines
        for (let x = 0; x <= width; x += width / 8) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, height);
            this.ctx.stroke();
        }
        
        // Horizontal grid lines
        for (let y = 0; y <= height; y += height / 10) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(width, y);
            this.ctx.stroke();
        }

        // Enhanced labels with better positioning
        this.ctx.fillStyle = 'rgba(154, 112, 255, 0.9)';
        this.ctx.font = 'bold 10px Orbitron, monospace';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'top';
        
        // Frequency labels
        const freqLabels = ['0', '0.25', '0.5', '0.75', '1.0'];
        for (let i = 0; i < 5; i++) {
            const x = (i * width) / 4;
            this.ctx.fillText(freqLabels[i], x, 5);
        }
        
        // Gain labels
        this.ctx.textAlign = 'left';
        this.ctx.textBaseline = 'middle';
        const gainLabels = ['3.0', '2.5', '2.0', '1.5', '1.0', '0.5', '0.0'];
        for (let i = 0; i < gainLabels.length; i++) {
            const y = (i * height) / (gainLabels.length - 1);
            this.ctx.fillText(gainLabels[i], 5, y);
        }

        // Draw EQ response curve with enhanced styling
        this.ctx.beginPath();
        this.ctx.strokeStyle = '#2ecc71';
        this.ctx.lineWidth = 3;
        
        const steps = 300;
        const baselineGain = 1.0;
        
        for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            const x = t * width;
            let totalGain = 0;
            let totalWeight = 0;

            this.eqPoints.forEach(point => {
                const centerX = point.center;
                const q = Math.max(0.01, point.q_factor);
                const sigma = 0.15 / q;
                const distance = t - centerX;
                const weight = Math.exp(-(distance * distance) / (2 * sigma * sigma));
                totalGain += point.gain * weight;
                totalWeight += weight;
            });

            const avgGain = totalWeight > 0 ? totalGain / totalWeight : baselineGain;
            const y = ((3 - avgGain) / 3) * height;
            
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        
        // Enhanced curve styling
        this.ctx.shadowBlur = 15;
        this.ctx.shadowColor = 'rgba(46, 204, 113, 0.8)';
        this.ctx.stroke();
        this.ctx.shadowBlur = 0;

        // Draw control points with enhanced styling
        this.eqPoints.forEach((point, index) => {
            const x = point.center * width;
            const y = ((3 - point.gain) / 3) * height;
            
            // Point shadow/glow
            this.ctx.beginPath();
            this.ctx.arc(x, y, 12, 0, Math.PI * 2);
            this.ctx.fillStyle = 'rgba(46, 204, 113, 0.3)';
            this.ctx.fill();
            
            // Main point
            this.ctx.beginPath();
            this.ctx.arc(x, y, 8, 0, Math.PI * 2);
            
            // Color based on state
            if (this.selectedPoint === index) {
                this.ctx.fillStyle = '#ff6b6b';
                this.ctx.shadowBlur = 10;
                this.ctx.shadowColor = '#ff6b6b';
            } else if (this.hoveredPoint === index) {
                this.ctx.fillStyle = '#4ecdc4';
                this.ctx.shadowBlur = 8;
                this.ctx.shadowColor = '#4ecdc4';
            } else {
                this.ctx.fillStyle = '#2ecc71';
                this.ctx.shadowBlur = 6;
                this.ctx.shadowColor = '#2ecc71';
            }
            
            this.ctx.fill();
            this.ctx.shadowBlur = 0;
            
            // Point border
            this.ctx.strokeStyle = '#ffffff';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
            
            // Band number label
            this.ctx.fillStyle = '#000000';
            this.ctx.font = 'bold 10px Orbitron, monospace';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText((index + 1).toString(), x, y);
            
            // Q factor label
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '9px Orbitron, monospace';
            this.ctx.textBaseline = 'bottom';
            this.ctx.fillText(`Q: ${point.q_factor.toFixed(1)}`, x, y - 12);
            
            // Gain label
            this.ctx.textBaseline = 'top';
            this.ctx.fillText(`${point.gain.toFixed(2)}`, x, y + 12);
        });
    }

    handleMouseDown(e) {
        if (!this.canvas) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        const mouseX = (e.clientX - rect.left) * scaleX;
        const mouseY = (e.clientY - rect.top) * scaleY;

        this.selectedPoint = null;
        this.eqPoints.forEach((point, index) => {
            const x = point.center * this.canvas.width;
            const y = ((3 - point.gain) / 3) * this.canvas.height;
            const distance = Math.hypot(mouseX - x, mouseY - y);
            
            if (distance < 20) {
                this.selectedPoint = index;
                this.eqPoints[index].isDragging = true;
                console.log(`[FluxSemanticEncoderUI] Selected point ${index + 1}`);
            }
        });
        
        if (this.selectedPoint !== null) {
            this.drawEQCanvas();
        }
    }

    handleMouseMove(e) {
        if (!this.canvas) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        const mouseX = (e.clientX - rect.left) * scaleX;
        const mouseY = (e.clientY - rect.top) * scaleY;

        // Update hover state
        let newHoveredPoint = null;
        this.eqPoints.forEach((point, index) => {
            const x = point.center * this.canvas.width;
            const y = ((3 - point.gain) / 3) * this.canvas.height;
            const distance = Math.hypot(mouseX - x, mouseY - y);
            
            if (distance < 20) {
                newHoveredPoint = index;
            }
        });
        
        if (newHoveredPoint !== this.hoveredPoint) {
            this.hoveredPoint = newHoveredPoint;
            this.canvas.style.cursor = this.hoveredPoint !== null ? 'pointer' : 'crosshair';
            this.drawEQCanvas();
        }

        // Handle dragging
        if (this.selectedPoint === null || !this.eqPoints[this.selectedPoint].isDragging) return;

        const point = this.eqPoints[this.selectedPoint];
        
        // Update center position with constraints
        let newCenter = mouseX / this.canvas.width;
        const minCenter = this.selectedPoint > 0 ? this.eqPoints[this.selectedPoint - 1].center + 0.05 : 0;
        const maxCenter = this.selectedPoint < this.eqPoints.length - 1 ? this.eqPoints[this.selectedPoint + 1].center - 0.05 : 1;
        newCenter = Math.max(minCenter, Math.min(maxCenter, newCenter));
        point.center = newCenter;
        
        // Update gain with proper mapping
        const rawGain = 3 - (mouseY / this.canvas.height) * 3;
        point.gain = Math.max(0, Math.min(3, rawGain));

        this.updateWidgetsFromCanvas();
        this.drawEQCanvas();
        this.node.setDirtyCanvas(true, true);
    }

    handleMouseUp() {
        if (this.selectedPoint !== null) {
            console.log(`[FluxSemanticEncoderUI] Released point ${this.selectedPoint + 1}`);
            this.eqPoints[this.selectedPoint].isDragging = false;
            this.selectedPoint = null;
            this.drawEQCanvas();
        }
    }

    handleMouseLeave() {
        this.hoveredPoint = null;
        this.canvas.style.cursor = 'crosshair';
        
        if (this.selectedPoint !== null) {
            this.eqPoints[this.selectedPoint].isDragging = false;
            this.selectedPoint = null;
        }
        this.drawEQCanvas();
    }

    handleWheel(e) {
        e.preventDefault();
        if (this.hoveredPoint === null) return;
        
        const point = this.eqPoints[this.hoveredPoint];
        const delta = e.deltaY > 0 ? -0.2 : 0.2;
        point.q_factor = Math.max(0.01, Math.min(10, point.q_factor + delta));
        
        console.log(`[FluxSemanticEncoderUI] Adjusted Q factor for band ${this.hoveredPoint + 1}: ${point.q_factor.toFixed(2)}`);
        
        this.updateWidgetsFromCanvas();
        this.drawEQCanvas();
        this.node.setDirtyCanvas(true, true);
    }

    updateWidgetsFromCanvas() {
        this.eqPoints.forEach((point, index) => {
            const centerWidget = this.node.widgets?.find(w => w.name === `band_${index + 1}_center`);
            const gainWidget = this.node.widgets?.find(w => w.name === `band_${index + 1}_gain`);
            const qWidget = this.node.widgets?.find(w => w.name === `band_${index + 1}_q_factor`);
            
            if (centerWidget) centerWidget.value = Math.round(point.center * 100) / 100;
            if (gainWidget) gainWidget.value = Math.round(point.gain * 100) / 100;
            if (qWidget) qWidget.value = Math.round(point.q_factor * 100) / 100;
        });
    }

    toggleCollapse() {
        this.isCollapsed = !this.isCollapsed;
        this.applyCollapseState();
    }
    
    applyCollapseState(instant = false) {
        if (!this.contentContainer || !this.collapseBar) return;
    
        this.contentContainer.classList.toggle('collapsed', this.isCollapsed);
        this.collapseBar.classList.toggle('collapsed', this.isCollapsed);
    
        if (instant) {
            this.setNodeSizeInstantly();
        } else {
            this.animateNodeSize();
        }
    }

    switchTab(type) {
        if (this.activeTab === type) return;
        
        this.activeTab = type;
        
        // Update tab visual states
        this.container.querySelectorAll('.semantic-eq-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.type === type);
        });
        
        // Update visibility and node size instantly
        this.updateTabVisibility();
        this.setNodeSizeInstantly();
        
        // Update modification mode widget
        const modeMap = {
            'shift': 'EQ (Positional)', // Will be handled by enable_semantic_shift
            'inject': 'EQ (Positional)', // Will be handled by enable_token_injection
            'eq_bands': 'EQ (Positional)',
            'magic': 'Magic (Random Scale)',
            'manual': 'Manual Keywords (word:SCALE)'
        };
        
        const widget = this.node.widgets?.find(w => w.name === 'tensor_modification_mode');
        if (widget) {
            widget.value = modeMap[type] || 'EQ (Positional)';
        }
        
        // Redraw canvas if on EQ tab
        if (type === 'eq_bands' && this.canvas) {
            requestAnimationFrame(() => {
                this.drawEQCanvas();
            });
        }
        
        this.syncInternalStateFromWidgets();
        this.node.setDirtyCanvas(true, true);
        
        console.log(`[FluxSemanticEncoderUI] Switched to tab: ${type}`);
    }

    updateTabVisibility() {
        if (!this.contentContainer) return;
        const sections = {
            'shift': this.contentContainer.querySelector('#shift-section'),
            'inject': this.contentContainer.querySelector('#inject-section'),
            'eq_bands': this.contentContainer.querySelector('#eq-bands-section'),
            'magic': this.contentContainer.querySelector('#magic-section'),
            'manual': this.contentContainer.querySelector('#manual-section')
        };
        
        Object.keys(sections).forEach(sectionKey => {
            const section = sections[sectionKey];
            if (section) {
                if (sectionKey === this.activeTab) {
                    section.className = section.className.replace('hidden', 'visible');
                    section.style.display = 'block';
                } else {
                    section.className = section.className.replace('visible', 'hidden');
                    section.style.display = 'none';
                }
            }
        });
    }

    getTargetHeight() {
        if (this.isCollapsed) {
            return 380; // New collapsed height with knobs
        }

        const sizeMap = {
            'shift': 880,
            'inject': 615,
            'eq_bands': 590,
            'magic': 590,
            'manual': 500
        };
        
        return sizeMap[this.activeTab] || 650;
    }

    setNodeSizeInstantly() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        const height = this.getTargetHeight();
        this.node.setSize([520, height]);
        this.node.setDirtyCanvas(true, true);
    }
    
    animateNodeSize() {
        const startHeight = this.node.size[1];
        const endHeight = this.getTargetHeight();

        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }

        if (startHeight === endHeight) {
            return;
        }
        
        const duration = 400; // ms, to match CSS transition
        const startTime = performance.now();

        const animate = (currentTime) => {
            const elapsedTime = currentTime - startTime;
            const progress = Math.min(elapsedTime / duration, 1);
            
            // Ease-in-out function for smooth acceleration and deceleration
            const easedProgress = progress < 0.5 
                ? 4 * progress * progress * progress 
                : 1 - Math.pow(-2 * progress + 2, 3) / 2;

            const currentHeight = startHeight + (endHeight - startHeight) * easedProgress;
            
            this.node.setSize([this.node.size[0], currentHeight]);
            this.node.setDirtyCanvas(true, true);
            
            if (progress < 1) {
                this.animationFrameId = requestAnimationFrame(animate);
            } else {
                // Ensure final size is exact
                this.node.setSize([this.node.size[0], endHeight]);
                this.node.setDirtyCanvas(true, true);
                this.animationFrameId = null;
            }
        };

        this.animationFrameId = requestAnimationFrame(animate);
    }


    syncInternalStateFromWidgets() {
        if (!this.node.widgets || !this.isInitialized) return;
        
        try {
            // Sync knobs
            ['flux_guidance', 'multiplier', 'dry_wet_mix'].forEach(param => {
                const widget = this.node.widgets.find(w => w.name === param);
                const knobControl = this.numberInputs[param];
                if (widget && knobControl && widget.value !== knobControl.getCurrentValue()) {
                    knobControl.update(widget.value);
                }
            });
            
            // Sync magic sliders
            ['magic_scale_min', 'magic_scale_max'].forEach(param => {
                const widget = this.node.widgets.find(w => w.name === param);
                if (widget && this.numberInputs[param]) {
                    this.numberInputs[param].value = widget.value;
                    const valueDisplay = this.numberInputs[param].parentElement.querySelector('.magic-value');
                    if (valueDisplay) {
                        valueDisplay.textContent = Number(widget.value).toFixed(2);
                    }
                }
            });

            // Sync shift section state
            const shiftWidget = this.node.widgets.find(w => w.name === 'enable_semantic_shift');
            const shiftSection = this.contentContainer.querySelector('#shift-section');
            if(shiftWidget && shiftSection) {
                const led = shiftSection.querySelector('.param-led');
                this.updateSectionVisualState(shiftSection, led, shiftWidget.value);
            }

            // Sync inject section state
            const injectWidget = this.node.widgets.find(w => w.name === 'enable_token_injection');
            const injectSection = this.contentContainer.querySelector('#inject-section');
            if(injectWidget && injectSection) {
                const led = injectSection.querySelector('.param-led');
                this.updateSectionVisualState(injectSection, led, injectWidget.value);
            }
            
            // Sync EQ points
            for (let i = 1; i <= MAX_EQ_BANDS; i++) {
                const centerWidget = this.node.widgets.find(w => w.name === `band_${i}_center`);
                const gainWidget = this.node.widgets.find(w => w.name === `band_${i}_gain`);
                const qWidget = this.node.widgets.find(w => w.name === `band_${i}_q_factor`);
                
                if (centerWidget && gainWidget && qWidget && this.eqPoints[i - 1]) {
                    if (typeof centerWidget.value === 'number' && centerWidget.value >= 0 && centerWidget.value <= 1) {
                        this.eqPoints[i - 1].center = centerWidget.value;
                    }
                    if (typeof gainWidget.value === 'number' && gainWidget.value >= 0 && gainWidget.value <= 3) {
                        this.eqPoints[i - 1].gain = gainWidget.value;
                    }
                    if (typeof qWidget.value === 'number' && qWidget.value >= 0.01 && qWidget.value <= 10) {
                        this.eqPoints[i - 1].q_factor = qWidget.value;
                    }
                }
            }
            
            // Redraw if necessary
            if (this.activeTab === 'eq_bands' && this.canvas) {
                requestAnimationFrame(() => {
                    this.drawEQCanvas();
                });
            }
            
        } catch (err) {
            console.error('[FluxSemanticEncoderUI] Error syncing state:', err);
        }
    }

    serialize() {
        return {
            activeTab: this.activeTab,
            isCollapsed: this.isCollapsed,
            eqPoints: this.eqPoints.map(p => ({
                center: p.center,
                gain: p.gain,
                q_factor: p.q_factor
            }))
        };
    }

    deserialize(data) {
        if (!data) return;
        
        // When deserializing, respect the saved state, otherwise keep the default (true).
        if (typeof data.isCollapsed === 'boolean') {
            this.isCollapsed = data.isCollapsed;
        } else {
            this.isCollapsed = true; // Fallback to collapsed if not specified in old saves
        }

        if (data.activeTab && data.activeTab !== this.activeTab) {
            this.activeTab = data.activeTab;
        }
        
        if (data.eqPoints && Array.isArray(data.eqPoints)) {
            this.eqPoints = data.eqPoints.map(p => ({
                center: p.center || 0.5,
                gain: p.gain || 1.0,
                q_factor: p.q_factor || 1.0,
                isDragging: false
            }));
        }
        
        // Apply changes instantly on load, without animation
        this.applyCollapseState(true); // Pass true for instant
        this.switchTab(this.activeTab);
        this.syncInternalStateFromWidgets();
        this.setNodeSizeInstantly();
        
        console.log('[FluxSemanticEncoderUI] Deserialized state:', data);
    }

    destroy() {
        // Clean up event listeners and observers
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }
        
        if (this.canvas) {
            this.canvas.removeEventListener('mousedown', this.handleMouseDown);
            this.canvas.removeEventListener('mousemove', this.handleMouseMove);
            this.canvas.removeEventListener('mouseup', this.handleMouseUp);
            this.canvas.removeEventListener('mouseleave', this.handleMouseLeave);
            this.canvas.removeEventListener('wheel', this.handleWheel);
        }

        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }
        
        if (this.container) {
            this.container.remove();
        }
        
        console.log(`[FluxSemanticEncoderUI] Destroyed UI for node ${this.node.id}`);
    }
}

// Register the extension
app.registerExtension({
    name: "Comfy.FluxSemanticEncoder.UI.Enhanced",
    async nodeCreated(node) {
        if (node.comfyClass === "FluxSemanticEncoder") {
            // Set node appearance
            node.bgcolor = "#000000";
            node.color = "#000000";
            node.title_style = "color: #7700ff; font-family: 'Orbitron', monospace; font-weight: bold;";
            
            // Clean up existing instance
            if (node.fluxSemanticEncoderUIInstance) {
                node.fluxSemanticEncoderUIInstance.destroy();
            }
            
            // Create new instance with delay to ensure proper initialization
            setTimeout(() => {
                node.fluxSemanticEncoderUIInstance = new FluxSemanticEncoderUI(node);
            }, 100);

            // Enhanced serialization
            const originalOnSerialize = node.onSerialize;
            node.onSerialize = function() {
                let comfyData = originalOnSerialize ? originalOnSerialize.call(node) : {};
                if (node.fluxSemanticEncoderUIInstance) {
                    comfyData.fluxSemanticEncoderUIData = node.fluxSemanticEncoderUIInstance.serialize();
                }
                return comfyData;
            };

            // Enhanced deserialization
            const originalOnDeserialize = node.onDeserialize;
            node.onDeserialize = function(data) {
                if (originalOnDeserialize) {
                    originalOnDeserialize.call(node, data);
                }
                
                // Delay deserialization to ensure UI is ready
                setTimeout(() => {
                    if (node.fluxSemanticEncoderUIInstance && data && data.fluxSemanticEncoderUIData) {
                        node.fluxSemanticEncoderUIInstance.deserialize(data.fluxSemanticEncoderUIData);
                    }
                }, 200);
            };

            // Handle node removal
            const originalOnRemove = node.onRemove;
            node.onRemove = function() {
                if (node.fluxSemanticEncoderUIInstance) {
                    node.fluxSemanticEncoderUIInstance.destroy();
                }
                if (originalOnRemove) {
                    originalOnRemove.call(node);
                }
            };
        }
    },
    
    async setup() {
        console.log("[FluxSemanticEncoderUI Enhanced] Extension registered successfully.");
    }
});

console.log("FluxSemanticEncoder_UI.js (Enhanced V7): Script fully loaded and registered.");
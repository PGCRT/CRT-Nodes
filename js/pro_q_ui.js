import { app } from "../../scripts/app.js";

console.log("pro_q_ui.js: Professional Parametric EQ UI loading...");

// ... (Your entire CSS string remains unchanged here) ...
const CSS_STYLES_PARAMETRIC_EQ = `
@font-face {
    font-family: 'Orbitron';
    src: url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&display=swap') format('woff2'),
         local('Orbitron');
}

:root {
    --eq-primary: #00d4ff;
    --eq-primary-light: #40e0ff;
    --eq-secondary: #ff6b00;
    --eq-active: #00ff88;
    --eq-inactive: #ff4444;
    --eq-background: #000000;
    --eq-surface: #1a1a1a;
    --eq-border: #333333;
    --eq-text: #ffffff;
    --eq-text-dim: #888888;
    --eq-grid: rgba(0, 212, 255, 0.2);
    --eq-gradient: linear-gradient(145deg, #0a0a0a, #1a1a1a);
}

/* Make all text non-selectable */
.parametric-eq-container * {
    user-select: none !important;
    -webkit-user-select: none !important;
    -moz-user-select: none !important;
    -ms-user-select: none !important;
}

/* Main container */
.parametric-eq-node-custom-widget {
    position: relative !important;
    box-sizing: border-box !important;
    width: 100% !important;
    min-height: 500px !important;
    padding: 0px !important;
    margin: 0 !important;
    overflow: visible !important;
    display: block !important;
    visibility: visible !important;
    z-index: 1 !important;
    top: -17px !important;
}

.parametric-eq-container {
    background: #000000;
    border: 2px solid var(--eq-primary);
    border-radius: 20px;
    padding: 20px;
    margin: 0;
    width: 100%;
    max-width: 900px;
    box-sizing: border-box;
    min-height: 480px;
    box-shadow: 
        0 12px 40px rgba(0, 212, 255, 0.4),
        inset 0 2px 0 rgba(255, 255, 255, 0.1),
        0 0 60px rgba(0, 212, 255, 0.2);
    display: flex;
    flex-direction: column;
    align-items: stretch;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(10px);
}

.parametric-eq-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(circle at 30% 20%, rgba(0, 212, 255, 0.1), transparent 50%),
                radial-gradient(circle at 70% 80%, rgba(0, 255, 136, 0.05), transparent 50%);
    pointer-events: none;
    z-index: 0;
}

.parametric-eq-container > * {
    position: relative;
    z-index: 1;
}

/* Title */
.parametric-eq-title {
    color: var(--eq-primary);
    font-family: 'Orbitron', monospace;
    font-size: 22px;
    font-weight: 700;
    text-align: center;
    margin: 0 0 20px 0;
    padding: 12px 0;
    text-shadow: 
        1px 1px 2px rgba(0, 0, 0, 0.8), 
        0 0 30px var(--eq-primary), 
        0 0 8px var(--eq-primary-light);
    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.15), transparent);
    border-radius: 12px;
    animation: eqTitleGlow 4s ease-in-out infinite;
    position: relative;
}

.parametric-eq-title::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    width: 60%;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--eq-primary), transparent);
    transform: translateX(-50%);
    border-radius: 1px;
}

@keyframes eqTitleGlow {
    0%, 100% {
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8), 0 0 30px var(--eq-primary), 0 0 8px var(--eq-primary-light);
        transform: scale(1);
    }
    50% {
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8), 0 0 40px var(--eq-primary-light), 0 0 15px var(--eq-primary);
        transform: scale(1.02);
        color: var(--eq-primary-light);
    }
}

/* Preset controls */
.parametric-eq-presets {
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-bottom: 16px;
    align-items: center;
    padding: 12px;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 12px;
    border: 1px solid rgba(0, 212, 255, 0.2);
}

.preset-select {
    background: var(--eq-background) !important;
    border: 2px solid var(--eq-border) !important;
    border-radius: 8px !important;
    color: var(--eq-text) !important;
    padding: 6px 10px !important;
    font-size: 11px !important;
    font-family: 'Orbitron', monospace !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    min-width: 120px !important;
}

.preset-select:focus {
    border-color: var(--eq-primary) !important;
    box-shadow: 0 0 8px rgba(0, 212, 255, 0.3) !important;
    outline: none !important;
}

.preset-select option {
    background: var(--eq-background) !important;
    color: var(--eq-text) !important;
}

.preset-input {
    background: var(--eq-background) !important;
    border: 2px solid var(--eq-border) !important;
    border-radius: 8px !important;
    color: var(--eq-text) !important;
    padding: 6px 10px !important;
    font-size: 11px !important;
    font-family: 'Orbitron', monospace !important;
    width: 120px !important;
    transition: all 0.3s ease !important;
}

.preset-input:focus {
    border-color: var(--eq-primary) !important;
    box-shadow: 0 0 8px rgba(0, 212, 255, 0.3) !important;
    outline: none !important;
}

.preset-button {
    background: linear-gradient(45deg, var(--eq-background), rgba(0, 212, 255, 0.1)) !important;
    color: var(--eq-primary) !important;
    border: 2px solid var(--eq-border) !important;
    padding: 6px 12px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    font-size: 11px !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
}

.preset-button:hover {
    background: linear-gradient(45deg, rgba(0, 212, 255, 0.1), rgba(0, 212, 255, 0.2)) !important;
    border-color: var(--eq-primary) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 212, 255, 0.4) !important;
}

.preset-button.load-button:hover {
    color: var(--eq-active) !important;
    border-color: var(--eq-active) !important;
    box-shadow: 0 4px 12px rgba(0, 255, 136, 0.4) !important;
}

.preset-button.delete-button:hover {
    color: #ff4444 !important;
    border-color: #ff4444 !important;
    box-shadow: 0 4px 12px rgba(255, 68, 68, 0.4) !important;
}

/* Main EQ Canvas */
.parametric-eq-canvas {
    width: 100%;
    max-width: 860px;
    height: 320px;
    margin: 0 auto 20px auto;
    background: var(--eq-background);
    border-radius: 16px;
    border: 2px solid var(--eq-border);
    box-shadow: 
        0 8px 25px rgba(0, 212, 255, 0.25),
        inset 0 2px 8px rgba(0, 0, 0, 0.6),
        inset 0 0 0 1px rgba(0, 212, 255, 0.1);
    cursor: crosshair;
    transition: all 0.3s ease;
    position: relative;
    z-index: 1;
}

.parametric-eq-canvas:hover {
    border-color: var(--eq-primary);
    box-shadow: 
        0 12px 30px rgba(0, 212, 255, 0.35),
        0 0 20px rgba(0, 255, 136, 0.25),
        inset 0 2px 8px rgba(0, 0, 0, 0.6);
    transform: translateY(-2px);
}

/* Enhanced Controls */
.parametric-eq-controls {
    margin-top: 8px;
}

.parametric-eq-types {
    display: grid !important;
    grid-template-columns: repeat(8, 1fr) !important;
    gap: 0px !important;
    margin-top: 0px !important;
    padding: 0px;
    background: #000000;
    border-radius: 0px !important;
    border: 0px;
    backdrop-filter: blur(5px) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}

.band-control {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 12px 8px !important;
    background: rgba(0, 0, 0, 0.3) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0, 212, 255, 0.2) !important;
    transition: all 0.3s ease !important;
}

.band-control:hover {
    background: rgba(0, 212, 255, 0.1) !important;
    border-color: var(--eq-primary) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3) !important;
}

.band-label {
    color: var(--eq-primary) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-shadow: 0 0 8px var(--eq-primary) !important;
}

.band-type-select {
    background: var(--eq-background) !important;
    border: 2px solid var(--eq-border) !important;
    border-radius: 8px !important;
    color: var(--eq-text) !important;
    padding: 8px 10px !important;
    font-size: 11px !important;
    font-family: 'Orbitron', monospace !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    text-align: center !important;
    font-weight: 500 !important;
}

.band-type-select:hover {
    border-color: var(--eq-primary) !important;
    box-shadow: 0 0 8px rgba(0, 212, 255, 0.3) !important;
    background: rgba(0, 212, 255, 0.05) !important;
}

.band-type-select:focus {
    border-color: var(--eq-primary) !important;
    box-shadow: 0 0 12px rgba(0, 212, 255, 0.5) !important;
    outline: none !important;
    background: rgba(0, 212, 255, 0.1) !important;
}

.band-type-select option {
    background: var(--eq-background) !important;
    color: var(--eq-text) !important;
    padding: 8px !important;
}

/* Status display */
.parametric-eq-status {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 16px;
    padding: 12px 16px;
    background: rgba(0, 0, 0, 0.4);
    border-radius: 12px;
    border: 1px solid rgba(0, 212, 255, 0.3);
    font-family: 'Orbitron', monospace;
    font-size: 11px;
    color: var(--eq-text-dim);
}

.eq-info-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
}

.eq-info-label {
    font-size: 9px;
    color: var(--eq-text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.eq-info-value {
    color: var(--eq-active);
    font-weight: 600;
    text-shadow: 0 0 6px var(--eq-active);
}

/* Loading and state indicators */
.eq-state-indicator {
    position: absolute;
    top: 16px;
    right: 16px;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--eq-active);
    box-shadow: 0 0 8px var(--eq-active);
    animation: pulseIndicator 2s ease-in-out infinite;
}

@keyframes pulseIndicator {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.2); }
}
`;
class ParametricEQUI {
    // ... (constructor and all other methods remain the same as the previous corrected version) ...
    // ... The only change is in app.registerExtension at the very end of the file ...
    constructor(node) {
        this.node = node;
        this.container = null;
        this.canvas = null;
        this.ctx = null;
        this.eqBands = [];
        this.selectedBand = null;
        this.hoveredBand = null;
        this.isDragging = false;
        this.retryCount = 0;
        this.maxRetries = 5;
        this.isInitialized = false;
        this.resizeObserver = null;
        this.animationFrameId = null;
        this.stateKey = `parametric_eq_state_${node.id}`;
        this.presets = this.loadPresets();
        this.presetSelect = null;
        
        // EQ parameters
        this.numBands = 8;
        this.sampleRate = 44100;
        this.frequencyRange = [20, 20000];
        this.gainRange = [-30, 30];
        this.qRange = [0.1, 100];
        
        // Bind methods
        this.handleMouseDown = this.handleMouseDown.bind(this);
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.handleMouseUp = this.handleMouseUp.bind(this);
        this.handleMouseLeave = this.handleMouseLeave.bind(this);
        this.handleWheel = this.handleWheel.bind(this);
        this.handleResize = this.handleResize.bind(this);
        this.saveState = this.saveState.bind(this);
        
        this.initializeWithRetry();
    }

    initializeWithRetry() {
        if (this.isInitialized) return;
        
        console.log(`[ParametricEQUI] Attempting to initialize UI for node ${this.node.id}, retry ${this.retryCount}`);
        
        if (document.readyState === 'loading' || !this.node.widgets) {
            if (this.retryCount < this.maxRetries) {
                this.retryCount++;
                setTimeout(() => this.initializeWithRetry(), Math.min(500 * this.retryCount, 2000));
                return;
            } else {
                console.error(`[ParametricEQUI] Failed to initialize after ${this.maxRetries} retries`);
                return;
            }
        }
        
        try {
            this.loadState(); // Load saved state first
            this.setWidgetDefaults();
            this.initializeUI();
            this.isInitialized = true;
            console.log(`[ParametricEQUI] UI initialized successfully for node ${this.node.id}`);
        } catch (err) {
            console.error(`[ParametricEQUI] Error initializing for node ${this.node.id}:`, err);
            if (this.retryCount < this.maxRetries) {
                this.retryCount++;
                setTimeout(() => this.initializeWithRetry(), 1000);
            }
        }
    }

    setWidgetDefaults() {
        if (!this.node.widgets) return;
        
        console.log(`[ParametricEQUI] Setting widget defaults for node ${this.node.id}`);
        
        const defaults = {
            'sample_rate': 44100,
            'output_gain': 0.0,
            'bypass': false
        };
        
        const baseFrequencies = [63, 125, 250, 500, 1000, 2000, 4000, 8000];
        for (let i = 1; i <= this.numBands; i++) {
            defaults[`band_${i}_enabled`] = true;
            defaults[`band_${i}_type`] = 'bell';
            defaults[`band_${i}_frequency`] = baseFrequencies[i - 1];
            defaults[`band_${i}_gain`] = 0.0;
            defaults[`band_${i}_q`] = 1.0;
        }
        
        Object.entries(defaults).forEach(([param, defaultValue]) => {
            const widget = this.node.widgets.find(w => w.name === param);
            if (widget && widget.value === undefined) {
                widget.value = defaultValue;
            }
        });
        
        this.node.setDirtyCanvas(true, true);
    }

    initializeUI() {
        this.injectStyles();
        this.hideOriginalWidgets();
        this.createCustomDOM();
        this.initializeEQBands();
        this.syncStateFromWidgets();
        this.setupResizeObserver();
        this.setNodeSizeOptimal();
        this.node.setDirtyCanvas(true, true);
    }

    injectStyles() {
        if (!document.getElementById('parametric-eq-styles')) {
            const styleSheet = document.createElement('style');
            styleSheet.id = 'parametric-eq-styles';
            styleSheet.textContent = CSS_STYLES_PARAMETRIC_EQ;
            document.head.appendChild(styleSheet);
        }

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
            if (widget.name && widget.name !== 'audio') {
                widget.computeSize = () => [0, -4];
                widget.type = "hidden_via_js";
                if (widget.element) {
                    widget.element.style.display = 'none';
                }
            }
        });
    }

    createCustomDOM() {
        console.log(`[ParametricEQUI] Creating custom DOM for node ${this.node.id}`);
        
        if (this.container) {
            this.container.remove();
        }

        this.container = document.createElement('div');
        this.container.className = 'parametric-eq-container';

        const title = document.createElement('div');
        title.className = 'parametric-eq-title';
        title.textContent = 'Parametric EQ';
        this.container.appendChild(title);

        const stateIndicator = document.createElement('div');
        stateIndicator.className = 'eq-state-indicator';
        this.container.appendChild(stateIndicator);
        
        this.createPresetControls();
        this.createEQCanvas();
        this.createMasterControls();
        this.createStatusDisplay();

        const widgetWrapper = document.createElement('div');
        widgetWrapper.className = 'parametric-eq-node-custom-widget';
        widgetWrapper.appendChild(this.container);
        
        const domWidget = this.node.addDOMWidget('parametric_eq_ui', 'div', widgetWrapper, {
            serialize: false
        });
        
        if (!domWidget) {
            console.error(`[ParametricEQUI] addDOMWidget FAILED for node ${this.node.id}`);
        }
    }

    createEQCanvas() {
        this.canvas = document.createElement('canvas');
        this.canvas.className = 'parametric-eq-canvas';
        this.canvas.width = 860;
        this.canvas.height = 320;
        this.ctx = this.canvas.getContext('2d');

        this.canvas.addEventListener('mousedown', this.handleMouseDown);
        this.canvas.addEventListener('mousemove', this.handleMouseMove);
        this.canvas.addEventListener('mouseup', this.handleMouseUp);
        this.canvas.addEventListener('mouseleave', this.handleMouseLeave);
        this.canvas.addEventListener('wheel', this.handleWheel, { passive: false });

        this.container.appendChild(this.canvas);
    }

    createMasterControls() {
        const controlsContainer = document.createElement('div');
        controlsContainer.className = 'parametric-eq-controls';

        const typeContainer = document.createElement('div');
        typeContainer.className = 'parametric-eq-types';

        for (let i = 1; i <= this.numBands; i++) {
            const bandTypeControl = this.createBandTypeControl(i);
            typeContainer.appendChild(bandTypeControl);
        }

        controlsContainer.appendChild(typeContainer);
        this.container.appendChild(controlsContainer);
    }

    createBandTypeControl(bandNum) {
        const bandControl = document.createElement('div');
        bandControl.className = 'band-control';

        const label = document.createElement('div');
        label.className = 'band-label';
        label.textContent = `Band ${bandNum}`;

        const typeSelect = document.createElement('select');
        typeSelect.className = 'band-type-select';
        typeSelect.dataset.bandNum = bandNum; 

        const filterTypes = ['bell', 'low_pass', 'high_pass', 'band_pass', 'low_shelf', 'high_shelf'];
        filterTypes.forEach(type => {
            const option = document.createElement('option');
            option.value = type;
            option.textContent = type.replace('_', ' ').toUpperCase();
            typeSelect.appendChild(option);
        });

        typeSelect.addEventListener('change', () => {
            const widget = this.node.widgets?.find(w => w.name === `band_${bandNum}_type`);
            if (widget) {
                widget.value = typeSelect.value;
                widget.callback?.(widget.value);
                this.eqBands[bandNum - 1].type = typeSelect.value;
                this.drawEQCanvas();
                this.updateStatusDisplay();
                this.saveState();
                if (this.presetSelect) this.presetSelect.value = '';
            }
        });

        bandControl.appendChild(label);
        bandControl.appendChild(typeSelect);

        return bandControl;
    }

    createStatusDisplay() {
        const statusContainer = document.createElement('div');
        statusContainer.className = 'parametric-eq-status';

        const activeBandsInfo = document.createElement('div');
        activeBandsInfo.className = 'eq-info-item';
        const activeBandsLabel = document.createElement('div');
        activeBandsLabel.className = 'eq-info-label';
        activeBandsLabel.textContent = 'Active Bands';
        const activeBandsValue = document.createElement('div');
        activeBandsValue.className = 'eq-info-value';
        activeBandsValue.id = `active-bands-count-${this.node.id}`;
        activeBandsValue.textContent = '8';
        activeBandsInfo.appendChild(activeBandsLabel);
        activeBandsInfo.appendChild(activeBandsValue);

        const selectedBandInfo = document.createElement('div');
        selectedBandInfo.className = 'eq-info-item';
        const selectedBandLabel = document.createElement('div');
        selectedBandLabel.className = 'eq-info-label';
        selectedBandLabel.textContent = 'Selected';
        const selectedBandValue = document.createElement('div');
        selectedBandValue.className = 'eq-info-value';
        selectedBandValue.id = `selected-band-info-${this.node.id}`;
        selectedBandValue.textContent = 'None';
        selectedBandInfo.appendChild(selectedBandLabel);
        selectedBandInfo.appendChild(selectedBandValue);

        const totalGainInfo = document.createElement('div');
        totalGainInfo.className = 'eq-info-item';
        const totalGainLabel = document.createElement('div');
        totalGainLabel.className = 'eq-info-label';
        totalGainLabel.textContent = 'Peak Gain';
        const totalGainValue = document.createElement('div');
        totalGainValue.className = 'eq-info-value';
        totalGainValue.id = `peak-gain-info-${this.node.id}`;
        totalGainValue.textContent = '0.0 dB';
        totalGainInfo.appendChild(totalGainLabel);
        totalGainInfo.appendChild(totalGainValue);

        statusContainer.appendChild(activeBandsInfo);
        statusContainer.appendChild(selectedBandInfo);
        statusContainer.appendChild(totalGainInfo);

        this.container.appendChild(statusContainer);
    }

    updateStatusDisplay() {
        if (!this.container) return;
        const activeBandsElement = this.container.querySelector(`#active-bands-count-${this.node.id}`);
        const selectedBandElement = this.container.querySelector(`#selected-band-info-${this.node.id}`);
        const peakGainElement = this.container.querySelector(`#peak-gain-info-${this.node.id}`);

        if (activeBandsElement) {
            const activeBands = this.eqBands.filter(band => band.enabled).length;
            activeBandsElement.textContent = activeBands.toString();
        }

        if (selectedBandElement) {
            const bandIndex = this.hoveredBand ?? this.selectedBand;
            if (bandIndex !== null) {
                const band = this.eqBands[bandIndex];
                const freqText = band.frequency >= 1000 ? `${(band.frequency / 1000).toFixed(1)}kHz` : `${band.frequency.toFixed(0)}Hz`;
                const qText = band.q.toFixed(2);
                selectedBandElement.textContent = `B${bandIndex + 1}: ${freqText}, ${band.gain.toFixed(1)}dB, Q:${qText}`;
            } else {
                selectedBandElement.textContent = 'None';
            }
        }

        if (peakGainElement) {
            let peakGain = 0;
            for (let f = 20; f <= 20000; f *= 1.05) {
                const gain = this.calculateTotalGain(f);
                if (Math.abs(gain) > Math.abs(peakGain)) {
                    peakGain = gain;
                }
            }
            peakGainElement.textContent = `${peakGain.toFixed(1)} dB`;
        }
    }

    initializeEQBands() {
        this.eqBands = [];
        const baseFrequencies = [63, 125, 250, 500, 1000, 2000, 4000, 8000];
        
        for (let i = 0; i < this.numBands; i++) {
            this.eqBands.push({
                enabled: true,
                type: 'bell',
                frequency: baseFrequencies[i],
                gain: 0.0,
                q: 1.0
            });
        }
        this.initializeBasicPresets();
        this.drawEQCanvas();
        this.updateStatusDisplay();
    }
    
    // --- PRESET METHODS ---

    initializeBasicPresets() {
        if (Object.keys(this.presets).length === 0) {
            this.presets = {
                "Flat": this.createFlatPreset(),
                "Vocal Clarity": this.createVocalClarityPreset(),
                "Bass Boost": this.createBassBoostPreset(),
                "Bright & Airy": this.createBrightAiryPreset(),
                "Warm & Full": this.createWarmFullPreset(),
                "De-Esser": this.createDeEsserPreset(),
                "Radio Voice": this.createRadioVoicePreset(),
                "Modern Pop": this.createModernPopPreset()
            };
            this.savePresets();
            console.log('[ParametricEQUI] Basic presets initialized');
        }
    }

    createFlatPreset() { return { eqBands: Array(8).fill(null).map((_, i) => ({ enabled: true, type: 'bell', frequency: [63, 125, 250, 500, 1000, 2000, 4000, 8000][i], gain: 0.0, q: 1.0 })) }; }
    createVocalClarityPreset() { return { eqBands: [ { enabled: true, type: 'high_pass', frequency: 80, gain: 0.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 200, gain: -2.0, q: 2.0 }, { enabled: true, type: 'bell', frequency: 500, gain: -1.0, q: 1.5 }, { enabled: true, type: 'bell', frequency: 1000, gain: 1.5, q: 1.0 }, { enabled: true, type: 'bell', frequency: 2500, gain: 2.5, q: 1.2 }, { enabled: true, type: 'bell', frequency: 5000, gain: 3.0, q: 1.5 }, { enabled: true, type: 'bell', frequency: 8000, gain: 1.0, q: 1.0 }, { enabled: true, type: 'low_pass', frequency: 12000, gain: 0.0, q: 1.0 } ] }; }
    createBassBoostPreset() { return { eqBands: [ { enabled: true, type: 'low_shelf', frequency: 150, gain: 4.0, q: 0.7 }, { enabled: true, type: 'bell', frequency: 250, gain: 1.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 500, gain: -0.5, q: 1.0 }, { enabled: true, type: 'bell', frequency: 1000, gain: 0.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 2000, gain: 0.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 4000, gain: 0.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 8000, gain: 0.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 12000, gain: 0.0, q: 1.0 } ] }; }
    createBrightAiryPreset() { return { eqBands: [ { enabled: true, type: 'bell', frequency: 250, gain: -1.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 500, gain: -0.5, q: 1.0 }, { enabled: true, type: 'bell', frequency: 2000, gain: 1.5, q: 1.0 }, { enabled: true, type: 'bell', frequency: 4000, gain: 3.0, q: 1.2 }, { enabled: true, type: 'high_shelf', frequency: 8000, gain: 4.0, q: 0.7 }, { enabled: true, type: 'bell', frequency: 8000, gain: 0.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 12000, gain: 0.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 16000, gain: 0.0, q: 1.0 } ] }; }
    createWarmFullPreset() { return { eqBands: [ { enabled: true, type: 'bell', frequency: 80, gain: 2.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 200, gain: 1.5, q: 1.0 }, { enabled: true, type: 'bell', frequency: 400, gain: 0.5, q: 1.0 }, { enabled: true, type: 'bell', frequency: 800, gain: 1.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 1500, gain: 0.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 3000, gain: -1.0, q: 1.5 }, { enabled: true, type: 'bell', frequency: 6000, gain: -2.0, q: 2.0 }, { enabled: true, type: 'bell', frequency: 10000, gain: -1.5, q: 1.0 } ] }; }
    createDeEsserPreset() { return { eqBands: [ { enabled: true, type: 'bell', frequency: 63, gain: 0.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 125, gain: 0.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 250, gain: 0.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 500, gain: 0.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 1000, gain: 0.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 3000, gain: -2.0, q: 3.0 }, { enabled: true, type: 'bell', frequency: 6000, gain: -4.0, q: 4.0 }, { enabled: true, type: 'bell', frequency: 8000, gain: -3.0, q: 3.0 } ] }; }
    createRadioVoicePreset() { return { eqBands: [ { enabled: true, type: 'high_pass', frequency: 100, gain: 0.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 200, gain: -3.0, q: 2.0 }, { enabled: true, type: 'bell', frequency: 400, gain: 2.0, q: 1.5 }, { enabled: true, type: 'bell', frequency: 1000, gain: 3.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 2500, gain: 4.0, q: 1.2 }, { enabled: true, type: 'bell', frequency: 4000, gain: 2.0, q: 1.5 }, { enabled: true, type: 'low_pass', frequency: 8000, gain: 0.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 10000, gain: 0.0, q: 1.0 } ] }; }
    createModernPopPreset() { return { eqBands: [ { enabled: true, type: 'bell', frequency: 60, gain: 2.5, q: 1.5 }, { enabled: true, type: 'bell', frequency: 150, gain: -1.0, q: 2.0 }, { enabled: true, type: 'bell', frequency: 500, gain: -0.5, q: 1.0 }, { enabled: true, type: 'bell', frequency: 1200, gain: 1.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 3000, gain: 2.0, q: 1.2 }, { enabled: true, type: 'bell', frequency: 5000, gain: 1.5, q: 1.5 }, { enabled: true, type: 'bell', frequency: 8000, gain: 3.0, q: 1.0 }, { enabled: true, type: 'bell', frequency: 12000, gain: 2.0, q: 1.0 } ] }; }

    createPresetControls() {
        const presetContainer = document.createElement('div');
        presetContainer.className = 'parametric-eq-presets';

        this.presetSelect = document.createElement('select');
        this.presetSelect.className = 'preset-select';
        this.updatePresetDropdown();
        this.presetSelect.addEventListener('change', () => this.loadPreset());
        presetContainer.appendChild(this.presetSelect);

        const presetInput = document.createElement('input');
        presetInput.className = 'preset-input';
        presetInput.type = 'text';
        presetInput.placeholder = 'Preset Name';
        presetContainer.appendChild(presetInput);

        const saveButton = document.createElement('button');
        saveButton.className = 'preset-button';
        saveButton.textContent = 'Save';
        saveButton.addEventListener('click', () => {
            const presetName = presetInput.value.trim();
            if (presetName) {
                this.savePreset(presetName);
                presetInput.value = '';
            } else {
                console.warn("Please enter a preset name");
            }
        });
        presetContainer.appendChild(saveButton);

        const deleteButton = document.createElement('button');
        deleteButton.className = 'preset-button delete-button';
        deleteButton.textContent = 'Delete';
        deleteButton.addEventListener('click', () => this.deletePreset());
        presetContainer.appendChild(deleteButton);

        this.container.appendChild(presetContainer);
    }

    updatePresetDropdown() {
        if (!this.presetSelect) return;
        const currentSelection = this.presetSelect.value;
        this.presetSelect.innerHTML = '';
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'Select Preset';
        this.presetSelect.appendChild(defaultOption);
        Object.keys(this.presets).sort().forEach(name => {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            this.presetSelect.appendChild(option);
        });
        this.presetSelect.value = currentSelection;
    }

    savePreset(name) {
        this.presets[name] = { eqBands: this.eqBands.map(band => ({ ...band })), timestamp: Date.now() };
        this.savePresets();
        this.updatePresetDropdown();
        this.presetSelect.value = name;
    }

    loadPreset() {
        const presetName = this.presetSelect.value;
        if (!presetName) return;
        const preset = this.presets[presetName];
        if (!preset || !preset.eqBands) return;

        // 1. Update the internal state array
        this.eqBands = preset.eqBands.map(bandData => ({
            enabled: bandData.enabled !== undefined ? bandData.enabled : true,
            type: bandData.type || 'bell',
            frequency: bandData.frequency || 1000,
            gain: bandData.gain || 0,
            q: bandData.q || 1
        }));

        // 2. Push the new state to the hidden ComfyUI widgets
        this.eqBands.forEach((band, index) => {
            this.updateWidgetFromBand(index, true, true);
        });

        // 3. Directly update the UI from the new internal state
        this.updateAllBandTypeSelects();
        this.drawEQCanvas();
        this.updateStatusDisplay();
        
        // 4. Save state
        this.saveState();
        console.log(`[ParametricEQUI] Preset "${presetName}" loaded.`);
    }

    deletePreset() {
        const name = this.presetSelect.value;
        if (!name) return;
        const basicPresets = Object.keys(this.createFlatPreset());
        if (basicPresets.includes(name)) return;
        delete this.presets[name];
        this.savePresets();
        this.updatePresetDropdown();
        this.presetSelect.value = '';
    }

    loadPresets() { try { return JSON.parse(localStorage.getItem('parametric_eq_presets')) || {}; } catch (e) { return {}; } }
    savePresets() { try { localStorage.setItem('parametric_eq_presets', JSON.stringify(this.presets)); } catch (e) { console.warn("Failed to save presets"); } }

    drawEQCanvas() {
        if (!this.ctx || !this.canvas) return;
        const { width, height } = this.canvas;
        this.ctx.clearRect(0, 0, width, height);
        this.drawGrid();
        this.drawFrequencyResponse();
        this.drawControlPoints();
        this.drawEnhancedLabels();
    }

    drawGrid() {
        const { width, height } = this.canvas;
        this.ctx.lineWidth = 0.5;
        const frequencies = [20, 30, 40, 50, 60, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000];
        frequencies.forEach(freq => {
            const x = this.frequencyToX(freq);
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, height);
            this.ctx.strokeStyle = [100, 1000, 10000].includes(freq) ? 'rgba(0, 212, 255, 0.3)' : 'rgba(0, 212, 255, 0.15)';
            this.ctx.lineWidth = [100, 1000, 10000].includes(freq) ? 1 : 0.5;
            this.ctx.stroke();
        });
        const gains = [-30, -24, -18, -12, -6, -3, 0, 3, 6, 12, 18, 24, 30];
        gains.forEach(gain => {
            const y = this.gainToY(gain);
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(width, y);
            if (gain === 0) { this.ctx.strokeStyle = 'rgba(0, 212, 255, 0.5)'; this.ctx.lineWidth = 2; }
            else if (gain % 6 === 0) { this.ctx.strokeStyle = 'rgba(0, 212, 255, 0.3)'; this.ctx.lineWidth = 1; }
            else { this.ctx.strokeStyle = 'rgba(0, 212, 255, 0.15)'; this.ctx.lineWidth = 0.5; }
            this.ctx.stroke();
        });
    }

    drawFrequencyResponse() {
        const { width, height } = this.canvas;
        const steps = 800;
        this.ctx.beginPath();
        this.ctx.moveTo(0, this.gainToY(0));
        for (let i = 0; i <= steps; i++) {
            const x = (i / steps) * width;
            const freq = this.xToFrequency(x);
            const totalGain = this.calculateTotalGain(freq);
            this.ctx.lineTo(x, this.gainToY(totalGain));
        }
        this.ctx.lineTo(width, this.gainToY(0));
        this.ctx.closePath();
        const gradient = this.ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, 'rgba(0, 255, 136, 0.2)');
        gradient.addColorStop(0.5, 'rgba(0, 255, 136, 0.1)');
        gradient.addColorStop(1, 'rgba(0, 255, 136, 0.05)');
        this.ctx.fillStyle = gradient;
        this.ctx.fill();
        this.ctx.beginPath();
        this.ctx.strokeStyle = '#00ff88';
        this.ctx.lineWidth = 4;
        this.ctx.shadowBlur = 20;
        this.ctx.shadowColor = 'rgba(0, 255, 136, 0.8)';
        for (let i = 0; i <= steps; i++) {
            const x = (i / steps) * width;
            const freq = this.xToFrequency(x);
            const totalGain = this.calculateTotalGain(freq);
            if (i === 0) this.ctx.moveTo(x, this.gainToY(totalGain));
            else this.ctx.lineTo(x, this.gainToY(totalGain));
        }
        this.ctx.stroke();
        this.ctx.shadowBlur = 0;
    }

    drawControlPoints() {
        this.eqBands.forEach((band, index) => {
            const x = this.frequencyToX(band.frequency);
            const y = this.gainToY(band.gain);
            this.ctx.beginPath();
            this.ctx.arc(x, y, 15, 0, 2 * Math.PI);
            this.ctx.fillStyle = 'rgba(0, 255, 136, 0.2)';
            this.ctx.fill();
            this.ctx.beginPath();
            this.ctx.arc(x, y, 10, 0, 2 * Math.PI);
            if (this.selectedBand === index) { this.ctx.fillStyle = '#ff6b00'; this.ctx.shadowBlur = 15; this.ctx.shadowColor = '#ff6b00'; }
            else if (this.hoveredBand === index) { this.ctx.fillStyle = '#40e0ff'; this.ctx.shadowBlur = 12; this.ctx.shadowColor = '#40e0ff'; }
            else { this.ctx.fillStyle = '#00ff88'; this.ctx.shadowBlur = 8; this.ctx.shadowColor = '#00ff88'; }
            this.ctx.fill();
            this.ctx.shadowBlur = 0;
            const borderGradient = this.ctx.createRadialGradient(x, y, 8, x, y, 12);
            borderGradient.addColorStop(0, 'rgba(255, 255, 255, 0.8)');
            borderGradient.addColorStop(1, 'rgba(255, 255, 255, 0.3)');
            this.ctx.strokeStyle = borderGradient;
            this.ctx.lineWidth = 3;
            this.ctx.stroke();
            this.ctx.fillStyle = '#000000';
            this.ctx.font = 'bold 11px Orbitron, monospace';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText((index + 1).toString(), x, y);
        });
    }

    drawEnhancedLabels() {
        const { width, height } = this.canvas;
        this.ctx.font = 'bold 10px Orbitron, monospace';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'top';
        this.ctx.shadowBlur = 3;
        this.ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
        const labelFrequencies = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];
        labelFrequencies.forEach(freq => {
            const x = this.frequencyToX(freq);
            let label = freq >= 1000 ? `${freq/1000}k` : freq.toString();
            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            this.ctx.fillRect(x - 15, 8, 30, 16);
            this.ctx.fillStyle = 'rgba(0, 212, 255, 0.9)';
            this.ctx.fillText(label, x, 12);
        });
        this.ctx.textAlign = 'left';
        this.ctx.textBaseline = 'middle';
        this.ctx.font = 'bold 9px Orbitron, monospace';
        const labelGains = [-24, -12, -6, 0, 6, 12, 24];
        labelGains.forEach(gain => {
            const y = this.gainToY(gain);
            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            this.ctx.fillRect(5, y - 8, 35, 16);
            if (gain > 0) this.ctx.fillStyle = 'rgba(255, 107, 0, 0.9)';
            else if (gain < 0) this.ctx.fillStyle = 'rgba(68, 68, 255, 0.9)';
            else this.ctx.fillStyle = 'rgba(0, 212, 255, 0.9)';
            this.ctx.fillText(`${gain > 0 ? '+' : ''}${gain}dB`, 8, y);
        });
        this.ctx.shadowBlur = 0;
    }

    frequencyToX(f) { const logMin=Math.log10(20), logMax=Math.log10(20000); return ((Math.log10(f)-logMin)/(logMax-logMin))*this.canvas.width; }
    xToFrequency(x) { const logMin=Math.log10(20), logMax=Math.log10(20000); return 10**(logMin+x/this.canvas.width*(logMax-logMin)); }
    gainToY(g) { return (1-(g-this.gainRange[0])/(this.gainRange[1]-this.gainRange[0]))*this.canvas.height; }
    yToGain(y) { return this.gainRange[0]+(1-y/this.canvas.height)*(this.gainRange[1]-this.gainRange[0]); }

    calculateTotalGain(frequency) {
        return this.eqBands.reduce((total, band) => {
            if(band.enabled) total += this.calculateBandGain(frequency, band);
            return total;
        }, 0);
    }
    
    calculateBandGain(frequency, band) {
        const { frequency: centerFreq, gain, q, type } = band;

        if (Math.abs(gain) < 0.001 && (type === 'bell' || type.includes('shelf'))) {
            return 0;
        }

        const octaves = Math.log2(frequency / centerFreq);

        switch(type) {
            case 'bell': {
                const bandwidth = 1/q;
                return gain * Math.exp(-Math.pow(octaves / bandwidth, 2) * 2);
            }
            case 'low_shelf': {
                return (1 + Math.tanh(q * -octaves)) / 2 * gain;
            }
            case 'high_shelf': {
                return (1 + Math.tanh(q * octaves)) / 2 * gain;
            }
            case 'low_pass': {
                if (octaves <= 0) return 0;
                return Math.max(-60, -6 * octaves * Math.sqrt(q));
            }
            case 'high_pass': {
                if (octaves >= 0) return 0;
                return Math.max(-60, -6 * Math.abs(octaves) * Math.sqrt(q));
            }
            case 'band_pass': {
                 const bandwidth = 1 / q;
                 const response = Math.exp(-Math.pow(octaves / bandwidth, 2) * 2);
                 return (response * 60) - 60; // Represents attenuation outside the band
            }
            default:
                return 0;
        }
    }

    handleMouseDown(e) {
        if (!this.canvas) return;
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = (e.clientX - rect.left) * (this.canvas.width / rect.width);
        const mouseY = (e.clientY - rect.top) * (this.canvas.height / rect.height);
        this.selectedBand = null;
        this.eqBands.forEach((band, index) => {
            const x = this.frequencyToX(band.frequency);
            const y = this.gainToY(band.gain);
            if (Math.hypot(mouseX - x, mouseY - y) < 25) {
                this.selectedBand = index;
                this.isDragging = true;
            }
        });
        if (this.selectedBand !== null) this.updateStatusDisplay();
    }

    handleMouseMove(e) {
        if (!this.canvas) return;
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = (e.clientX - rect.left) * (this.canvas.width / rect.width);
        const mouseY = (e.clientY - rect.top) * (this.canvas.height / rect.height);
        
        let newHoveredBand = null;
        this.eqBands.forEach((band, index) => {
            if (Math.hypot(mouseX - this.frequencyToX(band.frequency), mouseY - this.gainToY(band.gain)) < 25) {
                newHoveredBand = index;
            }
        });
        if (newHoveredBand !== this.hoveredBand) {
            this.hoveredBand = newHoveredBand;
            this.canvas.style.cursor = this.hoveredBand !== null ? 'pointer' : 'crosshair';
            this.updateStatusDisplay();
        }

        if (this.selectedBand !== null && this.isDragging) {
            const band = this.eqBands[this.selectedBand];
            band.frequency = Math.max(20, Math.min(20000, this.xToFrequency(mouseX)));
            band.gain = Math.max(-30, Math.min(30, this.yToGain(mouseY)));
            this.updateWidgetFromBand(this.selectedBand);
            this.updateStatusDisplay();
        }
        this.drawEQCanvas();
    }

    handleMouseUp() {
        if (this.selectedBand !== null) {
            this.isDragging = false;
            this.saveState();
            if (this.presetSelect) this.presetSelect.value = '';
        }
    }

    handleMouseLeave() {
        if (this.isDragging) this.saveState();
        this.hoveredBand = null;
        this.isDragging = false;
        this.drawEQCanvas();
        this.updateStatusDisplay();
    }

    handleWheel(e) {
        e.preventDefault();
        const bandIndex = this.hoveredBand ?? this.selectedBand;
        if (bandIndex === null) return;
        const band = this.eqBands[bandIndex];
        band.q = Math.max(0.1, Math.min(100, band.q - (e.deltaY > 0 ? 0.2 : -0.2)));
        this.updateWidgetFromBand(bandIndex);
        this.updateStatusDisplay();
        this.drawEQCanvas();
        this.saveState();
        if (this.presetSelect) this.presetSelect.value = '';
    }

    updateWidgetFromBand(bandIndex, updateType = false, updateEnabled = false) {
        const band = this.eqBands[bandIndex];
        const bandNum = bandIndex + 1;
        
        const freqWidget = this.node.widgets?.find(w => w.name === `band_${bandNum}_frequency`);
        const gainWidget = this.node.widgets?.find(w => w.name === `band_${bandNum}_gain`);
        const qWidget = this.node.widgets?.find(w => w.name === `band_${bandNum}_q`);
        const typeWidget = this.node.widgets?.find(w => w.name === `band_${bandNum}_type`);
        const enabledWidget = this.node.widgets?.find(w => w.name === `band_${bandNum}_enabled`);

        if (freqWidget) { freqWidget.value = band.frequency; freqWidget.callback?.(band.frequency); }
        if (gainWidget) { gainWidget.value = band.gain; gainWidget.callback?.(band.gain); }
        if (qWidget) { qWidget.value = band.q; qWidget.callback?.(band.q); }
        if (updateType && typeWidget) { typeWidget.value = band.type; typeWidget.callback?.(band.type); }
        if (updateEnabled && enabledWidget) { enabledWidget.value = band.enabled; enabledWidget.callback?.(band.enabled); }
    }

    saveState() { try { localStorage.setItem(this.stateKey, JSON.stringify(this.serialize())); } catch (e) {} }
    loadState() { try { const state = JSON.parse(localStorage.getItem(this.stateKey)); if(state) { this.deserialize(state); return true; } } catch(e) {} return false; }
    setNodeSizeOptimal() { this.node.setSize([920, 10]); }
    setupResizeObserver() { if (this.resizeObserver) this.resizeObserver.disconnect(); if (this.container) { this.resizeObserver = new ResizeObserver(() => this.handleResize()); this.resizeObserver.observe(this.container); } }
    handleResize() { requestAnimationFrame(() => { if(this.canvas) { this.drawEQCanvas(); this.updateStatusDisplay(); } }); }

    syncStateFromWidgets() {
        if (!this.node.widgets || !this.isInitialized) return;
        this.eqBands.forEach((band, index) => {
            const bandNum = index + 1;
            band.enabled = this.node.widgets.find(w => w.name === `band_${bandNum}_enabled`)?.value ?? band.enabled;
            band.type = this.node.widgets.find(w => w.name === `band_${bandNum}_type`)?.value ?? band.type;
            band.frequency = this.node.widgets.find(w => w.name === `band_${bandNum}_frequency`)?.value ?? band.frequency;
            band.gain = this.node.widgets.find(w => w.name === `band_${bandNum}_gain`)?.value ?? band.gain;
            band.q = this.node.widgets.find(w => w.name === `band_${bandNum}_q`)?.value ?? band.q;
        });
        this.updateAllBandTypeSelects();
        requestAnimationFrame(() => { this.drawEQCanvas(); this.updateStatusDisplay(); });
    }

    updateAllBandTypeSelects() {
        if (!this.container) return;
        this.eqBands.forEach((band, index) => {
            const bandNum = index + 1;
            const typeSelect = this.container.querySelector(`.band-type-select[data-band-num="${bandNum}"]`);
            if (typeSelect && typeSelect.value !== band.type) {
                typeSelect.value = band.type;
            }
        });
    }

    serialize() { return { eqBands: this.eqBands.map(band => ({ ...band })) }; }
    deserialize(data) { if (data?.eqBands) { this.eqBands = data.eqBands.map(b => ({...b})); this.syncStateFromWidgets(); } }
    destroy() { if (this.resizeObserver) this.resizeObserver.disconnect(); if (this.container) this.container.remove(); }
}

app.registerExtension({
    name: "Comfy.ParametricEQ.UI",
    async nodeCreated(node) {
        // --- FIX ---
        // Reverted to the original class name check, which is what the UI was
        // expecting all along. The Python code has been updated to match this.
        if (node.comfyClass === "ParametricEQNode") {
            node.bgcolor = "transparent";
            node.color = "transparent";
            node.title = "";
            if (node.parametricEQUIInstance) node.parametricEQUIInstance.destroy();
            setTimeout(() => {
                node.parametricEQUIInstance = new ParametricEQUI(node);
                setTimeout(() => node.parametricEQUIInstance?.setNodeSizeOptimal(), 100);
            }, 100);
            const onSerialize = node.onSerialize;
            node.onSerialize = function() { let data = onSerialize?.apply(this, arguments) || {}; if (this.parametricEQUIInstance) data.parametricEQUIData = this.parametricEQUIInstance.serialize(); return data; };
            const onDeserialize = node.onDeserialize;
            node.onDeserialize = function(data) { onDeserialize?.apply(this, arguments); setTimeout(() => this.parametricEQUIInstance?.deserialize(data?.parametricEQUIData), 200); };
            const onRemove = node.onRemove;
            node.onRemove = function() { this.parametricEQUIInstance?.destroy(); onRemove?.apply(this, arguments); };
            const onWidgetChanged = node.onWidgetChanged;
            node.onWidgetChanged = function(name, value, old) { 
                onWidgetChanged?.apply(this, arguments); 
                this.parametricEQUIInstance?.syncStateFromWidgets(); 
            };
        }
    }
});
import { app } from "../../scripts/app.js";

const MAX_SINGLE_BLOCKS = 38;
const MAX_DOUBLE_BLOCKS = 19;

console.log("FluxLoraBlocksPatcher_UI.js (Style V5 - Compact Layout): Script loading...");

const CSS_STYLES_FLUX_LORA_ENHANCED = `
@font-face {
    font-family: 'Orbitron';
    src: url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&display=swap') format('woff2'),
         local('Orbitron');
}

:root {
    --primary-accent: #7700ff;
    --primary-accent-light: #9a70ff;
    --active-green: #2ecc71;
    --reset-red: #d11d0a;
    --randomize-orange: #ff8c00;
    --text-white: #ffffff;
    --text-gray: #cccccc;
    --background-black: #000000;
    --translucent-light: rgba(255, 255, 255, 0.1);
    --translucent-lighter: rgba(255, 255, 255, 0.2);
    --focus-blue: #4a90e2;
}

.flux-lora-patcher-node-custom-widget { 
    box-sizing: content-box;
    position: relative;
    top: -15px;
    width: 100%;
    padding: 0; 
    margin: 0;
    overflow: hidden; 
}

.flux-lora-patcher-container {
    background: var(--background-black);
    border-radius: 12px;
    padding: 10px;
    margin: 0;
    border: 0px;
    width: 100%;
    box-sizing: border-box;
    min-height: 200px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    display: flex;
    flex-direction: column;
    align-items: stretch;
}

@keyframes breathePurpleTitle {
    0%, 100% { 
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8), 0 0 25px var(--primary-accent), 0 0 5px var(--primary-accent-light); 
        transform: scale(1); 
        opacity: 1; 
    }
    50% { 
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8), 0 0 35px var(--primary-accent-light), 0 0 10px var(--primary-accent); 
        transform: scale(0.97); 
        color: var(--primary-accent-light); 
    }
}

.flux-lora-patcher-title {
    color: var(--primary-accent);
    user-select: none;
    font-family: 'Orbitron', sans-serif;
    font-size: 15px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 10px;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8), 0 0 25px var(--primary-accent), 0 0 5px var(--primary-accent-light);
}

.flux-lora-controls-section {
    display: flex;
    flex-direction: column;
    gap: 5px;
    margin-bottom: 10px;
    align-items: center;
}

.flux-lora-preset-controls {
    display: flex;
    flex-wrap: nowrap;
    gap: 5px;
    justify-content: center;
    align-items: center;
}

.flux-lora-randomize-controls {
    display: flex;
    flex-wrap: nowrap;
    gap: 5px;
    justify-content: center;
    align-items: center;
    margin-top: 5px;
}

.flux-lora-randomize-label {
    color: var(--randomize-orange);
    font-size: 10px;
    font-weight: 500;
    font-family: 'Orbitron', sans-serif;
    text-shadow: 0 0 4px rgba(255, 140, 0, 0.6);
    margin-right: 5px;
    user-select: none;
}

.flux-lora-randomize-input {
    background: #000000;
    border: 0px;
    border-radius: 6px;
    color: var(--randomize-orange);
    padding: 2px 4px;
    font-size: 10px;
    width: 50px;
    height: 20px;
    text-align: center;
    font-family: 'Orbitron', sans-serif;
    font-weight: 500;
    transition: all 0.3s ease;
    text-shadow: 0 0 4px rgba(255, 140, 0, 0.6);
}

.flux-lora-randomize-input:focus {
    outline: none;
    border-color: #000000;
}

.flux-lora-value-controls {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    margin: 5px 0;
}

.flux-lora-preset-select, .flux-lora-preset-input {
    background: #000000;
    border: 0px;
    border-radius: 6px;
    color: var(--text-white);
    padding: 2px 6px;
    font-size: 10px;
    font-family: 'Orbitron', sans-serif;
    font-weight: 500;
    transition: all 0.3s ease;
    width: 80px;
}

.flux-lora-preset-input { width: 90px; }

.flux-lora-preset-select:focus, .flux-lora-preset-input:focus {
    outline: none;
    border-color: #000000;
    box-shadow: 0 0 0 2px rgba(154, 112, 255, 0.2);
}

.flux-lora-value-label {
    color: var(--active-green);
    font-size: 14px;
    font-weight: 500;
    font-family: 'Orbitron', sans-serif;
    text-shadow: 0 0 6px var(--active-green), 0 0 12px var(--active-green);
    margin-right: 10px;
    user-select: none;
}

.flux-lora-value-input {
    background: #000000;
    border: 0px;
    border-radius: 6px;
    color: var(--active-green);
    padding: 4px;
    font-size: 14px;
    width: 60px;
    text-align: center;
    font-family: 'Orbitron', sans-serif;
    font-weight: 500;
    transition: all 0.3s ease;
    text-shadow: 0 0 6px var(--active-green), 0 0 12 iet: none;
}

.flux-lora-value-input:focus {
    outline: none;
    border-color: 0 0 0 2px rgba(154, 112, 255, 0.2);
    box-shadow: 0 0 0 2px rgba(154, 112, 255, 0.2);
}

.flux-lora-preset-button, .flux-lora-action-button {
    background: linear-gradient(45deg, var(--background-black) 0%, var(--primary-accent) 100%);
    color: var(--text-white);
    border: 0px;
    padding: 2px 6px;
    border-radius: 15px;
    cursor: pointer;
    font-size: 10px;
    font-weight: 500;
    font-family: 'Orbitron', sans-serif;
    transition: all 0.3s ease, transform 0.2s ease-out;
    position: relative;
    overflow: hidden;
    user-select: none;
}

.flux-lora-preset-button.load-button,
.flux-lora-preset-button.delete-button {
    background: #000000;
    border: 0px;
}

.flux-lora-preset-button:hover, .flux-lora-action-button:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 18px rgba(119, 0, 255, 0.4);
    animation: button-glow 1.5s infinite;
}

@keyframes button-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(119, 0, 255, 0.4); }
    50% { box-shadow: 0 0 10px 5px rgba(119, 0, 255, 0.4); }
}

.flux-lora-action-button.flux-lora-reset-button {
    background: #000000;
    padding: 2px 8px;
    font-size: 9px;
}

.flux-lora-action-button.flux-lora-reset-button:hover {
    box-shadow: 0 6px 18px rgba(209, 29, 10, 0.6);
    animation: reset-glow 1.5s infinite;
}

@keyframes reset-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(209, 29, 10, 0.6); }
    50% { box-shadow: 0 0 10px 3px rgba(209, 29, 10, 0.6); }
}

.flux-lora-action-button.flux-lora-disableall-button {
    background: #000000;
    border: 0px;
    padding: 2px 8px;
    font-size: 9px;
}

.flux-lora-action-button.flux-lora-disableall-button:hover {
    box-shadow: 0 6px 18px rgba(255, 255, 255, 0.2);
}

.flux-lora-action-button.flux-lora-randomize-button {
    background: #000000;
    padding: 2px 8px;
    font-size: 9px;
}

.flux-lora-action-button.flux-lora-randomize-button:hover {
    box-shadow: 0 6px 18px rgba(255, 140, 0, 0.6);
    animation: randomize-glow 1.5s infinite;
}

@keyframes randomize-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255, 140, 0, 0.6); }
    50% { box-shadow: 0 0 10px 3px rgba(255, 140, 0, 0.6); }
}

.flux-lora-tabs {
    display: flex;
    flex-wrap: nowrap;
    gap: 5px;
    margin-bottom: 10px;
    padding: 5px 0;
    justify-content: space-between;
    border-bottom: 2px solid var(--translucent-light);
    align-items: center;
}

.flux-lora-tab {
    background: linear-gradient(45deg, var(--background-black) 0%, var(--background-black) 100%);
    color: var(--text-gray);
    border: 0px;
    padding: 4px 8px;
    border-radius: 20px;
    cursor: pointer;
    font-size: 10px;
    font-weight: 500;
    font-family: 'Orbitron', sans-serif;
    transition: all 0.3s ease, transform 0.2s ease-out;
    white-space: nowrap;
    position: relative;
    overflow: hidden;
    user-select: none;
}

.flux-lora-tab:hover {
    background: linear-gradient(45deg, var(--background-black) 0%, var(--primary-accent) 100%);
    color: var(--text-white);
    transform: translateY(-2px) scale(1.05);
    animation: tab-glow 1.5s infinite;
}

@keyframes tab-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(119, 0, 255, 0.4); }
    50% { box-shadow: 0 0 10px 5px rgba(119, 0, 255, 0.4); }
}

.flux-lora-tab.active {
    background: linear-gradient(45deg, var(--background-black) 0%, var(--primary-accent) 100%);
    color: var(--text-white);
    transform: translateY(-2px) scale(1.1);
    animation: tab-glow 1.5s infinite;
}

.flux-lora-sliders-grid {
    display: grid;
    gap: 10px 5px;
    padding: 5px;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 8px;
    border: 0px;
}
.flux-lora-sliders-grid#grid-double_blocks {
    grid-template-columns: repeat(19, minmax(15px, 1fr));
}
.flux-lora-sliders-grid#grid-single_blocks {
    grid-template-columns: repeat(19, minmax(15px, 1fr));
}

.flux-lora-slider-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 90px;
    padding: 3px;
}

.flux-lora-slider {
    -webkit-appearance: none;
    appearance: none;
    width: 70px;
    height: 4px;
    background: linear-gradient(90deg, rgba(119, 0, 255, 0.3), rgba(119, 0, 255, 0.1));
    border-radius: 2px;
    box-shadow: 0 0 8px var(--primary-accent);
    outline: none;
    transform-origin: center;
    transform: rotate(-90deg) translateX(-50%);
    position: relative;
    left: 0%;
    margin: 0;
    transition: box-shadow 0.2s ease-in-out, background 0.3s ease;
}

.flux-lora-slider:hover {
    background: linear-gradient(90deg, rgba(119, 0, 255, 0.5), rgba(119, 0, 255, 0.2));
    box-shadow: 0 0 12px var(--primary-accent-light);
}

.flux-lora-slider.active-slider {
    background: linear-gradient(90deg, var(--active-green), rgba(46, 204, 113, 0.5));
    box-shadow: 0 0 12px var(--active-green);
}

.flux-lora-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 12px;
    height: 12px;
    background: var(--primary-accent);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--primary-accent);
    cursor: grab;
    transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.3s ease;
}

.flux-lora-slider:active::-webkit-slider-thumb {
    cursor: grabbing;
}

.flux-lora-slider.active-slider::-webkit-slider-thumb {
    background: var(--active-green);
    box-shadow: 0 0 8px var(--active-green);
}

.flux-lora-slider::-webkit-slider-thumb:hover {
    transform: scale(1.2);
    box-shadow: 0 0 12px var(--primary-accent-light);
}

.flux-lora-slider.active-slider::-webkit-slider-thumb:hover {
    box-shadow: 0 0 12px var(--active-green);
}

.flux-lora-block-label {
    color: var(--primary-accent-light);
    font-size: 10px;
    font-weight: 500;
    text-align: center;
    font-family: 'Orbitron', sans-serif;
    margin-top: 8px;
    text-shadow: 0 0 4px rgba(119, 0, 255, 0.6);
    transition: color 0.2s ease, text-shadow 0.2s ease;
    user-select: none;
}

.flux-lora-block-label.active-label {
    color: var(--active-green);
    text-shadow: 0 0 4px rgba(46, 204, 113, 0.6);
}
`;

class FluxLoraBlocksPatcherUI {
    constructor(node) {
        this.node = node;
        this.container = null;
        this.presets = this.loadLocalPresets();
        this.sliderValues = { double_blocks: Array(MAX_DOUBLE_BLOCKS).fill(1.0), single_blocks: Array(MAX_SINGLE_BLOCKS).fill(1.0) };
        this.activeTab = "double_blocks";
        this.selectedSlider = null;
        this.valueInput = null;
        this.valueLabel = null;
        this.minRandomInput = null;
        this.maxRandomInput = null;
        setTimeout(() => this.initializeUI(), 150);
    }

    loadLocalPresets() {
        const presets = localStorage.getItem('fluxLoraPatcherPresets_v4_enhanced');
        return presets ? JSON.parse(presets) : {};
    }

    saveLocalPresets() {
        localStorage.setItem('fluxLoraPatcherPresets_v4_enhanced', JSON.stringify(this.presets));
    }

    updatePresetDropdown() {
        if (!this.presetSelect) return;
        this.presetSelect.innerHTML = '';
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'Preset';
        this.presetSelect.appendChild(defaultOption);
        Object.keys(this.presets).sort().forEach(presetName => { 
            const option = document.createElement('option');
            option.value = presetName;
            option.textContent = presetName;
            this.presetSelect.appendChild(option);
        });
    }
    
    initializeUI() {
        if (!this.node.widgets || this.node.widgets.length < Math.max(MAX_SINGLE_BLOCKS, MAX_DOUBLE_BLOCKS)) {
            setTimeout(() => this.initializeUI(), 250);
            return;
        }
        try {
            this.hideOriginalWidgets();
            this.createCustomDOM(); 
            this.syncInternalStateFromWidgets(); 
            this.updateAllUISliders();         
            this.switchTab(this.activeTab);
            setTimeout(() => { 
                this.updateNodeSize(); 
                this.node.setDirtyCanvas(true, true); 
            }, 200);
        } catch (error) {
            console.error(`[FluxLoraPatcherUI initializeUI V5] ERROR for node ${this.node.id}:`, error);
        }
    }

    hideOriginalWidgets() { 
        if (!this.node.widgets) return;
        this.node.widgets.forEach(widget => {
            if (widget.name && widget.name.startsWith("lora_block_")) { 
                widget.computeSize = () => [0, -4]; widget.type = "hidden_via_js"; 
            }
        });
    }

    createCustomDOM() { 
        if (!document.getElementById('flux-lora-patcher-styles-enhanced-v4')) { 
            const styleSheet = document.createElement('style');
            styleSheet.id = 'flux-lora-patcher-styles-enhanced-v4';
            styleSheet.textContent = CSS_STYLES_FLUX_LORA_ENHANCED;
            document.head.appendChild(styleSheet);
        }
        // Ensure Orbitron font is loaded
        if (!document.querySelector('link[href*="Orbitron"]')) {
            const fontLink = document.createElement("link");
            fontLink.href = "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&display=swap";
            fontLink.rel = "stylesheet";
            document.head.appendChild(fontLink);
        }
        this.container = document.createElement('div');
        this.container.className = 'flux-lora-patcher-container';
        const title = document.createElement('div');
        title.className = 'flux-lora-patcher-title';
        title.textContent = 'FLUX LoRA Blocks Patcher';
        this.container.appendChild(title);
        this.createControlsSection();
        this.createTabs();
        this.createSlidersSection();
        if (this.container.children.length > 0) {
            const widgetWrapper = document.createElement('div');
            widgetWrapper.className = 'flux-lora-patcher-node-custom-widget';
            widgetWrapper.appendChild(this.container);
            const domWidget = this.node.addDOMWidget('flux_lora_patcher_ui', 'div', widgetWrapper, 
                { serialize: false, draw: (ctx, node, widget_width, y, widget_height) => {} });
            if(!domWidget) console.error(`[FluxLoraPatcherUI createCustomDOM V5] addDOMWidget FAILED`);
        }
    }

    createControlsSection() { 
        const controlsSection = document.createElement('div');
        controlsSection.className = 'flux-lora-controls-section';
        
        const presetControls = document.createElement('div');
        presetControls.className = 'flux-lora-preset-controls';
        this.presetSelect = document.createElement('select');
        this.presetSelect.className = 'flux-lora-preset-select';
        this.updatePresetDropdown();
        presetControls.appendChild(this.presetSelect);
        const loadButton = document.createElement('button');
        loadButton.className = 'flux-lora-preset-button load-button';
        loadButton.textContent = 'Load';
        loadButton.title = "Load selected preset";
        loadButton.addEventListener('click', () => this.applySelectedPreset());
        presetControls.appendChild(loadButton);
        this.presetNameInput = document.createElement('input');
        this.presetNameInput.className = 'flux-lora-preset-input';
        this.presetNameInput.type = 'text';
        this.presetNameInput.placeholder = 'Preset Name';
        presetControls.appendChild(this.presetNameInput);
        const saveButton = document.createElement('button');
        saveButton.className = 'flux-lora-preset-button';
        saveButton.textContent = 'Save';
        saveButton.title = "Save current settings as new preset";
        saveButton.addEventListener('click', () => {
            const presetName = this.presetNameInput.value.trim();
            if (presetName) { this.saveCurrentStateAsPreset(presetName); this.presetNameInput.value = ''; } 
            else { alert("Please enter a name for the preset."); }
        });
        presetControls.appendChild(saveButton);
        const deleteButton = document.createElement('button');
        deleteButton.className = 'flux-lora-preset-button delete-button';
        deleteButton.textContent = 'Delete';
        deleteButton.title = "Delete selected preset";
        deleteButton.addEventListener('click', () => this.deleteSelectedPreset());
        presetControls.appendChild(deleteButton);
        controlsSection.appendChild(presetControls);

        const randomizeControls = document.createElement('div');
        randomizeControls.className = 'flux-lora-randomize-controls';
        
        const minLabel = document.createElement('span');
        minLabel.className = 'flux-lora-randomize-label';
        minLabel.textContent = 'Min:';
        randomizeControls.appendChild(minLabel);
        
        this.minRandomInput = document.createElement('input');
        this.minRandomInput.className = 'flux-lora-randomize-input';
        this.minRandomInput.type = 'number';
        this.minRandomInput.min = 0;
        this.minRandomInput.max = 2;
        this.minRandomInput.step = 0.01;
        this.minRandomInput.value = 0.5;
        this.minRandomInput.title = "Minimum value for randomization";
        randomizeControls.appendChild(this.minRandomInput);
        
        const maxLabel = document.createElement('span');
        maxLabel.className = 'flux-lora-randomize-label';
        maxLabel.textContent = 'Max:';
        randomizeControls.appendChild(maxLabel);
        
        this.maxRandomInput = document.createElement('input');
        this.maxRandomInput.className = 'flux-lora-randomize-input';
        this.maxRandomInput.type = 'number';
        this.maxRandomInput.min = 0;
        this.maxRandomInput.max = 2;
        this.maxRandomInput.step = 0.01;
        this.maxRandomInput.value = 1.5;
        this.maxRandomInput.title = "Maximum value for randomization";
        randomizeControls.appendChild(this.maxRandomInput);
        
        controlsSection.appendChild(randomizeControls);

        const valueControls = document.createElement('div');
        valueControls.className = 'flux-lora-value-controls';
        this.valueLabel = document.createElement('span');
        this.valueLabel.className = 'flux-lora-value-label'; 
        this.valueLabel.textContent = 'Selected Block: ';
        valueControls.appendChild(this.valueLabel);
        this.valueInput = document.createElement('input');
        this.valueInput.type = 'number'; 
        this.valueInput.className = 'flux-lora-value-input';
        this.valueInput.min = 0; 
        this.valueInput.max = 2; 
        this.valueInput.step = 0.01;
        this.valueInput.value = 1.0; 
        this.valueInput.disabled = true;
        this.valueInput.title = "Set weight for the selected block";
        this.valueInput.addEventListener('change', (e) => this.applyValueInputToSelectedSlider(e.target));
        valueControls.appendChild(this.valueInput);
        controlsSection.appendChild(valueControls);

        this.container.appendChild(controlsSection);
    }

    createTabs() {
        const tabsContainer = document.createElement('div');
        tabsContainer.className = 'flux-lora-tabs';
        const tabSection = document.createElement('div');
        tabSection.style.display = 'flex';
        tabSection.style.gap = '5px';
        ['double_blocks', 'single_blocks'].forEach(type => {
            const tab = document.createElement('button');
            tab.className = 'flux-lora-tab';
            tab.textContent = type.replace('_blocks', '').toUpperCase();
            tab.dataset.type = type;
            if (type === this.activeTab) tab.classList.add('active');
            tab.addEventListener('click', () => this.switchTab(type));
            tabSection.appendChild(tab);
        });
        tabsContainer.appendChild(tabSection);

        const actionButtons = document.createElement('div');
        actionButtons.style.display = 'flex';
        actionButtons.style.gap = '5px';
        const resetButton = document.createElement('button');
        resetButton.className = 'flux-lora-action-button flux-lora-reset-button';
        resetButton.textContent = 'Reset All';
        resetButton.title = "Set all LoRA block weights to 1.0";
        resetButton.addEventListener('click', () => this.resetAllSlidersToDefault());
        actionButtons.appendChild(resetButton);
        const disableAllButton = document.createElement('button');
        disableAllButton.className = 'flux-lora-action-button flux-lora-disableall-button';
        disableAllButton.textContent = 'Disable All';
        disableAllButton.title = "Set all LoRA block weights to 0.0";
        disableAllButton.addEventListener('click', () => this.setAllSlidersToZero());
        actionButtons.appendChild(disableAllButton);
        
        const randomizeDoubleButton = document.createElement('button');
        randomizeDoubleButton.className = 'flux-lora-action-button flux-lora-randomize-button';
        randomizeDoubleButton.textContent = 'Randomize Double';
        randomizeDoubleButton.title = "Randomize weights for DOUBLE blocks";
        randomizeDoubleButton.style.display = this.activeTab === 'double_blocks' ? 'inline-block' : 'none';
        randomizeDoubleButton.addEventListener('click', () => this.randomizeSliders('double_blocks'));
        actionButtons.appendChild(randomizeDoubleButton);
        
        const randomizeSingleButton = document.createElement('button');
        randomizeSingleButton.className = 'flux-lora-action-button flux-lora-randomize-button';
        randomizeSingleButton.textContent = 'Randomize Single';
        randomizeSingleButton.title = "Randomize weights for SINGLE blocks";
        randomizeSingleButton.style.display = this.activeTab === 'single_blocks' ? 'inline-block' : 'none';
        randomizeSingleButton.addEventListener('click', () => this.randomizeSliders('single_blocks'));
        actionButtons.appendChild(randomizeSingleButton);
        
        this.randomizeDoubleButton = randomizeDoubleButton;
        this.randomizeSingleButton = randomizeSingleButton;
        
        tabsContainer.appendChild(actionButtons);

        this.container.appendChild(tabsContainer);
    }

    createSlidersSection() {
        ['double_blocks', 'single_blocks'].forEach(type => {
            const grid = document.createElement('div');
            grid.className = 'flux-lora-sliders-grid';
            grid.id = `grid-${type}`;
            const maxBlocks = type === 'double_blocks' ? MAX_DOUBLE_BLOCKS : MAX_SINGLE_BLOCKS;
            for (let i = 0; i < maxBlocks; i++) {
                const sliderContainer = document.createElement('div');
                sliderContainer.className = 'flux-lora-slider-container';
                const slider = document.createElement('input');
                slider.type = 'range'; 
                slider.className = 'flux-lora-slider';
                slider.min = 0; 
                slider.max = 2; 
                slider.step = 0.01;
                slider.value = this.sliderValues[type][i] !== undefined ? this.sliderValues[type][i] : 1.0;
                slider.dataset.index = i;
                slider.addEventListener('input', (e) => {
                    const index = parseInt(e.target.dataset.index); 
                    const value = parseFloat(e.target.value);
                    this.sliderValues[type][index] = value; 
                    this.syncSliderToWidget(index, value, type);
                    this.updateSliderVisualState(e.target);
                    if (this.selectedSlider === e.target && this.valueInput) this.valueInput.value = value.toFixed(2);
                });
                slider.addEventListener('click', (e) => this.selectSliderForValueInput(e.target));
                const label = document.createElement('span');
                label.className = 'flux-lora-block-label'; 
                label.textContent = `${i}`; 
                label.dataset.index = i;
                sliderContainer.appendChild(slider); 
                sliderContainer.appendChild(label);
                grid.appendChild(sliderContainer);
            }
            this.container.appendChild(grid);
        });
    }

    randomizeSliders(type) {
        const maxBlocks = type === 'double_blocks' ? MAX_DOUBLE_BLOCKS : MAX_SINGLE_BLOCKS;
        
        let minValue = parseFloat(this.minRandomInput.value);
        let maxValue = parseFloat(this.maxRandomInput.value);
        
        // Validate and clamp min/max values
        if (isNaN(minValue) || minValue < 0) minValue = 0.5;
        if (isNaN(maxValue) || maxValue > 2) maxValue = 1.5;
        if (minValue > maxValue) {
            const temp = minValue;
            minValue = maxValue;
            maxValue = temp;
        }
        
        // Generate random values within the min/max range
        const values = Array(maxBlocks).fill(0).map(() => 
            parseFloat((minValue + Math.random() * (maxValue - minValue)).toFixed(3))
        );
        
        // Update slider values and sync with widgets
        this.sliderValues[type] = values;
        this.sliderValues[type].forEach((v, i) => this.syncSliderToWidget(i, v, type));
        this.updateAllUISliders();
        
        // Reset value input and selection
        if (this.valueInput) {
            this.valueInput.disabled = true;
            this.valueLabel.textContent = 'Selected Block: ';
            this.selectedSlider = null;
        }
        this.node.setDirtyCanvas(true, true);
    }

    switchTab(type) {
        this.activeTab = type;
        this.container.querySelectorAll('.flux-lora-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.type === type);
        });
        this.container.querySelectorAll('.flux-lora-sliders-grid').forEach(grid => {
            grid.style.display = grid.id === `grid-${type}` ? 'grid' : 'none';
        });
        this.randomizeDoubleButton.style.display = type === 'double_blocks' ? 'inline-block' : 'none';
        this.randomizeSingleButton.style.display = type === 'single_blocks' ? 'inline-block' : 'none';
        this.updateAllUISliders();
        this.selectSliderForValueInput(null);
        this.updateNodeSize();
        this.node.setDirtyCanvas(true, true);
    }

    setAllSlidersToZero() { 
        this.sliderValues[this.activeTab].fill(0.0); 
        this.sliderValues[this.activeTab].forEach((v, i) => this.syncSliderToWidget(i, v, this.activeTab));
        this.updateAllUISliders(); 
        if (this.valueInput && this.selectedSlider) this.valueInput.value = 0.0; 
        else if (this.valueInput) {
             this.valueInput.value = 0.0; 
             this.valueInput.disabled = true;
             if(this.valueLabel) this.valueLabel.textContent = 'Selected Block: ';
        }
        this.node.setDirtyCanvas(true, true);
    }

    selectSliderForValueInput(sliderElement) { 
        this.selectedSlider = sliderElement;
        const index = sliderElement ? sliderElement.dataset.index : null;
        const value = sliderElement ? parseFloat(sliderElement.value) : null;
        if (this.valueLabel) this.valueLabel.textContent = `Selected Block: ${index !== null ? index : ''}`;
        if (this.valueInput) { 
            this.valueInput.value = value !== null ? value.toFixed(2) : 1.0; 
            this.valueInput.disabled = !sliderElement;
        }
    }

    applyValueInputToSelectedSlider(inputElement) { 
        if (!this.selectedSlider) return; 
        let value = parseFloat(inputElement.value);
        if (isNaN(value)) value = 1.0; 
        value = Math.max(0, Math.min(2, value)); 
        const index = parseInt(this.selectedSlider.dataset.index);
        this.sliderValues[this.activeTab][index] = value; 
        this.selectedSlider.value = value; 
        inputElement.value = value.toFixed(2); 
        this.syncSliderToWidget(index, value, this.activeTab); 
        this.updateSliderVisualState(this.selectedSlider);
    }

    resetAllSlidersToDefault() {
        this.sliderValues[this.activeTab].fill(1.0); 
        this.sliderValues[this.activeTab].forEach((v, i) => this.syncSliderToWidget(i, v, this.activeTab));
        this.updateAllUISliders(); 
        if (this.valueInput) { 
            this.valueInput.value = 1.0; 
            this.valueInput.disabled = true; 
        }
        if (this.valueLabel) this.valueLabel.textContent = 'Selected Block: ';
        this.selectedSlider = null; 
        this.node.setDirtyCanvas(true, true);
    }

    updateSliderVisualState(sliderElement) { 
        const value = parseFloat(sliderElement.value); 
        const index = sliderElement.dataset.index;
        const label = this.container.querySelector(`#grid-${this.activeTab} .flux-lora-block-label[data-index="${index}"]`);
        if (Math.abs(value - 1.0) > 1e-3) { 
            sliderElement.classList.add('active-slider'); 
            if (label) label.classList.add('active-label');
        } else {
            sliderElement.classList.remove('active-slider'); 
            if (label) label.classList.remove('active-label');
        }
    }

    updateAllUISliders() { 
        if (!this.container) return;
        const sliders = this.container.querySelectorAll(`#grid-${this.activeTab} .flux-lora-slider`);
        sliders.forEach(slider => {
            const index = parseInt(slider.dataset.index);
            slider.value = this.sliderValues[this.activeTab][index] !== undefined ? this.sliderValues[this.activeTab][index] : 1.0;
            this.updateSliderVisualState(slider);
        });
    }

    syncSliderToWidget(index, value, type) { 
        const widgetName = type === 'double_blocks' ? `lora_block_${index}_double_weight` : `lora_block_${index}_weight`;
        const widget = this.node.widgets?.find(w => w.name === widgetName);
        if (widget) widget.value = parseFloat(value.toFixed(3)); 
    }

    syncInternalStateFromWidgets() { 
        if (!this.node.widgets) { 
            this.sliderValues = { double_blocks: Array(MAX_DOUBLE_BLOCKS).fill(1.0), single_blocks: Array(MAX_SINGLE_BLOCKS).fill(1.0) }; 
            return; 
        }
        for (let i = 0; i < MAX_DOUBLE_BLOCKS; i++) {
            const widget = this.node.widgets.find(w => w.name === `lora_block_${i}_double_weight`);
            const defVal = 1.0; 
            if (widget && widget.value !== undefined) {
                if (this.sliderValues.double_blocks[i] !== parseFloat(widget.value)) this.sliderValues.double_blocks[i] = parseFloat(widget.value);
            } else { 
                if (this.sliderValues.double_blocks[i] === undefined) this.sliderValues.double_blocks[i] = defVal; 
            }
        }
        for (let i = 0; i < MAX_SINGLE_BLOCKS; i++) {
            const widget = this.node.widgets.find(w => w.name === `lora_block_${i}_weight`);
            const defVal = 1.0; 
            if (widget && widget.value !== undefined) {
                if (this.sliderValues.single_blocks[i] !== parseFloat(widget.value)) this.sliderValues.single_blocks[i] = parseFloat(widget.value);
            } else { 
                if (this.sliderValues.single_blocks[i] === undefined) this.sliderValues.single_blocks[i] = defVal; 
            }
        }
    }

    saveCurrentStateAsPreset(presetName) { 
        this.presets[presetName] = { double_blocks: [...this.sliderValues.double_blocks], single_blocks: [...this.sliderValues.single_blocks] }; 
        this.saveLocalPresets(); 
        this.updatePresetDropdown();
    }

    applySelectedPreset() { 
        const presetName = this.presetSelect.value; 
        if (!presetName || !this.presets[presetName]) return;
        const presetValues = this.presets[presetName];
        this.sliderValues = {
            double_blocks: presetValues.double_blocks ? [...presetValues.double_blocks] : Array(MAX_DOUBLE_BLOCKS).fill(1.0),
            single_blocks: presetValues.single_blocks ? [...presetValues.single_blocks] : Array(MAX_SINGLE_BLOCKS).fill(1.0)
        }; 
        this.sliderValues.double_blocks.forEach((v, i) => this.syncSliderToWidget(i, v, 'double_blocks'));
        this.sliderValues.single_blocks.forEach((v, i) => this.syncSliderToWidget(i, v, 'single_blocks'));
        this.updateAllUISliders(); 
        this.node.setDirtyCanvas(true, true);
    }

    deleteSelectedPreset() { 
        const presetName = this.presetSelect.value; 
        if (!presetName) return;
        if (confirm(`Delete preset "${presetName}"?`)) {
            delete this.presets[presetName]; 
            this.saveLocalPresets(); 
            this.updatePresetDropdown();
        }
    }
    
    calculateApproximateHeight() { 
        const titleHeight = this.container?.querySelector('.flux-lora-patcher-title')?.offsetHeight || 40;
        const controlsHeight = this.container?.querySelector('.flux-lora-controls-section')?.offsetHeight || 60;
        const tabsHeight = this.container?.querySelector('.flux-lora-tabs')?.offsetHeight || 40;

        const sliderContainerCSSHeight = 90;
        const gridRowGapCSS = 10;
        const gridPaddingVertical = 2 * 5;

        const numRows = this.activeTab === 'double_blocks' ? 1 : 2;

        const slidersGridContentHeight = (numRows * sliderContainerCSSHeight) + (Math.max(10, numRows - 1) * gridRowGapCSS);
        const slidersGridTotalHeight = slidersGridContentHeight + gridPaddingVertical;

        const containerOwnPadding = 2 * 10;

        return Math.max(200, titleHeight + controlsHeight + tabsHeight + slidersGridTotalHeight + containerOwnPadding);
    }

    updateNodeSize() { 
        if (!this.container || !this.node) return;
        const newHeight = this.calculateApproximateHeight();
        const currentSize = this.node.size || [500, 200]; 
        if (Math.abs(newHeight - currentSize[1]) > 10 || newHeight > currentSize[1]) {
             this.node.setSize([Math.max(500, currentSize[0]), newHeight]);
             this.node.setDirtyCanvas(true, true);
        }
    }

    serialize() { 
        return { fluxLoraPatcherPresets_v4_enhanced: this.presets, activeTab: this.activeTab, sliderValues: this.sliderValues }; 
    }

    deserialize(data) { 
        if (data && data.fluxLoraPatcherPresets_v4_enhanced) { 
            this.presets = data.fluxLoraPatcherPresets_v4_enhanced;
            this.saveLocalPresets(); 
            if (this.presetSelect) { 
                this.updatePresetDropdown();
            }
        }
        if (data && data.activeTab) {
            this.activeTab = data.activeTab;
        }
        if (data && data.sliderValues) {
            this.sliderValues = {
                double_blocks: data.sliderValues.double_blocks || Array(MAX_DOUBLE_BLOCKS).fill(1.0),
                single_blocks: data.sliderValues.single_blocks || Array(MAX_SINGLE_BLOCKS).fill(1.0)
            };
        }
        if (this.container) { 
            this.switchTab(this.activeTab);
            this.syncInternalStateFromWidgets();
            this.updateAllUISliders();
            this.updateNodeSize();
        }
    }
}

app.registerExtension({
    name: "Comfy.FluxLoraBlocksPatcher.UI.EnhancedV4",
    async nodeCreated(node) {
        if (node.comfyClass === "FluxLoraBlocksPatcher") { 
            node.bgcolor = "#000000"; 
            node.color = "#000000"; 
            node.title_style = "color: #000000 ;"; 
            if (!node.fluxLoraPatcherUIInstance) { 
                node.fluxLoraPatcherUIInstance = new FluxLoraBlocksPatcherUI(node);
            }

            const originalOnSerialize = node.onSerialize;
            node.onSerialize = function() {
                let comfyData = originalOnSerialize ? originalOnSerialize.call(node) : {};
                if (node.fluxLoraPatcherUIInstance) {
                    const uiData = node.fluxLoraPatcherUIInstance.serialize();
                    comfyData.fluxLoraCustomUIData_v4_enhanced = uiData; 
                }
                return comfyData;
            };

            const originalOnDeserialize = node.onDeserialize;
            node.onDeserialize = function(data) {
                if (originalOnDeserialize) {
                    originalOnDeserialize.call(node, data); 
                }
                if (node.fluxLoraPatcherUIInstance && data && data.fluxLoraCustomUIData_v4_enhanced) {
                   node.fluxLoraPatcherUIInstance.deserialize(data.fluxLoraCustomUIData_v4_enhanced);
                }
            };
        }
    },
    async setup() { 
        console.log("[FluxLoraPatcher Ext EnhancedV4 Global Setup] Extension registered.");
    }
});

console.log("FluxLoraBlocksPatcher_UI.js (Enhanced Style V4 - Compact Layout): Script fully loaded.");
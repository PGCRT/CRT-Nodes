import { app } from "../../../scripts/app.js";

const WANCompareExtension = {
    name: "Comfy.WANCompareUI.CRT",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "WAN2.2 LoRA Compare Sampler") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);
                this.bgcolor = "transparent";
                this.onDrawBackground = function() {};
                this.title = "";
                this.color = "transparent";
                this.nodeData = nodeData;

                setTimeout(() => {
                    this.WANCompareUIInstance = new WANCompareUI(this);
                }, 10);
                
                this.computeSize = function() {
                    const MINIMUM_HEIGHT = 300; 
                    const MINIMUM_WIDTH = 1900;

                    if (this.WANCompareUIInstance?.container) {
                        try {
                            const content_height = this.WANCompareUIInstance.container.offsetHeight || 0;
                            const new_height = Math.max(MINIMUM_HEIGHT, content_height + 60);
                            return [MINIMUM_WIDTH, new_height];
                        } catch (e) {
                            console.warn('[WANCompareUI] Error computing size:', e);
                            return [MINIMUM_WIDTH, MINIMUM_HEIGHT];
                        }
                    }
                    
                    return [MINIMUM_WIDTH, MINIMUM_HEIGHT]; 
                };

                const originalSetSize = this.setSize;
                this.setSize = function(size) {
                    if (originalSetSize) {
                        originalSetSize.call(this, size);
                    }
                    this.size = size;
                };

                this.size = [1900, 300];
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                onConfigure?.apply(this, arguments);
                
                this.nodeData = nodeData;

                setTimeout(() => {
                    if (!this.WANCompareUIInstance) {
                        this.WANCompareUIInstance = new WANCompareUI(this);
                    }
                }, 100);
            };
        }
    }
};

class WANCompareUI {
    constructor(node) {
        this.node = node;
        this.loraGroups = [];
        this.loraList = [];
        this.activeDropdown = null;
        this.activePromptPopup = null;
        this.presets = new Map();
        this.draggedItem = null;
        this.initialize();
    }

    async initialize() {
        console.log('[WANCompareUI] Initializing UI...');
        try {
            this.injectStyles();
            await this.fetchLoRAList();
            this.createCustomDOM();
            this.loadPresets();
            this.renderPresets();
            this.parseConfigFromWidget();
            this.render();
            
            console.log('[WANCompareUI] UI initialized successfully');
            
            setTimeout(() => this.forceNodeResize(), 100);
            setTimeout(() => this.forceNodeResize(), 500);
            setTimeout(() => this.forceNodeResize(), 1000);
            
            // Global click listener to close popups
            document.addEventListener('mousedown', (e) => {
                if (this.activePromptPopup && 
                    !this.activePromptPopup.contains(e.target) && 
                    !e.target.classList.contains('wan-prompt-btn')) {
                    this.closePromptPopup();
                }
            });
        } catch (error) {
            console.error('[WANCompareUI] Failed to initialize:', error);
        }
    }

    async fetchLoRAList() {
        try {
            console.log('[WANCompareUI] Starting robust LoRA fetch...');
            let finalLoras = [];
            
            try {
                const response = await fetch('/loras');
                if (response.ok) {
                    const data = await response.json();
                    if (Array.isArray(data) && data.length > 0) {
                        finalLoras = data.filter(lora => 
                            lora && 
                            typeof lora === 'string' && 
                            lora.trim() && 
                            lora.toLowerCase().endsWith('.safetensors')
                        );
                    }
                }
            } catch (e) {
                console.warn('[WANCompareUI] /loras endpoint failed, trying deep search...', e.message);
            }

            if (finalLoras.length === 0) {
                try {
                    const response = await fetch('/object_info');
                    if (response.ok) {
                        const data = await response.json();
                        const loraSet = new Set();
                        const potentialLoraNodes = ['LoraLoader', 'LoRALoader', 'LoraLoader|pysssss', 'LoraLoaderSimple', 'Power Lora Loader (rgthree)'];

                        for (const nodeName in data) {
                            if (potentialLoraNodes.some(name => nodeName.includes(name))) {
                                const nodeInfo = data[nodeName];
                                if (nodeInfo.input?.required) {
                                    Object.values(nodeInfo.input.required).forEach(paramData => {
                                        if (Array.isArray(paramData) && Array.isArray(paramData[0])) {
                                            paramData[0].forEach(lora => {
                                                if (lora && 
                                                    typeof lora === 'string' && 
                                                    lora.trim() && 
                                                    lora.toLowerCase().endsWith('.safetensors')) {
                                                    loraSet.add(lora);
                                                }
                                            });
                                        }
                                    });
                                }
                            }
                        }
                        if (loraSet.size > 0) {
                            finalLoras = Array.from(loraSet);
                        }
                    }
                } catch (e) {
                    console.error('[WANCompareUI] Deep search failed:', e.message);
                }
            }
            
            if (finalLoras.length > 0) {
                this.loraList = finalLoras
                    .map(lora => lora.trim())
                    .sort((a, b) => a.split(/[\\/]/).pop().toLowerCase().localeCompare(b.split(/[\\/]/).pop().toLowerCase()));
            } else {
                this.loraList = ['No LoRAs found'];
                console.warn('[WANCompareUI] No LoRAs detected, using placeholder.');
            }
        } catch (error) {
            console.error('[WANCompareUI] Fatal error fetching LoRA list:', error);
            this.loraList = ['Error fetching LoRAs'];
        }
    }

    loadPresets() {
        const savedPresets = localStorage.getItem('wanComparePresets');
        if (savedPresets) {
            this.presets = new Map(JSON.parse(savedPresets));
        } else {
            this.presets = new Map();
        }
    }

    savePresets() {
        const presetData = JSON.stringify(Array.from(this.presets.entries()));
        localStorage.setItem('wanComparePresets', presetData);
    }

    savePreset(name, config) {
        if (!name) {
            console.warn('[WANCompareUI] Preset name is required.');
            return;
        }
        this.presets.set(name, JSON.parse(JSON.stringify(config)));
        this.savePresets();
        this.renderPresets();
    }

    loadPreset(name) {
        if (!name) {
            console.warn('[WANCompareUI] Preset name is required.');
            return;
        }
        const config = this.presets.get(name);
        if (config) {
            this.applyPreset(JSON.parse(JSON.stringify(config)));
        } else {
            console.warn('[WANCompareUI] Preset not found:', name);
        }
    }

    deletePreset(name) {
        if (!name) {
            console.warn('[WANCompareUI] Preset name is required.');
            return;
        }
        if (this.presets.delete(name)) {
            this.savePresets();
            this.renderPresets();
        } else {
            console.warn('[WANCompareUI] Preset not found:', name);
        }
    }

    applyPreset(config) {
        this.loraGroups = config.loraGroups || [];
        // Compatibility defaults
        this.loraGroups.forEach(group => {
            if (group.cfg_high === undefined) group.cfg_high = 1.0;
            if (group.cfg_low === undefined) group.cfg_low = 1.0;
            if (group.bypass_low === undefined) group.bypass_low = false;
            if (group.seed_offset === undefined) group.seed_offset = 0;
            if (group.prompt_override === undefined) group.prompt_override = "";
        });

        Object.entries(config.widgets || {}).forEach(([name, value]) => {
            if (name !== 'cfg_high_noise' && name !== 'cfg_low_noise') {
                this.setWidgetValue(name, value);
            }
        });
        this.render();
    }

    getCurrentConfig() {
        const widgets = {};
        this.node.widgets?.forEach(w => {
            if (w.name && w.name !== 'lora_batch_config' && w.name !== 'custom_labels' && w.name !== 'preset_data') {
                widgets[w.name] = w.value;
            }
        });
        return {
            loraGroups: this.loraGroups,
            widgets
        };
    }

    renderPresets() {
        this.presetSelect.innerHTML = '';
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'Select Preset';
        this.presetSelect.appendChild(defaultOption);
        for (const name of this.presets.keys()) {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            this.presetSelect.appendChild(option);
        }
    }

    forceNodeResize() {
        try {
            if (this.node) {
                const size = this.node.computeSize();
                this.node.size = size;
                if (this.node.setSize) {
                    this.node.setSize(size);
                }
                if (window.app && window.app.graph && window.app.graph.setDirtyCanvas) {
                    window.app.graph.setDirtyCanvas(true, true);
                }
            }
        } catch (e) {
            console.warn('[WANCompareUI] Error forcing resize:', e);
        }
    }

    injectStyles() {
        if (document.getElementById('wan-compare-styles-crt')) return;
        const style = document.createElement("style");
        style.id = "wan-compare-styles-crt";
        style.innerHTML = `
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
            :root { --wan-bg-main: #111113; --wan-bg-section: #1E1E22; --wan-bg-element: #2A2A2E; --wan-accent-purple: #7D26CD; --wan-accent-purple-light: #A158E2; --wan-accent-green: #2ECC71; --wan-accent-red: #E74C3C; --wan-accent-orange: #E67E22; --wan-text-light: #F0F0F0; --wan-text-med: #A0A0A0; --wan-text-dark: #666666; --wan-border-color: #333333; --wan-radius-lg: 16px; --wan-radius-md: 10px; --wan-radius-sm: 6px; }
            .wan-compare-container { background: var(--wan-bg-main); border: 2px solid var(--wan-accent-purple); border-radius: var(--wan-radius-lg); padding: 20px; margin-top: -10px; width: 1900px !important; font-family: 'Inter', sans-serif; color: var(--wan-text-light); box-shadow: 0 0 35px rgba(125, 38, 205, 0.6); position: relative; top: 0px; left: -10px; z-index: 1; user-select: none; box-sizing: border-box; }
            .wan-compare-header { text-align: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid var(--wan-border-color); }
            .wan-compare-title { font-family: 'Orbitron', monospace; font-size: 22px; font-weight: 700; color: var(--wan-accent-purple); text-shadow: 0 0 10px var(--wan-accent-purple-light); margin-bottom: 4px; }
            .wan-compare-subtitle { font-size: 13px; color: var(--wan-text-med); }
            .wan-compare-section { background: var(--wan-bg-section); border-radius: var(--wan-radius-md); padding: 15px 20px; margin-bottom: 15px; position: relative; }
            .wan-compare-section-title { font-family: 'Orbitron', monospace; font-size: 16px; font-weight: 700; color: var(--wan-accent-purple-light); margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }
            .wan-lora-groups-container { display: flex; flex-direction: column; gap: 15px; margin-bottom: 15px; }
            .wan-lora-stack-group { background: rgba(0,0,0,0.2); border: 1px solid var(--wan-accent-purple-light); border-radius: var(--wan-radius-md); padding: 10px; display: flex; flex-direction: column; gap: 8px; transition: opacity 0.3s ease; }
            .wan-lora-stack-group.disabled { opacity: 0.5; border-color: var(--wan-text-dark); }
            
            /* Group Settings Bar */
            .wan-group-settings-bar { display: flex; align-items: center; gap: 15px; padding: 5px 10px 10px 10px; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 5px; background: rgba(0,0,0,0.1); border-radius: var(--wan-radius-sm); }
            .wan-group-setting-item { display: flex; align-items: center; gap: 8px; font-size: 12px; font-family: 'Orbitron', monospace; color: var(--wan-text-med); }
            .wan-group-setting-input { width: 60px; height: 28px; background: var(--wan-bg-element); border: 1px solid var(--wan-border-color); color: var(--wan-text-light); border-radius: var(--wan-radius-sm); padding-left: 5px; }
            .wan-group-bypass-btn { font-size: 11px; padding: 4px 10px; height: 28px; }
            
            /* Prompt Button & Popup */
            .wan-prompt-btn { background: var(--wan-bg-element); border: 1px solid var(--wan-border-color); color: var(--wan-text-med); width: 30px; height: 30px; border-radius: 5px; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.2s; font-size: 16px; }
            .wan-prompt-btn:hover { background: var(--wan-accent-purple); color: white; border-color: var(--wan-accent-purple); }
            .wan-prompt-btn.active { background: rgba(230, 126, 34, 0.2); border-color: var(--wan-accent-orange); color: var(--wan-accent-orange); box-shadow: 0 0 8px rgba(230, 126, 34, 0.4); }
            .wan-prompt-popup { position: absolute; width: 400px; background: var(--wan-bg-section); border: 2px solid var(--wan-accent-purple); border-radius: var(--wan-radius-md); padding: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.8); z-index: 999999; display: flex; flex-direction: column; gap: 10px; }
            .wan-prompt-textarea { width: 100%; height: 150px; background: var(--wan-bg-main); border: 1px solid var(--wan-border-color); color: var(--wan-text-light); padding: 10px; border-radius: var(--wan-radius-sm); resize: vertical; font-family: 'Inter', sans-serif; font-size: 13px; box-sizing: border-box; }
            .wan-prompt-textarea:focus { outline: none; border-color: var(--wan-accent-purple); }
            .wan-popup-label { font-family: 'Orbitron', monospace; font-size: 12px; color: var(--wan-accent-purple-light); }
            .wan-popup-hint { font-size: 11px; color: var(--wan-text-dark); margin-top: -5px; }

            .wan-lora-header { display: grid; grid-template-columns: 30px 2.2fr 1fr 2.2fr 1fr auto; gap: 12px; padding: 0 10px 8px; font-family: 'Orbitron', monospace; font-size: 11px; color: var(--wan-text-med); text-transform: uppercase; text-align: center; }
            .wan-lora-row { display: grid; grid-template-columns: 30px 2.2fr 1fr 2.2fr 1fr auto; gap: 12px; align-items: center; }
            .wan-lora-row.disabled { opacity: 0.5; }
            .wan-row-actions { display: flex; align-items: center; justify-content: flex-end; gap: 8px; width: 120px; }
            .wan-group-footer { display: flex; align-items: center; gap: 12px; margin-top: 5px; padding: 0 10px; }
            .wan-group-actions { display: flex; align-items: center; gap: 8px; margin-left: auto; }
            .wan-btn { background: var(--wan-accent-purple); color: var(--wan-text-light); border: none; border-radius: var(--wan-radius-md); padding: 10px 20px; font-family: 'Orbitron', monospace; font-size: 13px; cursor: pointer; transition: all 0.2s ease; display: flex; align-items: center; justify-content: center; gap: 8px; width: 100%; }
            .wan-btn:hover { background: var(--wan-accent-purple-light); box-shadow: 0 0 15px var(--wan-accent-purple-light); }
            .wan-lora-action-btn { background: rgba(125, 38, 205, 0.7); color: white; border: none; border-radius: 50%; width: 24px; height: 24px; font-size: 16px; font-weight: bold; line-height: 24px; text-align: center; cursor: pointer; transition: all 0.2s ease; flex-shrink: 0; }
            .wan-lora-action-btn:hover { transform: scale(1.1); background: var(--wan-accent-purple-light); }
            .wan-duplicate-btn { background: var(--wan-accent-green); color: var(--wan-bg-main); }
            .wan-duplicate-btn:hover { background: var(--wan-accent-green); box-shadow: 0 0 15px var(--wan-accent-green); }
            .wan-delete-btn { background: var(--wan-accent-red); color: var(--wan-text-light); }
            .wan-delete-btn:hover { background: #ff6b5b; }
            .wan-on-off-button { border-radius: var(--wan-radius-md); padding: 8px 18px; font-family: 'Orbitron', monospace; font-size: 12px; font-weight: bold; cursor: pointer; transition: all 0.3s ease; text-align: center; }
            .wan-on-off-button.active { background: var(--wan-accent-green); color: var(--wan-bg-main); border: 2px solid var(--wan-accent-green); box-shadow: 0 0 10px var(--wan-accent-green); }
            .wan-on-off-button:not(.active) { background: var(--wan-text-dark); color: var(--wan-text-light); border: 2px solid var(--wan-text-dark); }
            .wan-row-toggle { background: var(--wan-bg-element); color: var(--wan-text-light); border: 1px solid var(--wan-border-color); border-radius: var(--wan-radius-sm); padding: 4px 8px; font-size: 11px; cursor: pointer; transition: all 0.2s; flex-shrink: 0; }
            .wan-row-toggle.active { background: var(--wan-accent-green); color: var(--wan-bg-main); border-color: var(--wan-accent-green); }
            .wan-input, .wan-select { background: var(--wan-bg-element); border: 1px solid var(--wan-border-color); border-radius: var(--wan-radius-sm); color: var(--wan-text-light); padding: 8px 12px; font-family: 'Inter', sans-serif; font-size: 13px; width: 100%; box-sizing: border-box; }
            .wan-select { height: 38px; }
            .wan-control-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
            .wan-control-group { display: flex; flex-direction: column; gap: 6px; }
            .wan-control-label { font-family: 'Orbitron', monospace; font-size: 11px; color: var(--wan-text-med); text-transform: uppercase; margin-left: 5px; }
            .wan-lora-select-container { position: relative; width: 100%; }
            .wan-lora-search { width: 100%; background: var(--wan-bg-element); border: 1px solid var(--wan-border-color); border-radius: var(--wan-radius-sm); color: var(--wan-text-light); padding: 8px 12px; font-family: 'Inter', sans-serif; font-size: 13px; }
            .wan-lora-search:focus { outline: none; border-color: var(--wan-accent-purple); box-shadow: 0 0 8px rgba(125, 38, 205, 0.7); }
            .wan-lora-search::placeholder { color: var(--wan-text-dark); font-style: italic; }
            .wan-lora-dropdown-portal { position: fixed !important; max-height: 200px; overflow-y: auto; background: var(--wan-bg-element); border: 1px solid var(--wan-border-color); border-radius: var(--wan-radius-sm); z-index: 999999 !important; display: none; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.8); min-width: 200px; }
            .wan-lora-option { padding: 8px 12px; cursor: pointer; color: var(--wan-text-light); font-size: 13px; border-bottom: 1px solid rgba(255, 255, 255, 0.1); }
            .wan-lora-option:last-child { border-bottom: none; }
            .wan-lora-option:hover { background: var(--wan-accent-purple); }
            .wan-lora-option.empty-option { color: var(--wan-text-dark); font-style: italic; }
            .wan-lora-dropdown-portal::-webkit-scrollbar { width: 6px; }
            .wan-lora-dropdown-portal::-webkit-scrollbar-track { background: transparent; }
            .wan-lora-dropdown-portal::-webkit-scrollbar-thumb { background: var(--wan-accent-purple); border-radius: 3px; }
            .wan-error-message { color: var(--wan-accent-red); font-family: 'Inter', sans-serif; font-size: 13px; text-align: center; margin-top: 10px; }
            .wan-label-input { font-size: calc(12px + 0.5vw); max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
            .wan-drag-handle { cursor: grab; text-align: center; color: var(--wan-text-med); font-size: 18px; line-height: 1; }
            .wan-drag-handle:active { cursor: grabbing; }
            .dragging { opacity: 0.4; }
            .wan-drop-target { border-top: 2px solid var(--wan-accent-purple-light) !important; }
        `;
        document.head.appendChild(style);
    }

    createCustomDOM() {
        if (this.container) return;

        console.log('[WANCompareUI] Creating custom DOM...');

        if (this.node.widgets) {
            this.node.widgets.forEach(w => { 
                if (w.name) {
                    w.computeSize = () => [0, -4]; 
                    w.type = "hidden";
                    w.hidden = true;
                }
            });
        }

        this.container = document.createElement('div');
        this.container.className = 'wan-compare-container';
        
        const header = document.createElement('div');
        header.className = 'wan-compare-header';
        header.innerHTML = `<div class="wan-compare-title">LoRA Compare Sampler</div><div class="wan-compare-subtitle">Advanced High/Low Noise LoRA Comparison Tool</div>`;
        this.container.appendChild(header);

        const presetSection = this.createSection("Presets", "💾");
        this.presetsContainer = document.createElement('div');
        this.presetsContainer.className = 'preset-controls';
        presetSection.appendChild(this.presetsContainer);
        this.container.appendChild(presetSection);

        this.presetSelect = document.createElement('select');
        this.presetSelect.className = 'wan-select';
        this.presetsContainer.appendChild(this.presetSelect);

        this.presetNameInput = document.createElement('input');
        this.presetNameInput.className = 'wan-input';
        this.presetNameInput.placeholder = 'Preset Name';
        this.presetsContainer.appendChild(this.presetNameInput);

        const saveBtn = document.createElement('button');
        saveBtn.className = 'wan-btn';
        saveBtn.textContent = 'Save Preset';
        saveBtn.onclick = () => {
            const name = this.presetNameInput.value.trim();
            if (name) {
                this.savePreset(name, this.getCurrentConfig());
                this.presetNameInput.value = '';
            }
        };
        this.presetsContainer.appendChild(saveBtn);

        const loadBtn = document.createElement('button');
        loadBtn.className = 'wan-btn';
        loadBtn.textContent = 'Load Preset';
        loadBtn.onclick = () => {
            if (this.presetSelect.value) this.loadPreset(this.presetSelect.value);
        };
        this.presetsContainer.appendChild(loadBtn);

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'wan-btn wan-delete-btn';
        deleteBtn.textContent = 'Delete Preset';
        deleteBtn.onclick = () => {
            if (this.presetSelect.value) this.deletePreset(this.presetSelect.value);
        };
        this.presetsContainer.appendChild(deleteBtn);

        this.renderPresets();

        const loraSection = this.createSection("LoRA Configurations", "🎨");
        this.groupsContainer = document.createElement('div');
        this.groupsContainer.className = 'wan-lora-groups-container';
        
        const addGroupBtn = document.createElement('button');
        addGroupBtn.className = 'wan-btn';
        addGroupBtn.innerHTML = '<span>➕</span> Add Comparison Group';
        addGroupBtn.onclick = () => {
            this.loraGroups.push({
                rows: [{ high_lora: '', high_strength: 1.0, low_lora: '', low_strength: 1.0, enabled: true }],
                label: '',
                enabled: true,
                cfg_high: 1.0,
                cfg_low: 1.0,
                bypass_low: false,
                seed_offset: 0,
                prompt_override: ""
            });
            this.render();
        };
        loraSection.append(this.groupsContainer, addGroupBtn);
        
        this.errorMessage = document.createElement('div');
        this.errorMessage.className = 'wan-error-message';
        this.errorMessage.style.display = 'none';
        loraSection.appendChild(this.errorMessage);

        const dimensionsSection = this.createSection("Dimensions & Generation", "📏");
        const samplerSection = this.createSection("Sampler Settings", "⚙️");
        const outputSection = this.createSection("Output Settings", "📤");

        const dimensionsGrid = document.createElement('div');
        dimensionsGrid.className = 'wan-control-grid';
        dimensionsGrid.style.gridTemplateColumns = 'repeat(3, 1fr)';
        dimensionsGrid.append(
            this.createNumberInput('width', 'Width', { min: 16, max: 4096, step: 16, default: 432 }),
            this.createNumberInput('height', 'Height', { min: 16, max: 4096, step: 16, default: 768 }),
            this.createNumberInput('frame_count', 'Frame Count', { min: 1, max: 4096, step: 1, default: 81 })
        );
        dimensionsSection.appendChild(dimensionsGrid);

        const samplerGrid = document.createElement('div');
        samplerGrid.className = 'wan-control-grid';

        const SAMPLERS = this.node.nodeData?.input?.required?.sampler_name?.[0] || ["euler"];
        const SCHEDULERS = this.node.nodeData?.input?.required?.scheduler?.[0] || ["normal"];
        
        samplerGrid.append(
            this.createNumberInput('boundary', 'Boundary', { min: 0, max: 1, step: 0.001, default: 0.875 }),
            this.createNumberInput('steps', 'Steps', { min: 1, max: 10000, step: 1, default: 8 }),
            this.createNumberInput('sigma_shift', 'Sigma Shift', { min: 0, max: 100, step: 0.01, default: 8.0 }),
            this.createSelect('sampler_name', 'Sampler', SAMPLERS, 'euler'),
            this.createSelect('scheduler', 'Scheduler', SCHEDULERS, 'simple')
        );
        samplerSection.appendChild(samplerGrid);

        const outputGrid = document.createElement('div');
        outputGrid.className = 'wan-control-grid';
        outputGrid.style.gridTemplateColumns = 'repeat(2, 1fr)';
        outputGrid.append(
            this.createToggle('enable_vae_decode', 'Enable VAE Decode', true),
            this.createToggle('create_comparison_grid', 'Create Comparison Grid', true),
            this.createToggle('add_labels', 'Add Custom Labels', true),
            this.createNumberInput('label_font_size', 'Label Font Size', { min: 8, max: 72, step: 1, default: 24 })
        );
        outputSection.appendChild(outputGrid);

        this.container.append(loraSection, dimensionsSection, samplerSection, outputSection);
        
        try {
            this.node.addDOMWidget('wan_compare_ui', 'div', this.container, { serialize: false });
        } catch (error) {
            console.error('[WANCompareUI] Error adding widgets:', error);
        }
        
        document.addEventListener('click', (e) => {
            if (this.activeDropdown && !e.target.closest('.wan-lora-select-container')) {
                this.closeActiveDropdown();
            }
        });
    }

    createSection(title, icon) {
        const section = document.createElement('div');
        section.className = 'wan-compare-section';
        const sectionTitle = document.createElement('div');
        sectionTitle.className = 'wan-compare-section-title';
        sectionTitle.innerHTML = `<span>${icon}</span>${title}`;
        section.appendChild(sectionTitle);
        return section;
    }

    createNumberInput(name, label, { min, max, step, default: defaultValue }) {
        const group = document.createElement('div');
        group.className = 'wan-control-group';
        const labelEl = document.createElement('label');
        labelEl.className = 'wan-control-label';
        labelEl.textContent = label;
        const input = document.createElement('input');
        input.type = 'number';
        input.className = 'wan-input';
        input.min = min;
        input.max = max;
        input.step = step;
        input.value = this.getWidgetValue(name) ?? defaultValue;
        this.setWidgetValue(name, input.value); 
        input.onchange = () => {
            this.setWidgetValue(name, parseFloat(input.value));
            this.updateNodeSize();
        };
        group.append(labelEl, input);
        return group;
    }

    createSelect(name, label, options, defaultValue) {
        const group = document.createElement('div');
        group.className = 'wan-control-group';
        const labelEl = document.createElement('label');
        labelEl.className = 'wan-control-label';
        labelEl.textContent = label;
        const select = document.createElement('select');
        select.className = 'wan-select';
        options.forEach(opt => {
            const option = document.createElement('option');
            option.value = opt;
            option.textContent = opt;
            select.appendChild(option);
        });
        select.value = this.getWidgetValue(name) ?? defaultValue;
        this.setWidgetValue(name, select.value);
        select.onchange = () => {
            this.setWidgetValue(name, select.value);
            this.updateNodeSize();
        };
        group.append(labelEl, select);
        return group;
    }

    createToggle(name, label, defaultValue) {
        const group = document.createElement('div');
        group.className = 'wan-control-group';
        const labelEl = document.createElement('label');
        labelEl.className = 'wan-control-label';
        labelEl.textContent = label;
        const toggle = document.createElement('button');
        const currentValue = this.getWidgetValue(name) ?? defaultValue;
        this.setWidgetValue(name, currentValue); 
        toggle.className = `wan-on-off-button ${currentValue ? 'active' : ''}`;
        toggle.textContent = currentValue ? 'ON' : 'OFF';
        toggle.onclick = () => {
            const newValue = !this.getWidgetValue(name);
            toggle.classList.toggle('active', newValue);
            toggle.textContent = newValue ? 'ON' : 'OFF';
            this.setWidgetValue(name, newValue);
            this.updateNodeSize();
        };
        group.append(labelEl, toggle);
        return group;
    }

    updateNodeSize() {
        setTimeout(() => {
            this.forceNodeResize();
        }, 10);
    }

    setWidgetValue(name, value) {
        if (name === 'preset_data') return;
        
        let widget = this.node.widgets.find(w => w.name === name);
        if (!widget) {
            widget = this.node.addWidget('number', name, value, () => {}, { serialize: true });
            widget.type = "hidden";
            widget.computeSize = () => [0, -4];
            widget.hidden = true;
        }
        if (widget) {
            widget.value = value;
            widget.callback?.(value);
        }
    }

    getWidgetValue(name) {
        const widget = this.node.widgets.find(w => w.name === name);
        return widget ? widget.value : undefined;
    }

    parseConfigFromWidget() {
        const configString = this.getWidgetValue('lora_batch_config') || '';
        const labelString = this.getWidgetValue('custom_labels') || '';
        this.loraGroups = [];
        const rows = configString.trim().split('\n').filter(line => line.trim());
        const labels = labelString.trim().split('\n');
        
        rows.forEach((line, idx) => {
            // Format: rows§enabled§cfg_h§cfg_l§bypass§seed_offset§encoded_prompt
            const parts = line.split('§');
            const rowStr = parts[0];
            const enabledStr = parts[1];
            
            // Backward compatibility
            const cfgHighStr = parts.length > 2 ? parts[2] : "1.0";
            const cfgLowStr = parts.length > 3 ? parts[3] : "1.0";
            const bypassStr = parts.length > 4 ? parts[4] : "false";
            const seedOffsetStr = parts.length > 5 ? parts[5] : "0";
            const encodedPrompt = parts.length > 6 ? parts[6] : "";

            if (rowStr && enabledStr !== undefined) {
                const group = {
                    rows: rowStr.split('|').map(r => {
                        const [high_lora, high_strength, low_lora, low_strength, enabled] = r.split(',');
                        return {
                            high_lora: high_lora === 'none' ? '' : high_lora,
                            high_strength: parseFloat(high_strength) || 1.0,
                            low_lora: low_lora === 'none' ? '' : low_lora,
                            low_strength: parseFloat(low_strength) || 1.0,
                            enabled: enabled === 'true'
                        };
                    }),
                    label: labels[idx] || '',
                    enabled: enabledStr === 'true',
                    cfg_high: parseFloat(cfgHighStr) || 1.0,
                    cfg_low: parseFloat(cfgLowStr) || 1.0,
                    bypass_low: bypassStr === 'true',
                    seed_offset: parseInt(seedOffsetStr) || 0,
                    prompt_override: decodeURIComponent(encodedPrompt)
                };
                this.loraGroups.push(group);
            }
        });
    }

    render() {
        this.groupsContainer.innerHTML = '';
        this.loraGroups.forEach((group, groupIdx) => {
            const groupEl = this.createGroupElement(group, groupIdx);
            this.groupsContainer.appendChild(groupEl);
        });
        this.syncConfigToWidget();
        this.updateNodeSize();
    }

    createGroupElement(group, groupIdx) {
        const groupEl = document.createElement('div');
        groupEl.className = `wan-lora-stack-group ${!group.enabled ? 'disabled' : ''}`;
        
        groupEl.draggable = true;
        groupEl.addEventListener('dragstart', (e) => {
            if (!e.target.classList.contains('wan-drag-handle')) {
                e.preventDefault();
                return;
            }
            this.draggedItem = { type: 'group', index: groupIdx };
            e.dataTransfer.effectAllowed = 'move';
            setTimeout(() => e.target.closest('.wan-lora-stack-group').classList.add('dragging'), 0);
        });
        groupEl.addEventListener('dragend', (e) => {
            e.target.closest('.wan-lora-stack-group').classList.remove('dragging');
        });
        groupEl.addEventListener('dragover', (e) => {
            e.preventDefault();
            if (this.draggedItem?.type === 'group') {
                groupEl.classList.add('wan-drop-target');
            }
        });
        groupEl.addEventListener('dragleave', () => {
            groupEl.classList.remove('wan-drop-target');
        });
        groupEl.addEventListener('drop', (e) => {
            e.preventDefault();
            groupEl.classList.remove('wan-drop-target');
            if (this.draggedItem?.type === 'group' && this.draggedItem.index !== groupIdx) {
                const [movedGroup] = this.loraGroups.splice(this.draggedItem.index, 1);
                this.loraGroups.splice(groupIdx, 0, movedGroup);
                this.render();
            }
            this.draggedItem = null;
        });

        // Group Settings Bar
        const settingsBar = document.createElement('div');
        settingsBar.className = 'wan-group-settings-bar';
        
        // CFG High
        const cfgHighItem = document.createElement('div');
        cfgHighItem.className = 'wan-group-setting-item';
        cfgHighItem.innerHTML = 'CFG H:';
        const cfgHighInput = document.createElement('input');
        cfgHighInput.type = 'number';
        cfgHighInput.className = 'wan-group-setting-input';
        cfgHighInput.step = 0.1;
        cfgHighInput.min = 0;
        cfgHighInput.value = group.cfg_high;
        cfgHighInput.onchange = () => {
            group.cfg_high = parseFloat(cfgHighInput.value) || 1.0;
            this.syncConfigToWidget();
        };
        cfgHighItem.appendChild(cfgHighInput);

        // CFG Low
        const cfgLowItem = document.createElement('div');
        cfgLowItem.className = 'wan-group-setting-item';
        cfgLowItem.innerHTML = 'CFG L:';
        const cfgLowInput = document.createElement('input');
        cfgLowInput.type = 'number';
        cfgLowInput.className = 'wan-group-setting-input';
        cfgLowInput.step = 0.1;
        cfgLowInput.min = 0;
        cfgLowInput.value = group.cfg_low;
        cfgLowInput.onchange = () => {
            group.cfg_low = parseFloat(cfgLowInput.value) || 1.0;
            this.syncConfigToWidget();
        };
        cfgLowItem.appendChild(cfgLowInput);

        // Seed Offset
        const seedItem = document.createElement('div');
        seedItem.className = 'wan-group-setting-item';
        seedItem.innerHTML = 'SEED OFF:';
        const seedInput = document.createElement('input');
        seedInput.type = 'number';
        seedInput.className = 'wan-group-setting-input';
        seedInput.step = 1;
        seedInput.value = group.seed_offset;
        seedInput.onchange = () => {
            group.seed_offset = parseInt(seedInput.value) || 0;
            this.syncConfigToWidget();
        };
        seedItem.appendChild(seedInput);

        // Prompt Override Button
        const promptItem = document.createElement('div');
        promptItem.className = 'wan-group-setting-item';
        const promptBtn = document.createElement('div');
        promptBtn.className = `wan-prompt-btn ${group.prompt_override ? 'active' : ''}`;
        promptBtn.innerHTML = '📝';
        promptBtn.title = "Override Positive Prompt";
        promptBtn.onclick = (e) => {
            e.stopPropagation();
            this.openPromptPopup(promptBtn, group);
        };
        promptItem.appendChild(promptBtn);

        // Bypass Toggle
        const bypassItem = document.createElement('div');
        bypassItem.className = 'wan-group-setting-item';
        bypassItem.style.marginLeft = 'auto'; 
        const bypassBtn = document.createElement('button');
        bypassBtn.className = `wan-on-off-button wan-group-bypass-btn ${group.bypass_low ? 'active' : ''}`;
        bypassBtn.textContent = group.bypass_low ? 'BYPASS LOW: ON' : 'BYPASS LOW: OFF';
        bypassBtn.onclick = () => {
            group.bypass_low = !group.bypass_low;
            bypassBtn.className = `wan-on-off-button wan-group-bypass-btn ${group.bypass_low ? 'active' : ''}`;
            bypassBtn.textContent = group.bypass_low ? 'BYPASS LOW: ON' : 'BYPASS LOW: OFF';
            this.syncConfigToWidget();
        };
        bypassItem.appendChild(bypassBtn);

        settingsBar.append(cfgHighItem, cfgLowItem, seedItem, promptItem, bypassItem);
        groupEl.appendChild(settingsBar);

        const header = document.createElement('div');
        header.className = 'wan-lora-header';
        header.innerHTML = '<span>⠿</span><span>High LoRA</span><span>Strength</span><span>Low LoRA</span><span>Strength</span><span>Actions</span>';
        groupEl.appendChild(header);

        group.rows.forEach((row, rowIdx) => {
            const rowEl = document.createElement('div');
            rowEl.className = `wan-lora-row ${!row.enabled ? 'disabled' : ''}`;
            rowEl.innerHTML = `
                <div class="wan-drag-handle">⠿</div>
                <div class="wan-lora-select-container"><input type="text" class="wan-lora-search" placeholder="Select LoRA" value="${row.high_lora || ''}"></div>
                <input type="number" class="wan-input" value="${row.high_strength}" min="-2" max="2" step="0.01">
                <div class="wan-lora-select-container"><input type="text" class="wan-lora-search" placeholder="Select LoRA" value="${row.low_lora || ''}"></div>
                <input type="number" class="wan-input" value="${row.low_strength}" min="-2" max="2" step="0.01">
                <div class="wan-row-actions">
                    <button class="wan-lora-action-btn wan-row-toggle ${row.enabled ? 'active' : ''}">✓</button>
                    <button class="wan-lora-action-btn">+</button>
                    <button class="wan-lora-action-btn wan-delete-btn">✖</button>
                </div>
            `;

            const dragHandle = rowEl.querySelector('.wan-drag-handle');
            dragHandle.draggable = true;
            dragHandle.addEventListener('dragstart', (e) => {
                e.stopPropagation();
                this.draggedItem = { type: 'row', groupIdx, rowIdx };
                e.dataTransfer.effectAllowed = 'move';
                setTimeout(() => rowEl.classList.add('dragging'), 0);
            });
            dragHandle.addEventListener('dragend', () => {
                rowEl.classList.remove('dragging');
            });
            rowEl.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (this.draggedItem?.type === 'row' && this.draggedItem.groupIdx === groupIdx) {
                    rowEl.classList.add('wan-drop-target');
                }
            });
            rowEl.addEventListener('dragleave', () => {
                rowEl.classList.remove('wan-drop-target');
            });
            rowEl.addEventListener('drop', (e) => {
                e.preventDefault();
                e.stopPropagation();
                rowEl.classList.remove('wan-drop-target');
                if (this.draggedItem?.type === 'row' && this.draggedItem.groupIdx === groupIdx && this.draggedItem.rowIdx !== rowIdx) {
                    const [movedRow] = group.rows.splice(this.draggedItem.rowIdx, 1);
                    group.rows.splice(rowIdx, 0, movedRow);
                    this.render();
                }
                this.draggedItem = null;
            });
            
            const [_, highInputContainer, highStrengthInput, lowInputContainer, lowStrengthInput, actions] = rowEl.children;
            const [toggleBtn, addBtn, deleteBtn] = actions.children;
            
            const highSearchInput = highInputContainer.querySelector('.wan-lora-search');
            const lowSearchInput = lowInputContainer.querySelector('.wan-lora-search');

            highSearchInput.addEventListener('focus', (e) => this.openDropdown(highInputContainer, row, 'high_lora'));
            highSearchInput.addEventListener('input', (e) => this.filterDropdown(e.target));
            highSearchInput.addEventListener('change', (e) => {
                row.high_lora = e.target.value;
                this.syncConfigToWidget();
            });

            highStrengthInput.addEventListener('change', (e) => {
                row.high_strength = parseFloat(e.target.value) || 1.0;
                this.syncConfigToWidget();
            });

            lowSearchInput.addEventListener('focus', (e) => this.openDropdown(lowInputContainer, row, 'low_lora'));
            lowSearchInput.addEventListener('input', (e) => this.filterDropdown(e.target));
            lowSearchInput.addEventListener('change', (e) => {
                row.low_lora = e.target.value;
                this.syncConfigToWidget();
            });

            lowStrengthInput.addEventListener('change', (e) => {
                row.low_strength = parseFloat(e.target.value) || 1.0;
                this.syncConfigToWidget();
            });
            
            toggleBtn.addEventListener('click', () => {
                row.enabled = !row.enabled;
                this.render();
            });
            
            addBtn.addEventListener('click', () => {
                group.rows.splice(rowIdx + 1, 0, { high_lora: '', high_strength: 1.0, low_lora: '', low_strength: 1.0, enabled: true });
                this.render();
            });

            deleteBtn.addEventListener('click', () => {
                if (group.rows.length > 1) {
                    group.rows.splice(rowIdx, 1);
                } else {
                    this.loraGroups.splice(groupIdx, 1);
                }
                this.render();
            });

            groupEl.appendChild(rowEl);
        });

        const footer = document.createElement('div');
        footer.className = 'wan-group-footer';
        
        const groupDragHandle = document.createElement('div');
        groupDragHandle.className = 'wan-drag-handle';
        groupDragHandle.innerHTML = '⠿';
        
        const labelInput = document.createElement('input');
        labelInput.type = 'text';
        labelInput.className = 'wan-input wan-label-input';
        labelInput.placeholder = 'Custom Label';
        labelInput.value = group.label;
        labelInput.onchange = () => {
            group.label = labelInput.value;
            this.syncConfigToWidget();
        };
        const toggleGroupBtn = document.createElement('button');
        toggleGroupBtn.className = `wan-on-off-button ${group.enabled ? 'active' : ''}`;
        toggleGroupBtn.textContent = group.enabled ? 'ON' : 'OFF';
        toggleGroupBtn.onclick = () => {
            group.enabled = !group.enabled;
            this.render();
        };
        const actions = document.createElement('div');
        actions.className = 'wan-group-actions';
        const duplicateGroupBtn = document.createElement('button');
        duplicateGroupBtn.className = 'wan-btn wan-duplicate-btn';
        duplicateGroupBtn.textContent = 'Duplicate Group';
        duplicateGroupBtn.onclick = () => {
            const duplicatedGroup = JSON.parse(JSON.stringify(group));
            duplicatedGroup.label = '';
            this.loraGroups.splice(groupIdx + 1, 0, duplicatedGroup);
            this.render();
        };
        const deleteGroupBtn = document.createElement('button');
        deleteGroupBtn.className = 'wan-btn wan-delete-btn';
        deleteGroupBtn.textContent = 'Delete Group';
        deleteGroupBtn.onclick = () => {
            this.loraGroups.splice(groupIdx, 1);
            this.render();
        };
        actions.appendChild(duplicateGroupBtn);
        actions.appendChild(deleteGroupBtn);
        footer.append(groupDragHandle, labelInput, toggleGroupBtn, actions);
        groupEl.appendChild(footer);

        return groupEl;
    }

    openPromptPopup(button, group) {
        this.closePromptPopup();
        
        const popup = document.createElement('div');
        popup.className = 'wan-prompt-popup';
        
        const label = document.createElement('div');
        label.className = 'wan-popup-label';
        label.textContent = "Positive Prompt Override";
        
        const hint = document.createElement('div');
        hint.className = 'wan-popup-hint';
        hint.textContent = "Leave empty to use global positive prompt. Requires CLIP input.";

        const textarea = document.createElement('textarea');
        textarea.className = 'wan-prompt-textarea';
        textarea.placeholder = "Enter custom positive prompt for this group...";
        textarea.value = group.prompt_override || "";
        
        // Auto-save on input
        textarea.addEventListener('input', () => {
            group.prompt_override = textarea.value;
            if (textarea.value.trim().length > 0) {
                button.classList.add('active');
            } else {
                button.classList.remove('active');
            }
            this.syncConfigToWidget();
        });

        popup.append(label, hint, textarea);
        document.body.appendChild(popup);
        
        const rect = button.getBoundingClientRect();
        popup.style.left = `${rect.left}px`;
        popup.style.top = `${rect.bottom + 5 + window.scrollY}px`;
        
        // Adjust if off-screen
        const popupRect = popup.getBoundingClientRect();
        if (popupRect.right > window.innerWidth) {
            popup.style.left = `${window.innerWidth - popupRect.width - 20}px`;
        }

        this.activePromptPopup = popup;
        textarea.focus();
    }

    closePromptPopup() {
        if (this.activePromptPopup) {
            this.activePromptPopup.remove();
            this.activePromptPopup = null;
        }
    }

    openDropdown(container, row, field) {
        if (this.activeDropdown) this.closeActiveDropdown();
        this.activeDropdown = document.createElement('div');
        this.activeDropdown.className = 'wan-lora-dropdown-portal';
        
        const addOption = (loraName, displayName) => {
            const option = document.createElement('div');
            option.className = 'wan-lora-option';
            option.textContent = displayName;
            option.dataset.value = loraName;
            option.onclick = () => {
                row[field] = loraName;
                container.querySelector('.wan-lora-search').value = loraName;
                this.closeActiveDropdown();
                this.syncConfigToWidget();
            };
            this.activeDropdown.appendChild(option);
        };
        
        this.loraList.forEach(lora => {
            addOption(lora, lora);
        });
        
        const emptyOption = document.createElement('div');
        emptyOption.className = 'wan-lora-option empty-option';
        emptyOption.textContent = 'None';
        emptyOption.onclick = () => {
            row[field] = '';
            container.querySelector('.wan-lora-search').value = '';
            this.closeActiveDropdown();
            this.syncConfigToWidget();
        };
        this.activeDropdown.prepend(emptyOption);
        
        document.body.appendChild(this.activeDropdown);
        const rect = container.getBoundingClientRect();
        this.activeDropdown.style.left = `${rect.left}px`;
        this.activeDropdown.style.top = `${rect.bottom + window.scrollY}px`;
        this.activeDropdown.style.width = `${rect.width}px`;
        this.activeDropdown.style.display = 'block';
        this.filterDropdown(container.querySelector('.wan-lora-search'));
    }

    filterDropdown(input) {
        const searchTerm = input.value.toLowerCase();
        const dropdown = this.activeDropdown;
        if (dropdown) {
            Array.from(dropdown.children).forEach(option => {
                const text = option.textContent.toLowerCase();
                option.style.display = text.includes(searchTerm) ? 'block' : 'none';
            });
        }
    }

    closeActiveDropdown() {
        if (this.activeDropdown) {
            this.activeDropdown.remove();
            this.activeDropdown = null;
        }
    }

    syncConfigToWidget() {
        const hasEnabledGroups = this.loraGroups.some(group => group.enabled);
        if (!hasEnabledGroups && this.loraGroups.length > 0) {
            this.errorMessage.style.display = 'block';
            this.errorMessage.textContent = 'Error: At least one group must be enabled to start inference.';
        } else {
            this.errorMessage.style.display = 'none';
        }
        
        const serializeRow = r => `${r.high_lora || 'none'},${r.high_strength},${r.low_lora || 'none'},${r.low_strength},${r.enabled ? 'true' : 'false'}`;
        // Append cfg_high, cfg_low, bypass, seed_offset, encoded_prompt
        const configString = this.loraGroups.map(g => 
            `${g.rows.map(serializeRow).join('|')}§${g.enabled}§${g.cfg_high}§${g.cfg_low}§${g.bypass_low}§${g.seed_offset}§${encodeURIComponent(g.prompt_override || "")}`
        ).join('\n');
        
        this.setWidgetValue('lora_batch_config', configString);
        this.setWidgetValue('custom_labels', this.loraGroups.map(g => g.label || '').join('\n'));
    }
}

app.registerExtension(WANCompareExtension);
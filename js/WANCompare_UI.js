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
                
                // Initialize UI asynchronously to ensure proper setup
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

                // Override setSize to ensure proper sizing
                const originalSetSize = this.setSize;
                this.setSize = function(size) {
                    if (originalSetSize) {
                        originalSetSize.call(this, size);
                    }
                    this.size = size;
                };

                // Force initial size
                this.size = [1900, 300];
            };

            // Add onConfigure to handle node loading from saved workflows
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                onConfigure?.apply(this, arguments);
                
                // Reinitialize UI when loading from saved workflow
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
        this.presets = new Map(); // For storing presets
        this.initialize();
    }

    async initialize() {
        console.log('[WANCompareUI] Initializing UI...');
        try {
            this.injectStyles();
            await this.fetchLoRAList();
            this.createCustomDOM();
            this.loadPresets();
            this.renderPresets(); // Render presets after loading
            this.parseConfigFromWidget();
            this.render();
            
            console.log('[WANCompareUI] UI initialized successfully');
            
            // Multiple attempts to force node size recalculation
            setTimeout(() => this.forceNodeResize(), 100);
            setTimeout(() => this.forceNodeResize(), 500);
            setTimeout(() => this.forceNodeResize(), 1000);
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
            console.log('[WANCompareUI] LoRA list fetched:', this.loraList);
        } catch (error) {
            console.error('[WANCompareUI] Fatal error fetching LoRA list:', error);
            this.loraList = ['Error fetching LoRAs'];
        }
    }

    loadPresets() {
        const savedPresets = localStorage.getItem('wanComparePresets');
        if (savedPresets) {
            this.presets = new Map(JSON.parse(savedPresets));
            console.log('[WANCompareUI] Presets loaded from localStorage:', Array.from(this.presets.keys()));
        } else {
            this.presets = new Map();
            console.log('[WANCompareUI] No presets found in localStorage');
        }
    }

    savePresets() {
        const presetData = JSON.stringify(Array.from(this.presets.entries()));
        localStorage.setItem('wanComparePresets', presetData);
        console.log('[WANCompareUI] Presets saved to localStorage:', Array.from(this.presets.keys()));
    }

    savePreset(name, config) {
        if (!name) {
            console.warn('[WANCompareUI] Preset name is required.');
            return;
        }
        console.log('[WANCompareUI] Saving preset:', name, config);
        this.presets.set(name, JSON.parse(JSON.stringify(config))); // Deep copy
        this.savePresets();
        this.renderPresets();
    }

    loadPreset(name) {
        if (!name) {
            console.warn('[WANCompareUI] Preset name is required.');
            return;
        }
        console.log('[WANCompareUI] Loading preset:', name);
        const config = this.presets.get(name);
        if (config) {
            this.applyPreset(JSON.parse(JSON.stringify(config))); // Deep copy
            console.log('[WANCompareUI] Preset loaded:', config);
        } else {
            console.warn('[WANCompareUI] Preset not found:', name);
        }
    }

    deletePreset(name) {
        if (!name) {
            console.warn('[WANCompareUI] Preset name is required.');
            return;
        }
        console.log('[WANCompareUI] Deleting preset:', name);
        if (this.presets.delete(name)) {
            this.savePresets();
            this.renderPresets();
            console.log('[WANCompareUI] Preset deleted successfully');
        } else {
            console.warn('[WANCompareUI] Preset not found:', name);
        }
    }

    applyPreset(config) {
        this.loraGroups = config.loraGroups || [];
        Object.entries(config.widgets || {}).forEach(([name, value]) => {
            this.setWidgetValue(name, value);
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
        if (this.presets.size === 0) {
            console.warn('[WANCompareUI] No presets available to render');
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
                // Force a redraw
                if (window.app && window.app.graph && window.app.graph.setDirtyCanvas) {
                    window.app.graph.setDirtyCanvas(true, true);
                }
                console.log('[WANCompareUI] Node size forced to:', size);
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
            :root { --wan-bg-main: #111113; --wan-bg-section: #1E1E22; --wan-bg-element: #2A2A2E; --wan-accent-purple: #7D26CD; --wan-accent-purple-light: #A158E2; --wan-accent-green: #2ECC71; --wan-accent-red: #E74C3C; --wan-text-light: #F0F0F0; --wan-text-med: #A0A0A0; --wan-text-dark: #666666; --wan-border-color: #333333; --wan-radius-lg: 16px; --wan-radius-md: 10px; --wan-radius-sm: 6px; }
            .wan-compare-container { background: var(--wan-bg-main); border: 2px solid var(--wan-accent-purple); border-radius: var(--wan-radius-lg); padding: 20px; margin-top: -10px; width: 1900px !important; font-family: 'Inter', sans-serif; color: var(--wan-text-light); box-shadow: 0 0 35px rgba(125, 38, 205, 0.6); position: relative; top: 0px; left: -10px; z-index: 1; user-select: none; box-sizing: border-box; }
            .wan-compare-header { text-align: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid var(--wan-border-color); }
            .wan-compare-title { font-family: 'Orbitron', monospace; font-size: 22px; font-weight: 700; color: var(--wan-accent-purple); text-shadow: 0 0 10px var(--wan-accent-purple-light); margin-bottom: 4px; }
            .wan-compare-subtitle { font-size: 13px; color: var(--wan-text-med); }
            .wan-compare-section { background: var(--wan-bg-section); border-radius: var(--wan-radius-md); padding: 15px 20px; margin-bottom: 15px; position: relative; }
            .wan-compare-section-title { font-family: 'Orbitron', monospace; font-size: 16px; font-weight: 700; color: var(--wan-accent-purple-light); margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }
            .wan-lora-groups-container { display: flex; flex-direction: column; gap: 15px; margin-bottom: 15px; }
            .wan-lora-stack-group { background: rgba(0,0,0,0.2); border: 1px solid var(--wan-accent-purple-light); border-radius: var(--wan-radius-md); padding: 10px; display: flex; flex-direction: column; gap: 8px; transition: opacity 0.3s ease; }
            .wan-lora-stack-group.disabled { opacity: 0.5; border-color: var(--wan-text-dark); }
            .wan-lora-header { display: grid; grid-template-columns: 2.2fr 1fr 2.2fr 1fr auto; gap: 12px; padding: 0 10px 8px; font-family: 'Orbitron', monospace; font-size: 11px; color: var(--wan-text-med); text-transform: uppercase; text-align: center; }
            .wan-lora-row { display: grid; grid-template-columns: 2.2fr 1fr 2.2fr 1fr auto; gap: 12px; align-items: center; }
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
        `;
        document.head.appendChild(style);
    }

    createCustomDOM() {
        // Ensure we don't create duplicate DOM elements
        if (this.container) {
            console.log('[WANCompareUI] DOM already exists, skipping creation');
            return;
        }

        console.log('[WANCompareUI] Creating custom DOM...');

        // Hide all existing widgets to avoid size issues
        if (this.node.widgets) {
            this.node.widgets.forEach(w => { 
                if (w.name) {
                    w.computeSize = () => [0, -4]; 
                    w.type = "hidden";
                    w.hidden = true;
                    
                    // Force hide for stubborn widgets
                    Object.defineProperty(w, 'computeSize', {
                        value: () => [0, -4],
                        writable: false
                    });
                }
            });
            console.log('[WANCompareUI] All existing widgets hidden');
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

        // Add preset UI elements
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
            } else {
                console.warn('[WANCompareUI] Preset name is required.');
            }
        };
        this.presetsContainer.appendChild(saveBtn);

        const loadBtn = document.createElement('button');
        loadBtn.className = 'wan-btn';
        loadBtn.textContent = 'Load Preset';
        loadBtn.onclick = () => {
            const name = this.presetSelect.value;
            if (name) {
                this.loadPreset(name);
            } else {
                console.warn('[WANCompareUI] Select a preset to load.');
            }
        };
        this.presetsContainer.appendChild(loadBtn);

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'wan-btn wan-delete-btn';
        deleteBtn.textContent = 'Delete Preset';
        deleteBtn.onclick = () => {
            const name = this.presetSelect.value;
            if (name) {
                this.deletePreset(name);
            } else {
                console.warn('[WANCompareUI] Select a preset to delete.');
            }
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
                enabled: true
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
        const SAMPLERS = ["euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ddim", "uni_pc", "uni_pc_bh2", "deis"];
        const SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"];
        samplerGrid.append(
            this.createNumberInput('boundary', 'Boundary', { min: 0, max: 1, step: 0.001, default: 0.875 }),
            this.createNumberInput('steps', 'Steps', { min: 1, max: 10000, step: 1, default: 8 }),
            this.createNumberInput('cfg_high_noise', 'CFG High', { min: 0, max: 100, step: 0.1, default: 1.0 }),
            this.createNumberInput('cfg_low_noise', 'CFG Low', { min: 0, max: 100, step: 0.1, default: 1.0 }),
            this.createNumberInput('sigma_shift', 'Sigma Shift', { min: 0, max: 100, step: 0.01, default: 8.0 }),
            this.createSelect('sampler_name', 'Sampler', SAMPLERS, 'euler'),
            this.createSelect('scheduler', 'Scheduler', SCHEDULERS, 'simple') // Changed default to "simple"
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
            // Add the DOM widget
            const domWidget = this.node.addDOMWidget('wan_compare_ui', 'div', this.container, { serialize: false });
            console.log('[WANCompareUI] DOM widget added successfully');
        } catch (error) {
            console.error('[WANCompareUI] Error adding widgets:', error);
        }
        
        document.addEventListener('click', (e) => {
            if (this.activeDropdown && !e.target.closest('.wan-lora-select-container')) {
                this.closeActiveDropdown();
            }
        });

        console.log('[WANCompareUI] DOM created successfully');
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
        input.value = defaultValue;
        this.setWidgetValue(name, defaultValue); // FIX: Ensure initial value is set
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
        select.value = defaultValue;
        this.setWidgetValue(name, defaultValue); // FIX: Ensure initial value is set
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
        this.setWidgetValue(name, defaultValue); // FIX: Ensure initial value is set
        toggle.className = `wan-on-off-button ${defaultValue ? 'active' : ''}`;
        toggle.textContent = defaultValue ? 'ON' : 'OFF';
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
        // Force node size recalculation with better error handling
        setTimeout(() => {
            this.forceNodeResize();
        }, 10);
    }

    setWidgetValue(name, value) {
        // Skip creating preset_data widget since we use node properties instead
        if (name === 'preset_data') {
            return; // Don't create this widget, we handle presets differently
        }
        
        let widget = this.node.widgets.find(w => w.name === name);
        if (!widget) {
            // Create widget if it doesn't exist
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
            const [rowStr, enabledStr] = line.split('§');
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
                    enabled: enabledStr === 'true'
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
        const header = document.createElement('div');
        header.className = 'wan-lora-header';
        header.innerHTML = '<span>High LoRA</span><span>Strength</span><span>Low LoRA</span><span>Strength</span><span>Actions</span>';
        groupEl.appendChild(header);

        group.rows.forEach((row, rowIdx) => {
            const rowEl = document.createElement('div');
            rowEl.className = `wan-lora-row ${!row.enabled ? 'disabled' : ''}`;
            rowEl.innerHTML = `
                <div class="wan-lora-select-container"><input type="text" class="wan-lora-search" placeholder="Select LoRA" value="${row.high_lora || ''}"></div>
                <input type="number" class="wan-input" value="${row.high_strength}" min="0" max="2" step="0.01">
                <div class="wan-lora-select-container"><input type="text" class="wan-lora-search" placeholder="Select LoRA" value="${row.low_lora || ''}"></div>
                <input type="number" class="wan-input" value="${row.low_strength}" min="0" max="2" step="0.01">
                <div class="wan-row-actions">
                    <button class="wan-lora-action-btn wan-row-toggle ${row.enabled ? 'active' : ''}">✓</button>
                    <button class="wan-lora-action-btn">+</button>
                    <button class="wan-lora-action-btn wan-delete-btn">✖</button>
                </div>
            `;
            const [highInputContainer, highStrengthInput, lowInputContainer, lowStrengthInput, actions] = rowEl.children;
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
            // Perform a deep copy to prevent shared references
            const duplicatedGroup = JSON.parse(JSON.stringify(group));
            duplicatedGroup.label = ''; // Keep label empty for duplicated group
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
        footer.append(labelInput, toggleGroupBtn, actions);
        groupEl.appendChild(footer);

        return groupEl;
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
        this.activeDropdown.prepend(emptyOption); // Add 'None' at the top
        
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
        const hasEnabledGroups = this.loraGroups.some(group => group.enabled);  // Allow enabled groups even with disabled rows
        if (!hasEnabledGroups && this.loraGroups.length > 0) {
            this.errorMessage.style.display = 'block';
            this.errorMessage.textContent = 'Error: At least one group must be enabled to start inference.';
        } else {
            this.errorMessage.style.display = 'none';
        }
        
        const serializeRow = r => `${r.high_lora || 'none'},${r.high_strength},${r.low_lora || 'none'},${r.low_strength},${r.enabled ? 'true' : 'false'}`;
        const configString = this.loraGroups.map(g => `${g.rows.map(serializeRow).join('|')}§${g.enabled}`).join('\n');
        this.setWidgetValue('lora_batch_config', configString);
        this.setWidgetValue('custom_labels', this.loraGroups.map(g => g.label || '').join('\n'));
    }
}

app.registerExtension(WANCompareExtension);
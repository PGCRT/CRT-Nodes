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
                this.WANCompareUIInstance = new WANCompareUI(this);
                
                this.computeSize = function() {
                    const MINIMUM_HEIGHT = 228; 

                    if (this.WANCompareUIInstance?.container) {
                        const content_height = this.WANCompareUIInstance.container.offsetHeight;
                        const new_height = Math.max(MINIMUM_HEIGHT, content_height + 20);
                        return [1900, new_height];
                    }
                    
                    return [1900, MINIMUM_HEIGHT]; 
                };
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
        this.initialize();
    }

    async initialize() {
        this.injectStyles();
        await this.fetchLoRAList();
        this.createCustomDOM();
        this.parseConfigFromWidget();
        this.render();
    }

    injectStyles() {
        if (document.getElementById('wan-compare-styles-crt')) return;
        const style = document.createElement("style");
        style.id = "wan-compare-styles-crt";
        style.innerHTML = `
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
            :root { --wan-bg-main: #111113; --wan-bg-section: #1E1E22; --wan-bg-element: #2A2A2E; --wan-accent-purple: #7D26CD; --wan-accent-purple-light: #A158E2; --wan-accent-green: #2ECC71; --wan-accent-red: #E74C3C; --wan-text-light: #F0F0F0; --wan-text-med: #A0A0A0; --wan-text-dark: #666666; --wan-border-color: #333333; --wan-radius-lg: 16px; --wan-radius-md: 10px; --wan-radius-sm: 6px; }
            .wan-compare-container { background: var(--wan-bg-main); border: 2px solid var(--wan-accent-purple); border-radius: var(--wan-radius-lg); padding: 20px; margin-top: -10px; width: 1900px !important; font-family: 'Inter', sans-serif; color: var(--wan-text-light); box-shadow: 0 0 35px rgba(125, 38, 205, 0.6); position: relative; top: 0px; left: -10px; z-index: 1; user-select: none; }
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
            .wan-row-actions { display: flex; align-items: center; justify-content: flex-end; gap: 8px; width: 120px; }
            .wan-group-footer { display: flex; align-items: center; gap: 12px; margin-top: 5px; padding: 0 10px; }
            .wan-group-actions { display: flex; align-items: center; gap: 8px; margin-left: auto; }
            .wan-btn { background: var(--wan-accent-purple); color: var(--wan-text-light); border: none; border-radius: var(--wan-radius-md); padding: 10px 20px; font-family: 'Orbitron', monospace; font-size: 13px; cursor: pointer; transition: all 0.2s ease; display: flex; align-items: center; justify-content: center; gap: 8px; width: 100%; }
            .wan-btn:hover { background: var(--wan-accent-purple-light); box-shadow: 0 0 15px var(--wan-accent-purple-light); }
            .wan-lora-action-btn { background: rgba(125, 38, 205, 0.7); color: white; border: none; border-radius: 50%; width: 24px; height: 24px; font-size: 16px; font-weight: bold; line-height: 24px; text-align: center; cursor: pointer; transition: all 0.2s ease; flex-shrink: 0; }
            .wan-lora-action-btn:hover { transform: scale(1.1); background: var(--wan-accent-purple-light); }
            .wan-on-off-button { border-radius: var(--wan-radius-md); padding: 8px 18px; font-family: 'Orbitron', monospace; font-size: 12px; font-weight: bold; cursor: pointer; transition: all 0.3s ease; text-align: center; }
            .wan-on-off-button.active { background: var(--wan-accent-green); color: var(--wan-bg-main); border: 2px solid var(--wan-accent-green); box-shadow: 0 0 10px var(--wan-accent-green); }
            .wan-on-off-button:not(.active) { background: var(--wan-text-dark); color: var(--wan-text-light); border: 2px solid var(--wan-text-dark); }
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
        `;
        document.head.appendChild(style);
    }

    createCustomDOM() {
        this.node.widgets?.forEach(w => { if (w.name) w.computeSize = () => [0, -4]; });
        this.container = document.createElement('div');
        this.container.className = 'wan-compare-container';
        
        const header = document.createElement('div');
        header.className = 'wan-compare-header';
        header.innerHTML = `<div class="wan-compare-title">LoRA Compare Sampler</div><div class="wan-compare-subtitle">Advanced High/Low Noise LoRA Comparison Tool</div>`;
        this.container.appendChild(header);

        const loraSection = this.createSection("LoRA Configurations", "🎨");
        this.groupsContainer = document.createElement('div');
        this.groupsContainer.className = 'wan-lora-groups-container';
        
        const addGroupBtn = document.createElement('button');
        addGroupBtn.className = 'wan-btn';
        addGroupBtn.innerHTML = '<span>➕</span> Add Comparison Group';
        addGroupBtn.onclick = () => {
            this.loraGroups.push({
                rows: [{ high_lora: '', high_strength: 1.0, low_lora: '', low_strength: 1.0 }],
                label: '',
                enabled: true
            });
            this.render();
        };
        loraSection.append(this.groupsContainer, addGroupBtn);
        
        const dimensionsSection = this.createSection("Dimensions & Generation", "📐");
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
            this.createSelect('scheduler', 'Scheduler', SCHEDULERS, 'normal')
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
        this.node.addDOMWidget('wan_compare_ui', 'div', this.container, { serialize: false });
        
        document.addEventListener('click', (e) => {
            if (this.activeDropdown && !e.target.closest('.wan-lora-select-container')) {
                this.closeActiveDropdown();
            }
        });
    }

    async fetchLoRAList() {
        try {
            console.log('[WANCompareUI] Starting robust LoRA fetch...');
            let finalLoras = [];
            
            // Primary attempt: Fetch from /loras endpoint, expecting LoRA models only
            try {
                const response = await fetch('/loras');
                if (response.ok) {
                    const data = await response.json();
                    if (Array.isArray(data) && data.length > 0) {
                        // Filter to include only .safetensors files
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

            // Fallback: Deep search through /object_info, but filter strictly for LoRA models
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
                this.loraList = [];
            }
        } catch (error) {
            console.error('[WANCompareUI] Critical failure in fetchLoRAList:', error);
            this.loraList = [];
        }
    }

    render() {
        if (!this.groupsContainer) return;
        this.groupsContainer.innerHTML = '';
        this.loraGroups.forEach((groupData, groupIndex) => {
            const groupEl = this.createGroupElement(groupData, groupIndex);
            this.groupsContainer.appendChild(groupEl);
        });
        this.syncConfigToWidget();
        this.node.size = this.node.computeSize();
        this.node.setDirtyCanvas(true, true);
    }

    createGroupElement(groupData, groupIndex) {
        const groupEl = document.createElement('div');
        groupEl.className = 'wan-lora-stack-group';
        groupEl.classList.toggle('disabled', !groupData.enabled);
        const header = document.createElement('div');
        header.className = 'wan-lora-header';
        header.innerHTML = `<div>High Noise LoRA</div><div>Strength</div><div>Low Noise LoRA</div><div>Strength</div><div>Actions</div>`;
        groupEl.appendChild(header);
        const isStack = groupData.rows.length > 1;
        groupData.rows.forEach((rowData, rowIndex) => {
            const rowEl = this.createLoraRowElement(groupData, groupIndex, rowIndex);
            groupEl.appendChild(rowEl);
        });
        if (isStack) {
            const footerEl = document.createElement('div');
            footerEl.className = 'wan-group-footer';
            const labelInput = document.createElement('input');
            labelInput.className = 'wan-input';
            labelInput.placeholder = 'LoRA Stack Name (optional)';
            labelInput.style.flexGrow = '1';
            labelInput.value = groupData.label || '';
            labelInput.oninput = (e) => { this.loraGroups[groupIndex].label = e.target.value; this.syncConfigToWidget(); };
            const groupActions = this.createGroupActions(groupIndex, true);
            footerEl.append(labelInput, groupActions);
            groupEl.appendChild(footerEl);
        }
        return groupEl;
    }

    createLoraRowElement(groupData, groupIndex, rowIndex) {
        const rowData = groupData.rows[rowIndex];
        const rowEl = document.createElement('div');
        rowEl.className = 'wan-lora-row';
        const createHandler = (prop) => (e) => { this.loraGroups[groupIndex].rows[rowIndex][prop] = e.target.value; this.syncConfigToWidget(); };
        const highLoraSelect = this.createLoRASelect(rowData.high_lora);
        highLoraSelect.querySelector('select').onchange = createHandler('high_lora');
        const highStrengthInput = this.createStrengthInput(rowData.high_strength);
        highStrengthInput.oninput = createHandler('high_strength');
        const lowLoraSelect = this.createLoRASelect(rowData.low_lora);
        lowLoraSelect.querySelector('select').onchange = createHandler('low_lora');
        const lowStrengthInput = this.createStrengthInput(rowData.low_strength);
        lowStrengthInput.oninput = createHandler('low_strength');
        const actionsContainer = document.createElement('div');
        actionsContainer.className = 'wan-row-actions';
        if (groupData.rows.length === 1) {
             const groupActions = this.createGroupActions(groupIndex, false);
             actionsContainer.appendChild(groupActions);
        } else {
            const removeRowBtn = document.createElement('button');
            removeRowBtn.className = 'wan-lora-action-btn';
            removeRowBtn.innerHTML = '&#10005;';
            removeRowBtn.style.background = 'var(--wan-accent-red)';
            removeRowBtn.onclick = () => { this.loraGroups[groupIndex].rows.splice(rowIndex, 1); this.render(); };
            actionsContainer.appendChild(removeRowBtn);
        }
        const addRowBtn = document.createElement('button');
        addRowBtn.className = 'wan-lora-action-btn';
        addRowBtn.innerHTML = '&#43;';
        addRowBtn.onclick = () => {
            this.loraGroups[groupIndex].rows.splice(rowIndex + 1, 0, { high_lora: '', high_strength: 1.0, low_lora: '', low_strength: 1.0 });
            this.render();
        };
        actionsContainer.appendChild(addRowBtn);
        rowEl.append(highLoraSelect, highStrengthInput, lowLoraSelect, lowStrengthInput, actionsContainer);
        return rowEl;
    }
    
    createGroupActions(groupIndex, isStack) {
        const actionsEl = document.createElement('div'); actionsEl.className = 'wan-group-actions';
        const toggleBtn = document.createElement('button'); toggleBtn.className = 'wan-on-off-button';
        toggleBtn.style.height = isStack ? '38px' : 'auto';
        const updateToggle = () => {
            const enabled = this.loraGroups[groupIndex].enabled;
            toggleBtn.textContent = enabled ? 'ON' : 'OFF';
            toggleBtn.classList.toggle('active', enabled);
        };
        toggleBtn.onclick = () => { this.loraGroups[groupIndex].enabled = !this.loraGroups[groupIndex].enabled; this.render(); };
        const removeGroupBtn = document.createElement('button'); removeGroupBtn.className = 'wan-lora-action-btn'; removeGroupBtn.style.width = '28px'; removeGroupBtn.style.height = '28px'; removeGroupBtn.style.background = 'var(--wan-accent-red)'; removeGroupBtn.textContent = '×';
        removeGroupBtn.onclick = () => { this.loraGroups.splice(groupIndex, 1); this.render(); };
        actionsEl.append(toggleBtn, removeGroupBtn);
        updateToggle();
        return actionsEl;
    }

    syncConfigToWidget() {
        const serializeRow = r => `${r.high_lora || 'none'},${r.high_strength},${r.low_lora || 'none'},${r.low_strength}`;
        const configString = this.loraGroups.map(g => `${g.rows.map(serializeRow).join('|')}§${g.enabled}`).join('\n');
        this.setWidgetValue('lora_batch_config', configString);
        this.setWidgetValue('custom_labels', this.loraGroups.map(g => g.label || '').join('\n'));
    }

    parseConfigFromWidget() {
        const configString = this.getWidgetValue('lora_batch_config', '');
        const labels = (this.getWidgetValue('custom_labels', '') || '').split('\n');
        if (!configString.trim()) { this.loraGroups = []; return; }
        try {
            this.loraGroups = configString.split('\n').map((line, groupIndex) => {
                const [rowsStr, enabledStr] = line.split('§');
                const rows = rowsStr.split('|').map(rowStr => {
                    const parts = rowStr.split(',');
                    return { high_lora: parts[0] === 'none' ? '' : parts[0], high_strength: parseFloat(parts[1]), low_lora: parts[2] === 'none' ? '' : parts[2], low_strength: parseFloat(parts[3]) };
                });
                return { rows, enabled: enabledStr === 'true', label: labels[groupIndex] || '' };
            }).filter(g => g.rows.length > 0);
        } catch (e) { this.loraGroups = []; }
    }
    
    createStrengthInput(value) { 
        const input = document.createElement('input'); 
        input.type = 'number'; 
        input.className = 'wan-input'; 
        input.value = value; 
        input.step = 0.05; 
        input.min = -2; 
        input.max = 2; 
        return input; 
    }

    createLoRASelect(selectedValue) {
        const container = document.createElement('div');
        container.className = 'wan-lora-select-container';
        const searchInput = document.createElement('input');
        searchInput.className = 'wan-lora-search';
        searchInput.type = 'text';
        searchInput.placeholder = 'Select LoRA or leave empty...';
        const dropdown = document.createElement('div');
        dropdown.className = 'wan-lora-dropdown-portal';
        document.body.appendChild(dropdown);
        const select = document.createElement('select');
        select.className = 'wan-select';
        select.style.display = 'none';
        
        const emptyOpt = document.createElement('option');
        emptyOpt.value = '';
        emptyOpt.textContent = '';
        select.appendChild(emptyOpt);
        
        const emptyDropdownOption = document.createElement('div');
        emptyDropdownOption.className = 'wan-lora-option empty-option';
        emptyDropdownOption.textContent = '(No LoRA)';
        emptyDropdownOption.dataset.value = '';
        emptyDropdownOption.onclick = (e) => {
            e.stopPropagation();
            select.value = '';
            searchInput.value = '';
            this.closeActiveDropdown();
            select.dispatchEvent(new Event('change'));
        };
        dropdown.appendChild(emptyDropdownOption);
        
        this.loraList.forEach(option => {
            const opt = document.createElement('option');
            opt.value = option;
            opt.textContent = option.split(/[\\/]/).pop();
            select.appendChild(opt);
            const dropdownOption = document.createElement('div');
            dropdownOption.className = 'wan-lora-option';
            dropdownOption.textContent = opt.textContent;
            dropdownOption.dataset.value = option;
            dropdownOption.onclick = (e) => {
                e.stopPropagation();
                select.value = option;
                searchInput.value = opt.textContent.replace(".safetensors", "");
                this.closeActiveDropdown();
                select.dispatchEvent(new Event('change'));
            };
            dropdown.appendChild(dropdownOption);
        });
        
        select.value = selectedValue || '';
        
        if (selectedValue && selectedValue.trim()) {
            const matchingLora = this.loraList.find(opt => opt === selectedValue);
            searchInput.value = matchingLora ? matchingLora.split(/[\\/]/).pop().replace(".safetensors", "") : '';
        } else {
            searchInput.value = '';
        }

        const filterOptions = () => {
            const term = searchInput.value.toLowerCase();
            dropdown.querySelectorAll('.wan-lora-option').forEach(opt => {
                if (opt.classList.contains('empty-option')) {
                    opt.style.display = (!term || '(no lora)'.includes(term)) ? 'block' : 'none';
                } else {
                    opt.style.display = opt.textContent.toLowerCase().includes(term) ? 'block' : 'none';
                }
            });
        };

        const showDropdown = () => {
            this.closeActiveDropdown();
            const rect = searchInput.getBoundingClientRect();
            const viewportHeight = window.innerHeight;
            const dropdownMaxHeight = 200;
            dropdown.style.left = `${rect.left}px`;
            dropdown.style.width = `${rect.width}px`;
            if (rect.bottom + dropdownMaxHeight > viewportHeight && rect.top > dropdownMaxHeight) {
                dropdown.style.top = `${rect.top - dropdownMaxHeight}px`;
                dropdown.style.maxHeight = `${Math.min(dropdownMaxHeight, rect.top)}px`;
            } else {
                dropdown.style.top = `${rect.bottom}px`;
                dropdown.style.maxHeight = `${Math.min(dropdownMaxHeight, viewportHeight - rect.bottom - 10)}px`;
            }
            dropdown.style.display = 'block';
            this.activeDropdown = dropdown;
            filterOptions();
        };

        searchInput.oninput = () => {
            if (dropdown.style.display !== 'block') { showDropdown(); }
            filterOptions();
        };
        searchInput.onfocus = (e) => { e.stopPropagation(); showDropdown(); };
        container.addEventListener('DOMNodeRemoved', () => { if (dropdown.parentNode) { dropdown.parentNode.removeChild(dropdown); } });
        container.append(searchInput, select);
        return container;
    }
    
    closeActiveDropdown() { 
        if (this.activeDropdown) { 
            this.activeDropdown.style.display = 'none';
            this.activeDropdown = null; 
        } 
    }

    createSection(title, icon) { 
        const section = document.createElement('div'); 
        section.className = 'wan-compare-section'; 
        const titleEl = document.createElement('div'); 
        titleEl.className = 'wan-compare-section-title'; 
        titleEl.innerHTML = `<span>${icon}</span> ${title}`; 
        section.appendChild(titleEl); 
        return section; 
    }

    getWidgetValue(name, defaultValue) { 
        const widget = this.node.widgets?.find(w => w.name === name); 
        return widget?.value ?? defaultValue; 
    }

    setWidgetValue(name, value) { 
        const widget = this.node.widgets?.find(w => w.name === name); 
        if (widget) widget.value = value; 
    }

    createNumberInput(name, label, params) { 
        const container = document.createElement('div'); 
        container.className = 'wan-control-group'; 
        const labelEl = document.createElement('label'); 
        labelEl.className = 'wan-control-label'; 
        labelEl.textContent = label; 
        const input = document.createElement('input'); 
        input.type = 'number'; 
        input.className = 'wan-input'; 
        Object.assign(input, params); 
        input.value = this.getWidgetValue(name, params.default); 
        input.oninput = e => this.setWidgetValue(name, parseFloat(e.target.value)); 
        container.append(labelEl, input); 
        return container; 
    }

    createSelect(name, label, options, defaultOption) { 
        const container = document.createElement('div'); 
        container.className = 'wan-control-group'; 
        const labelEl = document.createElement('label'); 
        labelEl.className = 'wan-control-label'; 
        labelEl.textContent = label; 
        const select = document.createElement('select'); 
        select.className = 'wan-input wan-select'; 
        options.forEach(opt => { 
            const option = document.createElement('option'); 
            option.value = opt; 
            option.textContent = opt; 
            select.appendChild(option); 
        }); 
        select.value = this.getWidgetValue(name, defaultOption || options[0]); 
        select.onchange = e => this.setWidgetValue(name, e.target.value); 
        container.append(labelEl, select); 
        return container; 
    }

    createToggle(name, label, defaultValue) {
        const container = document.createElement('div'); 
        container.className = 'wan-control-group';
        const labelEl = document.createElement('label'); 
        labelEl.className = 'wan-control-label'; 
        labelEl.textContent = label;
        const button = document.createElement('button'); 
        button.className = 'wan-on-off-button';
        if (this.getWidgetValue(name) === undefined) { 
            this.setWidgetValue(name, defaultValue); 
        }
        const initialValue = this.getWidgetValue(name);
        const updateVisual = (value) => { 
            button.classList.toggle('active', !!value); 
            button.textContent = value ? 'ENABLED' : 'DISABLED'; 
        };
        button.onclick = () => { 
            const newValue = !this.getWidgetValue(name); 
            this.setWidgetValue(name, newValue); 
            updateVisual(newValue); 
        };
        container.append(labelEl, button);
        updateVisual(initialValue);
        return container;
    }
}

app.registerExtension(WANCompareExtension);
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
                
                const BACKEND_MIN_WIDTH = 1900;
                const FIXED_BACKEND_HEIGHT = 0;

                this.WANCompareUIInstance = new WANCompareUI(this);

                this.computeSize = function() {
                    return [BACKEND_MIN_WIDTH, FIXED_BACKEND_HEIGHT];
                };
                
                this.size = this.computeSize();
            };
        }
    }
};

class WANCompareUI {
    constructor(node) {
        this.node = node;
        this.container = null;
        this.loraRows = [];
        this.loraList = [];
        this.CONTAINER_WIDTH = 1900;
        this.activeDropdown = null;
        this.initialize();
    }

    async initialize() {
        this.injectStyles();
        await this.fetchLoRAList();
        this.createCustomDOM();
        this.parseConfigFromWidget();
        this.renderLoRaRows();
    }

    injectStyles() {
        if (document.getElementById('wan-compare-styles-crt')) return;
        const style = document.createElement("style");
        style.id = "wan-compare-styles-crt";
        style.innerHTML = `
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
            :root { --wan-bg-main: #111113; --wan-bg-section: #1E1E22; --wan-bg-element: #2A2A2E; --wan-accent-purple: #7D26CD; --wan-accent-purple-light: #A158E2; --wan-accent-green: #2ECC71; --wan-accent-red: #E74C3C; --wan-text-light: #F0F0F0; --wan-text-med: #A0A0A0; --wan-text-dark: #666666; --wan-border-color: #333333; --wan-radius-lg: 16px; --wan-radius-md: 10px; --wan-radius-sm: 6px; }
            .wan-compare-container { background: var(--wan-bg-main); border: 2px solid var(--wan-accent-purple); border-radius: var(--wan-radius-lg); padding: 20px; margin-top: -10px; width: ${this.CONTAINER_WIDTH}px !important; height: auto; min-height: fit-content; box-sizing: border-box; font-family: 'Inter', sans-serif; color: var(--wan-text-light); box-shadow: 0 0 35px rgba(125, 38, 205, 0.6); position: relative; top: 0px; left: -10px; z-index: 1; -webkit-user-select: none; user-select: none; }
            .wan-compare-header { text-align: center; margin-bottom: 20px; padding-bottom: 15px; }
            .wan-compare-title { font-family: 'Orbitron', monospace; font-size: 22px; font-weight: 700; color: var(--wan-accent-purple); text-shadow: 0 0 10px var(--wan-accent-purple-light); margin-bottom: 4px; }
            .wan-compare-subtitle { font-size: 13px; color: var(--wan-text-med); }
            .wan-compare-section { background: var(--wan-bg-section); border-radius: var(--wan-radius-md); padding: 15px 20px; margin-bottom: 15px; position: relative; }
            .wan-compare-section-title { font-family: 'Orbitron', monospace; font-size: 16px; font-weight: 700; color: var(--wan-accent-purple-light); margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }
            .wan-lora-header { display: grid; grid-template-columns: 2fr 1fr 2fr 1fr 2fr 120px; gap: 12px; padding: 0 10px 8px 10px; font-family: 'Orbitron', monospace; font-size: 11px; color: var(--wan-text-med); text-transform: uppercase; text-align: center; }
            .wan-lora-rows { display: flex; flex-direction: column; gap: 8px; padding: 8px; background: rgba(0,0,0,0.2); border-radius: var(--wan-radius-sm); margin-bottom: 12px; min-height: 50px; }
            .wan-lora-row { display: grid; grid-template-columns: 2fr 1fr 2fr 1fr 2fr 120px; gap: 12px; align-items: center; padding: 8px; background: var(--wan-bg-element); border-radius: var(--wan-radius-sm); border: 1px solid var(--wan-border-color); position: relative; transition: opacity 0.3s ease; }
            .wan-lora-row.disabled { opacity: 0.5; background: #222; }
            .wan-lora-actions { display: flex; align-items: center; justify-content: center; gap: 8px; }
            .wan-btn { background: var(--wan-accent-purple); color: var(--wan-text-light); border: none; border-radius: var(--wan-radius-md); padding: 10px 20px; font-family: 'Orbitron', monospace; font-size: 13px; cursor: pointer; transition: all 0.2s ease; display: flex; align-items: center; justify-content: center; gap: 8px; width: 100%; }
            .wan-btn:hover { background: var(--wan-accent-purple-light); box-shadow: 0 0 15px var(--wan-accent-purple-light); }
            .wan-lora-remove-btn { background: var(--wan-accent-red); color: white; border: none; border-radius: 50%; width: 28px; height: 28px; font-size: 16px; font-weight: bold; cursor: pointer; transition: all 0.2s ease; flex-shrink: 0; }
            .wan-lora-remove-btn:hover { transform: scale(1.1); }
            .wan-control-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
            .wan-control-group { display: flex; flex-direction: column; gap: 6px; }
            .wan-control-label { font-family: 'Orbitron', monospace; font-size: 11px; color: var(--wan-text-med); text-transform: uppercase; margin-left: 5px; }
            .wan-input, .wan-select { background: var(--wan-bg-element); border: 1px solid var(--wan-border-color); border-radius: var(--wan-radius-sm); color: var(--wan-text-light); padding: 8px 12px; font-family: 'Inter', sans-serif; font-size: 13px; transition: all 0.2s ease; width: 100%; box-sizing: border-box; -webkit-user-select: text; user-select: text; }
            .wan-input:focus, .wan-select:focus { outline: none; border-color: var(--wan-accent-purple); box-shadow: 0 0 8px rgba(125, 38, 205, 0.7); }
            .wan-on-off-button { background: var(--wan-bg-element); border: 2px solid var(--wan-border-color); border-radius: var(--wan-radius-sm); padding: 8px 15px; font-family: 'Orbitron', monospace; font-size: 12px; font-weight: bold; text-transform: uppercase; color: var(--wan-text-med); cursor: pointer; transition: all 0.3s ease; text-align: center; height: 38px; }
            .wan-on-off-button.wan-row-toggle { height: 28px; padding: 5px 10px; font-size: 11px; flex-grow: 1; }
            .wan-on-off-button:hover { border-color: var(--wan-accent-purple-light); }
            .wan-on-off-button.active { background: var(--wan-accent-green); border-color: var(--wan-accent-green); color: var(--wan-bg-main); box-shadow: 0 0 10px var(--wan-accent-green); }
            .wan-on-off-button.wan-row-toggle:not(.active) { background: var(--wan-text-dark); border-color: var(--wan-text-dark); }
            .wan-lora-select-container { position: relative; width: 100%; }
            .wan-lora-search { width: 100%; background: var(--wan-bg-element); border: 1px solid var(--wan-border-color); border-radius: var(--wan-radius-sm); color: var(--wan-text-light); padding: 8px 12px; font-family: 'Inter', sans-serif; font-size: 13px; }
            .wan-lora-search:focus { outline: none; border-color: var(--wan-accent-purple); box-shadow: 0 0 8px rgba(125, 38, 205, 0.7); }
            .wan-lora-search::placeholder { color: var(--wan-text-dark); font-style: italic; }
            .wan-lora-dropdown-portal { position: fixed !important; max-height: 200px; overflow-y: auto; background: var(--wan-bg-element); border: 1px solid var(--wan-border-color); border-radius: var(--wan-radius-sm); z-index: 999999 !important; display: none; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.8); min-width: 200px; }
            .wan-lora-option { padding: 8px 12px; cursor: pointer; color: var(--wan-text-light); font-family: 'Inter', sans-serif; font-size: 13px; border-bottom: 1px solid rgba(255, 255, 255, 0.1); }
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
        const loraHeader = document.createElement('div');
        loraHeader.className = 'wan-lora-header';
        loraHeader.innerHTML = `<div>High Noise LoRA</div><div>Strength</div><div>Low Noise LoRA</div><div>Strength</div><div>Custom Label</div><div>Action</div>`;
        this.rowsContainer = document.createElement('div');
        this.rowsContainer.className = 'wan-lora-rows';
        const addBtn = document.createElement('button');
        addBtn.className = 'wan-btn';
        addBtn.innerHTML = '<span>➕</span> Add Comparison';
        addBtn.onclick = () => {
            this.loraRows.push({ high_lora: '', low_lora: '', high_strength: 0.0, low_strength: 0.0, label: '', enabled: true });
            this.renderLoRaRows();
            this.syncConfigToWidget();
        };
        loraSection.append(loraHeader, this.rowsContainer, addBtn);

        const dimensionsSection = this.createSection("Dimensions & Generation", "📐");
        const dimensionsGrid = document.createElement('div');
        dimensionsGrid.className = 'wan-control-grid';
        dimensionsGrid.style.gridTemplateColumns = 'repeat(3, 1fr)';
        dimensionsGrid.append(
            this.createNumberInput('width', 'Width', { min: 16, max: 4096, step: 16, default: 432 }),
            this.createNumberInput('height', 'Height', { min: 16, max: 4096, step: 16, default: 768 }),
            this.createNumberInput('frame_count', 'Frame Count', { min: 1, max: 4096, step: 1, default: 81 })
        );
        dimensionsSection.appendChild(dimensionsGrid);

        const samplerSection = this.createSection("Sampler Settings", "⚙️");
        const samplerGrid = document.createElement('div');
        samplerGrid.className = 'wan-control-grid';
        const SAMPLERS = ["euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ddim", "uni_pc", "uni_pc_bh2", "deis"];
        const SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"];
        samplerGrid.append(
            this.createNumberInput('boundary', 'Boundary', { min: 0, max: 1, step: 0.001, default: 0.875 }),
            this.createNumberInput('steps', 'Steps', { min: 1, max: 10000, step: 1, default: 8 }),
            this.createNumberInput('cfg_high_noise', 'CFG High', { min: 0, max: 100, step: 0.1, default: 1.0 }),
            this.createNumberInput('cfg_low_noise', 'CFG Low (Set "0" to bypass)', { min: 0, max: 100, step: 0.1, default: 1.0 }),
            this.createNumberInput('sigma_shift', 'Sigma Shift', { min: 0, max: 100, step: 0.01, default: 8.0 }),
            this.createSelect('sampler_name', 'Sampler', SAMPLERS, 'euler'),
            this.createSelect('scheduler', 'Scheduler', SCHEDULERS, 'normal')
        );
        samplerSection.appendChild(samplerGrid);

        const outputSection = this.createSection("Output Settings", "📤");
        const outputGrid = document.createElement('div');
        outputGrid.className = 'wan-control-grid';
        outputGrid.style.gridTemplateColumns = 'repeat(2, 1fr)';
        outputGrid.append(
            this.createToggle('enable_vae_decode', 'Enable VAE Decode', false),
            this.createToggle('create_comparison_grid', 'Create Comparison Grid', false),
            this.createToggle('add_labels', 'Add Custom Labels', false),
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
            
            try {
                const response = await fetch('/loras');
                if (response.ok) {
                    const data = await response.json();
                    if (Array.isArray(data) && data.length > 0) {
                        finalLoras = data;
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
                            const nodeInfo = data[nodeName];
                            if (potentialLoraNodes.some(name => nodeName.includes(name)) || nodeName.toLowerCase().includes('lora')) {
                                if (nodeInfo.input?.required) {
                                    Object.values(nodeInfo.input.required).forEach(paramData => {
                                        if (Array.isArray(paramData) && Array.isArray(paramData[0])) {
                                            paramData[0].forEach(lora => {
                                                if (lora && typeof lora === 'string' && lora.trim()) {
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
                const cleanedLoras = finalLoras
                    .filter(lora => lora && typeof lora === 'string' && lora.toLowerCase().endsWith('.safetensors'))
                    .map(lora => lora.trim())
                    .sort((a, b) => {
                        const aFileName = a.split(/[\\/]/).pop().toLowerCase();
                        const bFileName = b.split(/[\\/]/).pop().toLowerCase();
                        if (aFileName === bFileName) {
                            return a.toLowerCase().localeCompare(b.toLowerCase());
                        }
                        return aFileName.localeCompare(bFileName);
                    });
                
                this.loraList = cleanedLoras;
            } else {
                this.loraList = [];
            }

        } catch (error) {
            console.error('[WANCompareUI] Critical failure in fetchLoRAList:', error);
            this.loraList = [];
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

    renderLoRaRows() {
        if (!this.rowsContainer) return;
        this.rowsContainer.innerHTML = '';
        this.loraRows.forEach((rowData, index) => {
            const row = document.createElement('div');
            row.className = 'wan-lora-row';
            row.classList.toggle('disabled', !rowData.enabled);

            const createHandler = (prop, isFloat = false) => e => {
                this.loraRows[index][prop] = isFloat ? (parseFloat(e.target.value) || 0) : e.target.value;
                this.syncConfigToWidget();
            };

            const highLoraSelect = this.createLoRASelect(rowData.high_lora);
            highLoraSelect.querySelector('select').onchange = createHandler('high_lora');
            const highStrengthInput = this.createStrengthInput(rowData.high_strength);
            highStrengthInput.oninput = createHandler('high_strength', true);
            const lowLoraSelect = this.createLoRASelect(rowData.low_lora);
            lowLoraSelect.querySelector('select').onchange = createHandler('low_lora');
            const lowStrengthInput = this.createStrengthInput(rowData.low_strength);
            lowStrengthInput.oninput = createHandler('low_strength', true);
            
            const labelInput = document.createElement('input');
            labelInput.type = 'text';
            labelInput.className = 'wan-input';
            labelInput.placeholder = 'Auto (uses LoRA name)';
            labelInput.value = rowData.label || '';
            labelInput.oninput = createHandler('label');

            const actionsContainer = document.createElement('div');
            actionsContainer.className = 'wan-lora-actions';
            
            const toggleBtn = document.createElement('button');
            toggleBtn.className = 'wan-on-off-button wan-row-toggle';
            const updateToggleVisual = () => {
                toggleBtn.classList.toggle('active', !!rowData.enabled);
                toggleBtn.textContent = rowData.enabled ? 'On' : 'Off';
            };
            toggleBtn.onclick = () => {
                this.loraRows[index].enabled = !this.loraRows[index].enabled;
                row.classList.toggle('disabled', !this.loraRows[index].enabled);
                updateToggleVisual();
                this.syncConfigToWidget();
            };
            
            const removeBtn = document.createElement('button');
            removeBtn.className = 'wan-lora-remove-btn';
            removeBtn.innerHTML = '×';
            removeBtn.onclick = () => {
                this.loraRows.splice(index, 1);
                this.renderLoRaRows();
                this.syncConfigToWidget();
            };
            
            actionsContainer.append(toggleBtn, removeBtn);
            row.append(highLoraSelect, highStrengthInput, lowLoraSelect, lowStrengthInput, labelInput, actionsContainer);
            this.rowsContainer.appendChild(row);
            updateToggleVisual();
        });
    }

    closeActiveDropdown() {
        if (this.activeDropdown) {
            this.activeDropdown.style.display = 'none';
            this.activeDropdown = null;
        }
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
                searchInput.value = opt.textContent;
                this.closeActiveDropdown();
                select.dispatchEvent(new Event('change'));
            };
            dropdown.appendChild(dropdownOption);
        });
        
        select.value = selectedValue || '';
        
        if (selectedValue && selectedValue.trim()) {
            const matchingLora = this.loraList.find(opt => opt === selectedValue);
            searchInput.value = matchingLora ? matchingLora.split(/[\\/]/).pop() : '';
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
        searchInput.onblur = () => { setTimeout(() => { if (!dropdown.matches(':hover')) { this.closeActiveDropdown(); } }, 150); };
        container.addEventListener('DOMNodeRemoved', () => { if (dropdown.parentNode) { dropdown.parentNode.removeChild(dropdown); } });
        container.append(searchInput, select);
        return container;
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

    createNumberInput(name, label, params) {
        const group = document.createElement('div');
        group.className = 'wan-control-group';
        const labelEl = document.createElement('label');
        labelEl.className = 'wan-control-label';
        labelEl.textContent = label;
        const input = document.createElement('input');
        input.type = 'number';
        input.className = 'wan-input';
        Object.assign(input, params);
        input.value = this.getWidgetValue(name, params.default);
        input.oninput = e => this.setWidgetValue(name, parseFloat(e.target.value));
        group.append(labelEl, input);
        return group;
    }

    createSelect(name, label, options, defaultOption) {
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
        select.value = this.getWidgetValue(name, defaultOption || options[0]);
        select.onchange = e => this.setWidgetValue(name, e.target.value);
        group.append(labelEl, select);
        return group;
    }

    createToggle(name, label, defaultValue) {
        const group = document.createElement('div');
        group.className = 'wan-control-group';
        const labelEl = document.createElement('label');
        labelEl.className = 'wan-control-label';
        labelEl.textContent = label;
        const button = document.createElement('button');
        button.className = 'wan-on-off-button';
        const updateVisual = (value) => {
            button.classList.toggle('active', !!value);
            button.textContent = value ? 'Enabled' : 'Disabled';
        };
        button.onclick = () => {
            const newValue = !this.getWidgetValue(name, defaultValue);
            this.setWidgetValue(name, newValue);
            updateVisual(newValue);
        };
        group.append(labelEl, button);
        updateVisual(this.getWidgetValue(name, defaultValue));
        return group;
    }

    syncConfigToWidget() {
        const configString = this.loraRows
            .map(r => {
                const highLora = r.high_lora.trim() || 'none';
                const lowLora = r.low_lora.trim() || 'none';
                return `${highLora},${lowLora},${r.high_strength},${r.low_strength},${r.enabled}`;
            })
            .join('\n');
        this.setWidgetValue('lora_batch_config', configString);

        const labelsString = this.loraRows.map(r => r.label || '').join('\n');
        this.setWidgetValue('custom_labels', labelsString);
    }

    parseConfigFromWidget() {
        const configString = this.getWidgetValue('lora_batch_config', '');
        const labelsString = this.getWidgetValue('custom_labels', '');
        const labels = labelsString.split('\n');

        if (!configString.trim()) { 
            this.loraRows = []; 
        } else {
            this.loraRows = configString.split('\n').map((line, index) => {
                const parts = line.split(',').map(item => item.trim());
                if (parts.length >= 4) {
                    return { 
                        high_lora: parts[0] === 'none' ? '' : parts[0], 
                        low_lora: parts[1] === 'none' ? '' : parts[1], 
                        high_strength: parseFloat(parts[2]), 
                        low_strength: parseFloat(parts[3]),
                        enabled: parts.length > 4 ? parts[4] === 'true' : true,
                        label: labels[index] || ''
                    };
                }
                return null;
            }).filter(Boolean);
        }

        if (this.loraRows.length === 0) {
            this.loraRows.push({ high_lora: '', low_lora: '', high_strength: 0.0, low_strength: 0.0, label: '', enabled: true });
            this.syncConfigToWidget();
        }
    }

    getWidgetValue(name, defaultValue) {
        const widget = this.node.widgets?.find(w => w.name === name);
        return widget?.value ?? defaultValue;
    }

    setWidgetValue(name, value) {
        const widget = this.node.widgets?.find(w => w.name === name);
        if (widget) widget.value = value;
    }
}

app.registerExtension(WANCompareExtension);
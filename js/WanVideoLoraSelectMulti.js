import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Comfy.WanVideoLoraSelectMulti",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WanVideoLoraSelectMultiImproved") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            this.title_height = 0;
            this.widgets_start_y = 0;
            this.size = [1400, 0]; 
            this.bgcolor = "#00000000";
            this.color = "#00000000"; 

            if (this.widgets) {
                for (const w of this.widgets) {
                    w.computeSize = () => [0, -4];
                    w.hidden = true; 
                }
            }

            setTimeout(() => {
                this.ui = new WanVideoLoraSelectMultiUI(this);
            }, 0);
        };

        nodeType.prototype.computeSize = function () {
            const height = this.ui?.container?.offsetHeight;
            if (height && height > 50) {
                return [1400, height + 32];
            }
            return [1400, 0];
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            onConfigure?.apply(this, arguments);
            setTimeout(() => {
                if (!this.ui) this.ui = new WanVideoLoraSelectMultiUI(this);
                else this.ui.rebuild();
            }, 50);
        };
    }
});

class WanVideoLoraSelectMultiUI {
    constructor(node) {
        this.node = node;
        this.rows = [];
        this.loras = [];
        this.presets = new Map();
        this.init();
    }

    async init() {
        this.injectCSS();
        await this.loadLoras();
        this.createUI();
        this.loadPresets();
        this.parse();
        this.render();
        
        setTimeout(() => this.sync(), 100);
    }

    injectCSS() {
        if (document.getElementById("wan-final-css")) return;
        const s = document.createElement("style");
        s.id = "wan-final-css";
        s.textContent = `
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&family=Inter:wght@400;500&display=swap');
            .wan-wrap{background:transparent;padding:16px 20px;margin:-8px -10px 0;width:calc(100% + 20px);font-family:'Inter',sans-serif;color:#eee;box-sizing:border-box;user-select:none;}
            .wan-title{text-align:center;margin:0 0 18px;font-family:'Orbitron',monospace;font-size:19px;color:#c27dff;text-shadow:0 0 12px #a158e2;background:transparent;}
            .wan-bar{background:rgba(30,30,34,0.6);border:1px solid rgba(125,38,205,0.6);border-radius:12px;padding:10px;margin-bottom:14px;display:flex;gap:10px;}
            .wan-opt{background:rgba(30,30,34,0.6);border:1px solid rgba(125,38,205,0.6);border-radius:12px;padding:10px;margin-bottom:14px;display:flex;gap:20px;align-items:center;}
            .wan-opt label{display:flex;align-items:center;gap:8px;font-size:11px;color:#aaa;font-family:'Orbitron',monospace;text-transform:uppercase;}
            .wan-box{background:rgba(30,30,34,0.6);border:1px solid rgba(125,38,205,0.6);border-radius:12px;padding:10px;}
            .wan-header{display:grid;grid-template-columns:20px 3fr 1fr 3fr 1fr 80px;gap:10px;font-size:10px;color:#888;margin-bottom:8px;text-transform:uppercase;font-family:'Orbitron',monospace;text-align:center;}
            .wan-row{display:grid;grid-template-columns:20px 3fr 1fr 3fr 1fr 80px;gap:10px;align-items:center;margin-bottom:5px;background:rgba(42,42,46,0.8);padding:8px;border-radius:8px;}
            .wan-in{background:#2a2a2e;border:1px solid #444;border-radius:8px;padding:7px 10px;font-size:12px;color:#fff;width:100%;box-sizing:border-box;}
            .wan-in:focus{border-color:#c27dff;outline:none;box-shadow:0 0 10px rgba(194,125,255,0.5);}
            .wan-btn{background:#7d26cd;color:#fff;border:none;border-radius:10px;padding:8px 16px;font-family:'Orbitron',monospace;font-size:11px;cursor:pointer;transition:all 0.2s;}
            .wan-btn:hover{background:#a158e2;box-shadow:0 0 15px #a158e2;}
            .wan-del{background:#e74c3c!important;}
            .wan-del:hover{background:#ff6b5b!important;box-shadow:0 0 15px #ff6b5b!important;}
            .wan-drag{cursor:grab;color:#888;text-align:center;font-size:18px;}
            .wan-drag:active{cursor:grabbing;}
            .wan-drop{position:fixed;background:#2a2a2e;border:1px solid #7d26cd;border-radius:8px;max-height:240px;overflow-y:auto;z-index:999999;box-shadow:0 10px 30px rgba(0,0,0,0.8);}
            .wan-drop div{padding:10px 14px;cursor:pointer;font-size:12px;color:#eee;}
            .wan-drop div:hover{background:#7d26cd;color:#fff;}
            .wan-drop div[data-val=""]{color:#888;font-style:italic;}
            .wan-add-btn{margin-top:10px;text-align:center;}
            .wan-toggle-on{background:#2ecc71!important;color:#000!important;font-weight:bold;}
            .wan-toggle-off{background:#555!important;color:#ccc!important;}
        `;
        document.head.appendChild(s);
    }

    async loadLoras() {
        try {
            let list = [];
            const r = await fetch("/loras");
            if (r.ok) {
                list = await r.json();
            } else {
                const r2 = await fetch("/object_info");
                if (r2.ok) {
                    const data = await r2.json();
                    const loraNode = Object.values(data).find(n => n.input?.required?.lora_name);
                    if (loraNode) {
                        list = loraNode.input.required.lora_name[0];
                    }
                }
            }
            this.loras = list.filter(x => x.endsWith(".safetensors") || x.endsWith(".pt")).sort();
            if (!this.loras.length) this.loras = ["None found"];
        } catch (e) { 
            console.error("WanVideoLoraSelect: Error loading LoRAs", e);
            this.loras = ["Error loading"]; 
        }
    }

    createUI() {
        this.container = document.createElement("div");
        this.container.className = "wan-wrap";
        this.node.addDOMWidget("ui", "div", this.container);
    }

    loadPresets() {
        try { this.presets = new Map(JSON.parse(localStorage.getItem("WanLoraFinal2025")||"[]")); }
        catch { this.presets = new Map(); }
    }
    savePresets() { localStorage.setItem("WanLoraFinal2025", JSON.stringify(Array.from(this.presets.entries()))); }

    parse() {
        const w = this.node.widgets?.find(x => x.name === "lora_batch_config");
        this.rows = [];
        
        if (!w?.value || typeof w.value !== 'string' || !w.value.trim()) { 
            this.rows = [this.nr()]; 
            return; 
        }
        
        try {
            const [configData, valid] = w.value.split("§");
            if (valid !== "true") { 
                this.rows = [this.nr()]; 
                return; 
            }

            const lines = configData.split("|");
            for (const line of lines) {
                if(!line) continue;
                const [h, hs, l, ls, on] = line.split(",");
                if (h !== undefined) {
                    this.rows.push({
                        high: h === "none" ? "" : h,
                        hstr: parseFloat(hs) || 1.0,
                        low: l === "none" ? "" : l,
                        lstr: parseFloat(ls) || 1.0,
                        on: on === "true"
                    });
                }
            }
        } catch (e) {
            console.error("WanVideoLoraSelect: Parse error", e);
            this.rows = [this.nr()];
        }
        
        if (!this.rows.length) this.rows = [this.nr()];
    }

    nr() { return {high:"", hstr:1.0, low:"", lstr:1.0, on:true}; }

    render() {
        const mergeVal = this.getWidgetValue("merge_loras", true);
        const lowMemVal = this.getWidgetValue("low_mem_load", false);

        this.container.innerHTML = `
            <div class="wan-title">LoRA Configuration</div>
            <div class="wan-bar">
                <select class="wan-in" style="flex:1"></select>
                <input class="wan-in" style="flex:1" placeholder="Preset name">
                <button class="wan-btn" id="btn-save">Save</button>
                <button class="wan-btn" id="btn-load">Load</button>
                <button class="wan-btn wan-del" id="btn-delete">Delete</button>
            </div>
            <div class="wan-opt">
                <label><input type="checkbox" id="chk-merge" ${mergeVal?"checked":""}> Merge LoRAs</label>
                <label><input type="checkbox" id="chk-mem" ${lowMemVal?"checked":""}> Low Memory Load</label>
            </div>
            <div class="wan-box">
                <div class="wan-header">
                    <div></div><div>HIGH LORA</div><div>STR</div><div>LOW LORA</div><div>STR</div><div>ACTIONS</div>
                </div>
                <div id="rows-container"></div>
                <div class="wan-add-btn">
                    <button class="wan-btn" id="btn-add">+ Add LoRA Layer</button>
                </div>
            </div>
        `;

        const sel = this.container.querySelector("select");
        sel.innerHTML = `<option>Select preset</option>` + [...this.presets.keys()].map(n=>`<option>${n}</option>`).join("");
        
        this.container.querySelector("#btn-save").onclick = () => {
            const name = this.container.querySelector("input[placeholder='Preset name']").value;
            if(name) { this.presets.set(name, this.cfg()); this.savePresets(); this.render(); }
        };
        this.container.querySelector("#btn-load").onclick = () => {
            if(sel.value && sel.value !== "Select preset") this.apply(this.presets.get(sel.value));
        };
        this.container.querySelector("#btn-delete").onclick = () => {
            if(sel.value && sel.value !== "Select preset" && this.presets.delete(sel.value)) {
                this.savePresets(); 
                this.render();
            }
        };

        this.container.querySelector("#chk-merge").onchange = (e) => {
            this.setWidgetValue("merge_loras", e.target.checked);
        };
        this.container.querySelector("#chk-mem").onchange = (e) => {
            this.setWidgetValue("low_mem_load", e.target.checked);
        };
        
        this.container.querySelector("#btn-add").onclick = () => {
            this.rows.push(this.nr());
            this.render();
            this.sync();
        };

        const rc = this.container.querySelector("#rows-container");
        this.rows.forEach((r, i) => rc.appendChild(this.createRow(r, i)));
    }

    createRow(r, i) {
        const d = document.createElement("div");
        d.className = "wan-row";
        if (!r.on) d.style.opacity = "0.5";
        
        d.innerHTML = `
            <div class="wan-drag" draggable="true">⋮⋮</div>
            <input class="wan-in wan-search" value="${r.high}" placeholder="Select High LoRA...">
            <input class="wan-in" type="number" step="0.05" value="${r.hstr}">
            <input class="wan-in wan-search" value="${r.low}" placeholder="Select Low LoRA...">
            <input class="wan-in" type="number" step="0.05" value="${r.lstr}">
            <div style="display:flex;gap:4px;justify-content:flex-end;">
                <button class="wan-btn ${r.on?'wan-toggle-on':'wan-toggle-off'}" style="padding:4px 8px;font-size:10px;">${r.on?'ON':'OFF'}</button>
                <button class="wan-btn wan-del" style="padding:4px 8px;font-size:10px;">✕</button>
            </div>
        `;

        const dragHandle = d.querySelector(".wan-drag");
        dragHandle.ondragstart = (e) => {
            e.dataTransfer.setData("text/plain", i);
            e.dataTransfer.effectAllowed = "move";
        };
        dragHandle.ondragover = (e) => { e.preventDefault(); }; // Allow drop
        dragHandle.ondrop = (e) => {
            e.preventDefault();
            const fromIndex = parseInt(e.dataTransfer.getData("text/plain"));
            if (fromIndex !== i && !isNaN(fromIndex)) {
                const item = this.rows.splice(fromIndex, 1)[0];
                this.rows.splice(i, 0, item);
                this.render();
                this.sync();
            }
        };

        const inputs = d.querySelectorAll("input");
        
        [0, 2].forEach(idx => {
            const key = idx === 0 ? 'high' : 'low';
            const inp = inputs[idx];
            
            inp.oninput = (e) => { 
                this.rows[i][key] = e.target.value; 
                this.sync();
                // Show or update dropdown filter when typing
                if (!this.dropdown || this.currentInput !== inp) {
                    this.currentInput = inp;
                    this.showDropdown(e.target, (val) => {
                        this.rows[i][key] = val;
                        inp.value = val;
                        this.sync();
                    });
                } else {
                    this.updateDropdownFilter(inp);
                }
            };
            
            inp.onclick = (e) => {
                e.stopPropagation();
                this.currentInput = inp;
                this.showDropdown(e.target, (val) => {
                    this.rows[i][key] = val;
                    inp.value = val;
                    this.sync();
                });
            };
            
            inp.onfocus = (e) => {
                if (!this.dropdown || this.currentInput !== inp) {
                    this.currentInput = inp;
                    this.showDropdown(e.target, (val) => {
                        this.rows[i][key] = val;
                        inp.value = val;
                        this.sync();
                    });
                }
            };
        });

        [1, 3].forEach(idx => {
            const key = idx === 1 ? 'hstr' : 'lstr';
            inputs[idx].onchange = (e) => { 
                this.rows[i][key] = parseFloat(e.target.value) || 0; 
                this.sync(); 
            };
        });

        const btns = d.querySelectorAll("button");
        btns[0].onclick = () => {
            this.rows[i].on = !this.rows[i].on;
            this.render();
            this.sync();
        };
        btns[1].onclick = () => {
            this.rows.splice(i, 1);
            if(this.rows.length === 0) this.rows.push(this.nr());
            this.render();
            this.sync();
        };

        return d;
    }

    showDropdown(input, callback) {
        if (this.dropdown) this.dropdown.remove();
        this.dropdown = document.createElement("div");
        this.dropdown.className = "wan-drop";
        this.dropdownCallback = callback;
        this.dropdownInput = input;
        
        const filter = input.value.toLowerCase();
        const items = this.loras.filter(l => l.toLowerCase().includes(filter));
        
        this.dropdown.innerHTML = `<div data-val="">None</div>` + 
            (items.length ? items.map(l => `<div data-val="${l}">${l}</div>`).join("") : `<div style="color:#777">No match</div>`);

        document.body.appendChild(this.dropdown);
        const rect = input.getBoundingClientRect();
        this.dropdown.style.left = rect.left + "px";
        this.dropdown.style.top = (rect.bottom + window.scrollY) + "px";
        this.dropdown.style.width = rect.width + "px";

        this.dropdown.onclick = (e) => {
            if (e.target.hasAttribute("data-val")) {
                callback(e.target.getAttribute("data-val"));
                this.dropdown.remove();
                this.dropdown = null;
                this.currentInput = null;
                this.dropdownInput = null;
            }
        };

        const close = (e) => {
            // Don't close if clicking on the input field that owns this dropdown
            if (this.dropdown && !this.dropdown.contains(e.target) && e.target !== input && e.target !== this.dropdownInput) {
                this.dropdown.remove();
                this.dropdown = null;
                this.currentInput = null;
                this.dropdownInput = null;
                document.removeEventListener("click", close);
            }
        };
        setTimeout(() => document.addEventListener("click", close), 10);
    }

    updateDropdownFilter(input) {
        if (!this.dropdown) return;
        
        const filter = input.value.toLowerCase();
        const items = this.loras.filter(l => l.toLowerCase().includes(filter));
        
        this.dropdown.innerHTML = `<div data-val="">None</div>` + 
            (items.length ? items.map(l => `<div data-val="${l}">${l}</div>`).join("") : `<div style="color:#777">No match</div>`);
        
        // Re-attach click handler
        this.dropdown.onclick = (e) => {
            if (e.target.hasAttribute("data-val")) {
                if (this.dropdownCallback) {
                    this.dropdownCallback(e.target.getAttribute("data-val"));
                }
                this.dropdown.remove();
                this.dropdown = null;
                this.currentInput = null;
            }
        };
    }

    getWidgetValue(name, def) {
        const w = this.node.widgets?.find(x => x.name === name);
        return w ? w.value : def;
    }

    setWidgetValue(name, val) {
        const w = this.node.widgets?.find(x => x.name === name);
        if (w) {
            w.value = val;
            if (this.node.graph) {
                this.node.graph.setDirtyCanvas(true, true);
            }
        }
    }

    sync() {
        const str = this.rows.map(r => 
            `${r.high||"none"},${r.hstr},${r.low||"none"},${r.lstr},${r.on}`
        ).join("|") + "§true";
        
        this.setWidgetValue("lora_batch_config", str);
    }

    cfg() {
        return {
            rows: JSON.parse(JSON.stringify(this.rows)),
            merge: this.getWidgetValue("merge_loras", true),
            lowmem: this.getWidgetValue("low_mem_load", false)
        };
    }

    apply(c) {
        if (!c) return;
        this.rows = c.rows || [this.nr()];
        this.setWidgetValue("merge_loras", c.merge ?? true);
        this.setWidgetValue("low_mem_load", c.lowmem ?? false);
        this.render();
        this.sync();
    }
    
    rebuild() {
        this.parse();
        this.render();
    }
}
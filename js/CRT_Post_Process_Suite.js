import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("Loading CRT_Post_Process_Suite.js");

const CSS_STYLES = `
@font-face {
    font-family: 'Orbitron';
    src: url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&display=swap') format('woff2'),
         local('Orbitron');
}

/* Define key colors from Flux for consistency, prefixed with pps- */
:root {
    --pps-primary-accent: #7700ff;
    --pps-primary-accent-light: #9a70ff;
    --pps-active-green: #2ecc71;
    /* Other colors like text-white, background-black are common and can be used directly or redefined if needed */
}

.postprocess-container {
    background: #000000;
    border-radius: 12px;
	position: relative;
    width: 100%;
	top: -22px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    border: 0px;
    box-sizing: border-box;
    min-height: 220px;
    overflow: hidden; /* Prevent any overflow */
}

/* Prevent horizontal scroll and stabilize layout */
.postprocess-container,
.postprocess-tabs,
.parameters-grid {
    overflow-x: hidden !important;
}

.postprocess-tabs {
    overflow-x: hidden !important;
}

.parameter-control {
    max-width: 100%;
}

/* .parameter-slider max-width is applied inline in its creation or via other rules */

@keyframes pps-breathe-title { /* Renamed to avoid conflict if 'breathe' is global */
    0%, 100% { 
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8), 0 0 25px var(--pps-primary-accent), 0 0 5px var(--pps-primary-accent-light); 
        transform: scale(1); 
        opacity: 1; 
    }
    50% { 
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8), 0 0 35px var(--pps-primary-accent-light), 0 0 10px var(--pps-primary-accent); 
        transform: scale(0.97); 
        color: var(--pps-primary-accent-light); 
    }
}

.postprocess-title {
    color: var(--pps-primary-accent); /* Flux style */
    font-family: 'Orbitron', sans-serif; /* Flux font */
    font-size: 20px; /* Adjusted from original 25px for balance with Orbitron */
    font-weight: 700; /* Flux style */
    text-align: center;
    margin-bottom: 15px;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8), 0 0 25px var(--pps-primary-accent), 0 0 5px var(--pps-primary-accent-light); /* Flux style */
    animation: pps-breathe-title 4s ease-in-out infinite; /* Use renamed animation */
    user-select: none; /* Flux style */
}

.preset-controls {
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-bottom: 15px;
    align-items: center;
}

.preset-select {
    background: #000000; /* Original Style */
    border: 0px; /* Original Style, consistent with Flux */
    color: #ffffff;
    padding: 4px 8px;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.preset-select:focus {
    outline: none;
    border-color: #000000; /* Original Style */
}

.preset-input {
    background: #000000; /* Original Style */
    border: 0px; /* Original Style, consistent with Flux */
    border-radius: 6px;
    color: #ffffff;
    padding: 4px 8px;
    font-size: 11px;
    width: 120px;
    transition: all 0.3s ease;
}

.preset-input:focus {
    outline: none;
    border-color: #4a90e2; /* Original Style */
    box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2); /* Original Style */
}

.preset-button {
    background: #000000; /* Original Style */
    color: white;
    border: none; /* Original Style, consistent with Flux */
    padding: 6px 12px;
    border-radius: 15px;
    cursor: pointer;
    font-size: 11px;
    font-weight: 500;
    transition: transform 0.2s ease-out;
    position: relative;
    overflow: hidden;
}

.preset-button.load-button,
.preset-button.save-button, /* Note: Original save-button was transparent, might need adjustment if specific look is desired */
.preset-button.delete-button {
    background: transparent; /* Original Style */
    border: 0px; /* Original Style, consistent with Flux */
}

.preset-button:hover {
    transform: scale(1.1); /* Original Style */
}

.postprocess-tabs {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 15px;
    border-bottom: 0px; /* Original Style */
    padding: 10px 0;
    justify-content: center;
    overflow-x: auto; 
}

.postprocess-tab {
    background: #000000; /* Original Style */
    color: #cccccc;
    border: none; /* Original Style, consistent with Flux */
    padding: 6px 12px;
    border-radius: 20px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 500;
    transition: all 0.3s ease, transform 0.2s ease-out;
    border: 1px solid transparent; /* Original Style */
    white-space: nowrap;
    position: relative;
    overflow: hidden;
    flex: 0 0 auto; 
}

@keyframes tab-glow { /* Original Style */
    0%, 100% { box-shadow: 0 0 0 0 rgba(119, 0, 255, 0.4); }
    50% { box-shadow: 0 0 10px 5px rgba(119, 0, 255, 0.4); }
}

.postprocess-tab:hover {
    background: linear-gradient(45deg, #000000 0%, #7700ff 100%); /* Original Style */
    color: #000000; /* Original Style text color on hover */
    transform: translateY(-2px) scale(1.05);
    animation: tab-glow 1.5s infinite;
}

.postprocess-tab.active {
    background: linear-gradient(45deg, #000000 0%, #7700ff 100%); /* Original Style */
    color: #7700ff; /* Original Style text color on active */
    transform: translateY(-2px) scale(1.1);
    animation: tab-glow 1.5s infinite;
}

.postprocess-tab.enabled {
    background: linear-gradient(45deg, #000000 0%, #2ecc71 100%); /* Original Style */
    color: #ffffff;
    border-color: rgba(255, 255, 255, 0.2); /* Original Style */
    box-shadow: 0 6px 18px rgba(46, 204, 113, 0.4); /* Original Style */
}

.postprocess-tab.disabled {
    background: linear-gradient(45deg, #000000 0%, #000000 100%); /* Original Style */
    color: #ffffff;
    border-color: rgba(255, 255, 255, 0.2); /* Original Style */
    box-shadow: 0 6px 18px rgba(119, 0, 255, 0.4); /* Original Style */
}

.postprocess-section {
    display: none;
    opacity: 0;
    transform: translateY(20px);
    transition: opacity 0.4s ease-out, transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.postprocess-section.active {
    display: block;
    opacity: 1;
    transform: translateY(0);
    animation: bounceIn 0.5s ease-out;
}

@keyframes bounceIn { /* Original Style */
    0% { transform: translateY(20px); opacity: 0; }
    60% { transform: translateY(-5px); opacity: 1; }
    100% { transform: translateY(0); }
}

.section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
    padding: 10px;
    background: #000000; /* Original Style */
    border-radius: 0px; /* Original Style */
    border-left: 0; /* Original Style */
    cursor: pointer;
    transition: background 0.3s ease;
}


.section-title {
    color: #ffffff;
    font-size: 16px;
    font-weight: 600;
    margin: 0;
}

.section-toggle {
    background: linear-gradient(45deg, #000000 0%, #2ecc71 100%); /* Original Style */
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 15px;
    cursor: pointer;
    font-size: 11px;
    font-weight: 500;
    transition: all 0.3s ease, transform 0.2s ease-out;
    min-width: 50px;
    position: relative;
    overflow: hidden;
}

.section-toggle:hover {
    transform: scale(1.1); /* Original Style */
    box-shadow: 0 6px 18px rgba(46, 204, 113, 0.6); /* Original Style */
    animation: toggle-glow 1.5s infinite; /* Original Style */
}

@keyframes toggle-glow { /* Original Style */
    0%, 100% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.6); }
    50% { box-shadow: 0 0 10px 3px rgba(46, 204, 113, 0.6); }
}

.section-toggle.disabled {
    background: linear-gradient(45deg, #000000 0%, #000000 100%); /* Original Style */
}

.section-toggle.disabled:hover {
    box-shadow: none; /* Original Style */
    transform: none; /* Original Style */
}

.reset-button {
    background: linear-gradient(45deg, #000000 0%, #d11d0a 100%); /* Original Style */
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 15px;
    cursor: pointer;
    font-size: 11px;
    font-weight: 500;
    transition: all 0.3s ease, transform 0.2s ease-out;
    margin-left: 8px;
    position: relative;
    overflow: hidden;
}

.reset-button:hover {
    transform: scale(1.1); /* Original Style */
    box-shadow: 0 6px 18px rgba(209, 29, 10, 0.6); /* Original Style */
    animation: reset-glow 1.5s infinite; /* Original Style */
}

@keyframes reset-glow { /* Original Style */
    0%, 100% { box-shadow: 0 0 0 0 rgba(209, 29, 10, 0.6); }
    50% { box-shadow: 0 0 10px 3px rgba(209, 29, 10, 0.6); }
}

.collapse-button {
    background: transparent;
    border: none;
    color: #ffffff;
    cursor: pointer;
    font-size: 12px;
    margin-right: 8px;
    transition: color 0.3s ease;
}

.collapse-button:hover {
    color: #7700ff; /* Original Style */
}

.parameters-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 10px;
    margin-top: 10px;
    overflow-y: hidden; 
}

.parameter-group {
    background: #000000; /* Original Style */
    padding: 10px;
    border-radius: 8px;
    border: 0px; /* Original Style */
    width: 100%;
    animation: fadeIn 0.5s ease-out;
}

@keyframes fadeIn { /* Original Style */
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}

.parameter-group-header {
    color: #ffffff;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 8px;
    padding-left: 4px;
    border-left: 2px solid #7700ff; /* Original Style */
}

.parameter-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    padding: 5px 0;
    width: 100%;
    opacity: 0;
    animation: rowFadeIn 0.3s ease-out forwards;
    animation-delay: calc(var(--order) * 0.1s);
}

.parameter-row.modified .parameter-label {
    color: #2ecc71; /* Original Style */
    border-left: 2px solid #2ecc71; /* Original Style */
    padding-left: 4px; /* Original Style */
}

@keyframes rowFadeIn { /* Original Style */
    0% { opacity: 0; transform: translateX(20px); }
    100% { opacity: 1; transform: translateX(0); }
}

.parameter-label {
    color: #cccccc;
    font-size: 12px;
    font-weight: 500;
    min-width: 100px;
    text-transform: capitalize;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.parameter-control {
    display: flex;
    align-items: center;
    gap: 5px;
    flex: 1;
    justify-content: flex-end;
    overflow: hidden;
}

.parameter-input {
    background: #000000; /* Original Style */
    border: 0px; /* Original Style */
    border-radius: 6px;
    color: #ffffff;
    padding: 4px;
    font-size: 11px;
    width: 60px;
    text-align: center;
    transition: all 0.3s ease;
}

.parameter-input:focus {
    outline: none;
    border-color: #4a90e2; /* Original Style */
    box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2); /* Original Style */
}

/* === FANCY SLIDER STYLES START === */
.parameter-slider {
    -webkit-appearance: none;
    appearance: none;
    flex: 1;
    max-width: 500px; /* Or original max-width: 95%; */
    margin: 0 5px;
    height: 4px; 
    background: linear-gradient(90deg, rgba(119, 0, 255, 0.3), rgba(119, 0, 255, 0.1));
    border-radius: 2px;
    box-shadow: 0 0 8px var(--pps-primary-accent);
    outline: none;
    transition: box-shadow 0.2s ease-in-out, background 0.3s ease;
}

.parameter-slider:hover {
    background: linear-gradient(90deg, rgba(119, 0, 255, 0.5), rgba(119, 0, 255, 0.2));
    box-shadow: 0 0 12px var(--pps-primary-accent-light);
}

/* Style for modified slider track using existing .parameter-row.modified class */
.parameter-row.modified .parameter-slider {
    background: linear-gradient(90deg, var(--pps-active-green), rgba(46, 204, 113, 0.5));
    box-shadow: 0 0 12px var(--pps-active-green);
}

.parameter-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 12px;
    height: 12px;
    background: var(--pps-primary-accent);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--pps-primary-accent);
    cursor: grab;
    transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.3s ease;
}
.parameter-slider::-moz-range-thumb { /* Firefox */
    width: 12px;
    height: 12px;
    background: var(--pps-primary-accent);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--pps-primary-accent);
    cursor: grab;
    border: 0px; /* Important for Firefox */
}

.parameter-slider:active::-webkit-slider-thumb {
    cursor: grabbing;
}
.parameter-slider:active::-moz-range-thumb {
    cursor: grabbing;
}

.parameter-slider::-webkit-slider-thumb:hover {
    transform: scale(1.2);
    box-shadow: 0 0 12px var(--pps-primary-accent-light);
}
.parameter-slider::-moz-range-thumb:hover {
    transform: scale(1.2);
    box-shadow: 0 0 12px var(--pps-primary-accent-light);
}

/* Style for modified slider thumb */
.parameter-row.modified .parameter-slider::-webkit-slider-thumb {
    background: var(--pps-active-green);
    box-shadow: 0 0 8px var(--pps-active-green);
}
.parameter-row.modified .parameter-slider::-moz-range-thumb {
    background: var(--pps-active-green);
    box-shadow: 0 0 8px var(--pps-active-green);
}

.parameter-row.modified .parameter-slider::-webkit-slider-thumb:hover {
    box-shadow: 0 0 12px var(--pps-active-green);
}
.parameter-row.modified .parameter-slider::-moz-range-thumb:hover {
    box-shadow: 0 0 12px var(--pps-active-green);
}
/* === FANCY SLIDER STYLES END === */

.dropdown-select {
    background: #000000; /* Original Style */
    border: 1px solid rgba(255, 255, 255, 0.2); /* Original Style */
    border-radius: 6px;
    color: #ffffff;
    padding: 4px 8px;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.dropdown-select:focus {
    outline: none;
    border-color: #7700ff; /* Original Style */
    box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2); /* Original Style */
}

.disabled-section {
    opacity: 0.5;
    pointer-events: none;
}
`;

function injectStyles() {
    if (!document.getElementById('postprocess-styles-orbitron-sliders')) { 
        const styleSheet = document.createElement('style');
        styleSheet.id = 'postprocess-styles-orbitron-sliders';
        styleSheet.textContent = CSS_STYLES;
        document.head.appendChild(styleSheet);
        console.log("CSS styles (Orbitron Title & Fancy Sliders) injected successfully");
        
        if (!document.querySelector('link[href*="Orbitron"]')) {
            const fontLink = document.createElement("link");
            fontLink.href = "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&display=swap";
            fontLink.rel = "stylesheet";
            document.head.appendChild(fontLink);
        }

    } else {
        console.log("CSS styles (Orbitron Title & Fancy Sliders) already injected");
    }
}

const OPERATION_GROUPS = {
    "UPSCALE": {
        icon: "🍆",
        params: ["upscale_model_path", "downscale_by", "rescale_method", "precision"],
        enable: "enable_upscale"
    },
    "LEVELS": {
        icon: "🔧",
        params: ["exposure", "gamma", "brightness", "contrast", "saturation", "vibrance"],
        enable: "enable_levels"
    },
    "COLOR WHEELS": {
        icon: "🎨",
        subgroups: {
            "Lift": ["lift_r", "lift_g", "lift_b"],
            "Gamma": ["gamma_r", "gamma_g", "gamma_b"],
            "Gain": ["gain_r", "gain_g", "gain_b"]
        },
        enable: "enable_color_wheels"
    },
    "TEMPERATURE & TINT": {
        icon: "🌡️",
        params: ["temperature", "tint"],
        enable: "enable_temp_tint"
    },
    "SHARPENING": {
        icon: "💎",
        params: ["sharpen_strength", "sharpen_radius", "sharpen_threshold"],
        enable: "enable_sharpen"
    },
    "GLOWS": {
        icon: "🔅",
        subgroups: {
            "Small Glow": ["small_glow_intensity", "small_glow_radius", "small_glow_threshold"],
            "Large Glow": ["large_glow_intensity", "large_glow_radius", "large_glow_threshold"]
        },
        enable: "enable_small_glow"
    },
    "GLARE/FLARES": {
        icon: "🔆",
        params: ["glare_type", "glare_intensity", "glare_length", "glare_angle", "glare_threshold", "glare_quality"],
        enable: "enable_glare"
    },
    "CHROMATIC ABERRATION": {
        icon: "🌈",
        params: ["ca_strength", "ca_edge_falloff"],
        enable: "enable_chromatic_aberration"
    },
    "VIGNETTE": {
        icon: "🎴",
        params: ["vignette_strength", "vignette_radius", "vignette_softness"],
        enable: "enable_vignette"
    },
    "RADIAL BLUR": {
        icon: "🌀",
        params: ["radial_blur_type", "radial_blur_strength", "radial_blur_center_x", "radial_blur_center_y", "radial_blur_falloff", "radial_blur_samples"],
        enable: "enable_radial_blur"
    },
    "LENS DISTORTION": {
        icon: "📷",
        params: ["barrel_distortion"],
        enable: "enable_lens_distortion"
    },
    "FILM GRAIN": {
        icon: "🎞️",
        params: ["grain_intensity", "grain_size", "grain_color_amount"],
        enable: "enable_film_grain"
    }
};

const DEFAULTS = {
    "downscale_by": 0.5,
    "batch_size": 1,
    "exposure": 0.0,
    "gamma": 1.0,
    "brightness": 0.0,
    "contrast": 1.0,
    "saturation": 1.0,
    "vibrance": 0.0,
    "lift_r": 0.0,
    "lift_g": 0.0,
    "lift_b": 0.0,
    "gamma_r": 1.0,
    "gamma_g": 1.0,
    "gamma_b": 1.0,
    "gain_r": 1.0,
    "gain_g": 1.0,
    "gain_b": 1.0,
    "temperature": 0.0,
    "tint": 0.0,
    "sharpen_strength": 2.5,
    "sharpen_radius": 1.85,
    "sharpen_threshold": 0.015,
    "small_glow_intensity": 0.015,
    "small_glow_radius": 0.1,
    "small_glow_threshold": 0.25,
    "large_glow_intensity": 0.25,
    "large_glow_radius": 50.0,
    "large_glow_threshold": 0.3,
    "glare_intensity": 0.65,
    "glare_length": 1.5,
    "glare_angle": 0.0,
    "glare_threshold": 0.95,
    "glare_quality": 16,
    "ca_strength": 0.005,
    "ca_edge_falloff": 2.0,
    "vignette_strength": 0.5,
    "vignette_radius": 0.7,
    "vignette_softness": 2.0,
    "radial_blur_type": "spin",
    "radial_blur_strength": 0.02,
    "radial_blur_center_x": 0.5,
    "radial_blur_center_y": 0.25,
    "radial_blur_falloff": 0.05,
    "radial_blur_samples": 16,
    "grain_intensity": 0.02,
    "grain_size": 0.03,
    "grain_color_amount": 0.0,
    "barrel_distortion": 0.0
};

const PARAM_LABELS = {
    "upscale_model_path": "Model Path",
    "downscale_by": "Downscale By",
    "rescale_method": "Rescale Method",
    "precision": "Precision",
    "exposure": "Exposure",
    "gamma": "Gamma",
    "brightness": "Brightness",
    "contrast": "Contrast",
    "saturation": "Saturation",
    "vibrance": "Vibrance",
    "lift_r": "Lift Red",
    "lift_g": "Lift Green",
    "lift_b": "Lift Blue",
    "gamma_r": "Gamma Red",
    "gamma_g": "Gamma Green",
    "gamma_b": "Gamma Blue",
    "gain_r": "Gain Red",
    "gain_g": "Gain Green",
    "gain_b": "Gain Blue",
    "temperature": "Temperature",
    "tint": "Tint",
    "sharpen_strength": "Strength",
    "sharpen_radius": "Radius",
    "sharpen_threshold": "Threshold",
    "small_glow_intensity": "Small Intensity",
    "small_glow_radius": "Small Radius",
    "small_glow_threshold": "Small Threshold",
    "large_glow_intensity": "Large Intensity",
    "large_glow_radius": "Large Radius",
    "large_glow_threshold": "Large Threshold",
    "glare_type": "Type",
    "glare_intensity": "Intensity",
    "glare_length": "Length",
    "glare_angle": "Angle",
    "glare_threshold": "Threshold",
    "glare_quality": "Quality",
    "ca_strength": "Strength",
    "ca_edge_falloff": "Edge Falloff",
    "vignette_strength": "Strength",
    "vignette_radius": "Radius",
    "vignette_softness": "Softness",
    "radial_blur_type": "Type",
    "radial_blur_strength": "Strength",
    "radial_blur_center_x": "Center X",
    "radial_blur_center_y": "Center Y",
    "radial_blur_falloff": "Falloff",
    "radial_blur_samples": "Samples",
    "grain_intensity": "Intensity",
    "grain_size": "Size",
    "grain_color_amount": "Color Amount",
    "barrel_distortion": "Barrel"
};

class ProfessionalPostProcessUI {
    constructor(node) {
        this.node = node;
        this.activeTab = "UPSCALE";
        this.container = null;
        this.enabledSections = new Set();
        this.presetSelect = null;
        this.presets = this.loadPresets();
        console.log("ProfessionalPostProcessUI instance created for node:", this.node);
        this.setupUI();
    }

    loadPresets() {
        const presets = localStorage.getItem('crtPostProcessPresets');
        return presets ? JSON.parse(presets) : {};
    }

    savePresets() {
        localStorage.setItem('crtPostProcessPresets', JSON.stringify(this.presets));
    }

    updatePresetDropdown() {
        if (!this.presetSelect) return;
        this.presetSelect.innerHTML = '';
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'Select Preset';
        this.presetSelect.appendChild(defaultOption);

        Object.keys(this.presets).forEach(presetName => {
            const option = document.createElement('option');
            option.value = presetName;
            option.textContent = presetName;
            this.presetSelect.appendChild(option);
        });
    }

    sanitizeId(groupName) {
        return groupName.replace(/[^a-zA-Z0-9-]/g, '-').replace(/-+/g, '-').toLowerCase();
    }

    setupUI(attempt = 1) {
        console.log(`Attempting UI setup (attempt ${attempt})...`);
        if (!this.node || !this.node.widgets || this.node.widgets.length === 0) {
            console.warn(`Node or widgets not available, retrying (attempt ${attempt})`);
            if (attempt > 10) {
                console.error("Failed to initialize UI after 10 attempts. Proceeding with defaults.");
                this.initializeUI();
                return;
            }
            setTimeout(() => this.setupUI(attempt + 1), 100);
            return;
        }

        let allWidgetsReady = true;
        Object.keys(OPERATION_GROUPS).forEach(groupName => {
            const group = OPERATION_GROUPS[groupName];
            const params = group.params || Object.values(group.subgroups || {}).flat();
            params.concat(group.enable).forEach(paramName => {
                const widget = this.findWidget(paramName);
                if (!widget || widget.value === undefined) {
                    console.warn(`Widget ${paramName} not ready (value: ${widget?.value})`);
                    allWidgetsReady = false;
                }
            });
        });

        if (allWidgetsReady) {
            console.log("All widgets ready, initializing UI");
            this.initializeUI();
        } else {
            console.warn(`Widgets not fully loaded, retrying (attempt ${attempt})`);
            if (attempt > 10) {
                console.error("Failed to load widget values after 10 attempts. Proceeding with defaults.");
                this.initializeUI();
                return;
            }
            setTimeout(() => this.setupUI(attempt + 1), 100);
        }
    }

    initializeUI() {
        console.log("Initializing UI... Initial widget values:", this.getAllWidgetValues());
        this.hideOriginalWidgets();
        this.createCustomUI();
        this.updateEnabledSections();
        this.syncUIWithWidgets();
        Object.keys(OPERATION_GROUPS).forEach(groupName => {
            this.refreshSectionControls(groupName);
        });

        this.node.setSize([650, 220]);
        console.log("Node size set to 650x220");

        setTimeout(() => {
            console.log("Triggering initial tab switch to:", this.activeTab);
            this.syncUIWithWidgets();
            this.switchTab(this.activeTab);
            this.node.setDirtyCanvas(true, true);
            console.log("Canvas marked as dirty for redraw");
        }, 200);
    }

    getAllWidgetValues() {
        const values = {};
        Object.keys(OPERATION_GROUPS).forEach(groupName => {
            const group = OPERATION_GROUPS[groupName];
            const params = group.params || Object.values(group.subgroups || {}).flat();
            params.forEach(paramName => {
                const widget = this.findWidget(paramName);
                if (widget) values[paramName] = widget.value;
            });
            const enableWidget = this.findWidget(group.enable);
            if (enableWidget) values[group.enable] = enableWidget.value;
        });
        return values;
    }

    hideOriginalWidgets() {
        if (!this.node.widgets) {
            console.warn("No widgets to hide; node.widgets is undefined");
            return;
        }
        this.node.widgets.forEach(widget => {
            if (widget.name !== 'image') {
                widget.computeSize = () => [0, -4];
                widget.type = "hidden";
                widget.serialize = true;
                if (widget.element) {
                    widget.element.remove();
                    console.log(`Hiding widget by removal: ${widget.name}`);
                }
            }
        });
        console.log("Original widgets hidden");
    }

    createCustomUI() {
        console.log("Creating custom UI...");
        this.container = document.createElement('div');
        this.container.className = 'postprocess-container';

        const title = document.createElement('div');
        title.className = 'postprocess-title';
        title.textContent = 'CRT Post-Process Suite';
        this.container.appendChild(title);

        this.createPresetControls();
        this.createTabs();
        this.createSections();

        if (this.node.addDOMWidget) {
            try {
                this.node.addDOMWidget('postprocess_ui', 'div', this.container);
                console.log("Custom UI added via addDOMWidget");
            } catch (error) {
                console.error("Failed to add DOM widget:", error);
            }
        } else {
            console.error("addDOMWidget method not available on node");
        }

        this.node.setDirtyCanvas(true, true);
    }

    createPresetControls() {
        const presetControls = document.createElement('div');
        presetControls.className = 'preset-controls';

        this.presetSelect = document.createElement('select');
        this.presetSelect.className = 'preset-select';
        this.updatePresetDropdown();
        presetControls.appendChild(this.presetSelect);

        const loadButton = document.createElement('button');
        loadButton.className = 'preset-button load-button';
        loadButton.textContent = 'Load';
        loadButton.addEventListener('click', () => this.loadPreset());
        presetControls.appendChild(loadButton);

        const presetInput = document.createElement('input');
        presetInput.className = 'preset-input';
        presetInput.type = 'text';
        presetInput.placeholder = 'Preset Name';
        presetControls.appendChild(presetInput);

        const saveButton = document.createElement('button');
        saveButton.className = 'preset-button';
        saveButton.textContent = '💜';
        saveButton.addEventListener('click', () => {
            const presetName = presetInput.value.trim();
            if (presetName) {
                this.savePreset(presetName);
                presetInput.value = '';
            } else {
                console.warn("Please enter a preset name");
            }
        });
        presetControls.appendChild(saveButton);

        // Delete preset button
        const deleteButton = document.createElement('button');
        deleteButton.className = 'preset-button';
        deleteButton.textContent = '♻️';
        deleteButton.addEventListener('click', () => this.deletePreset());
        presetControls.appendChild(deleteButton);

        this.container.appendChild(presetControls);
        console.log("Preset controls created");
    }

    savePreset(presetName) {
        const values = this.getAllWidgetValues();
        this.presets[presetName] = values;
        this.savePresets();
        this.updatePresetDropdown();
        console.log(`Preset "${presetName}" saved`, values);
    }

    loadPreset() {
        const presetName = this.presetSelect.value;
        if (!presetName) {
            console.warn("Please select a preset to load");
            return;
        }

        const preset = this.presets[presetName];
        if (!preset) {
            console.warn(`Preset "${presetName}" not found`);
            return;
        }

        Object.keys(preset).forEach(paramName => {
            const widget = this.findWidget(paramName);
            if (widget) {
                if (paramName === 'upscale_model_path' && widget.options?.values) {
                    if (!widget.options.values.includes(preset[paramName])) {
                        console.warn(`Preset value '${preset[paramName]}' for upscale_model_path is invalid, using first valid option`);
                        widget.value = widget.options.values[0];
                    } else {
                        widget.value = preset[paramName];
                    }
                } else {
                    widget.value = preset[paramName];
                }
            }
        });

        this.syncUIWithWidgets();
        Object.keys(OPERATION_GROUPS).forEach(groupName => {
            this.refreshSectionControls(groupName);
        });
        this.switchTab(this.activeTab);
        this.node.setDirtyCanvas(true, true);
        console.log(`Preset "${presetName}" loaded`, preset);
    }

    deletePreset() {
        const presetName = this.presetSelect.value;
        if (!presetName) {
            console.warn("Please select a preset to delete");
            return;
        }

        delete this.presets[presetName];
        this.savePresets();
        this.updatePresetDropdown();
        console.log(`Preset "${presetName}" deleted`);
    }

    createTabs() {
        const tabsContainer = document.createElement('div');
        tabsContainer.className = 'postprocess-tabs';

        Object.keys(OPERATION_GROUPS).forEach(groupName => {
            const tab = document.createElement('button');
            tab.className = 'postprocess-tab';
            const enableWidget = this.findWidget(OPERATION_GROUPS[groupName].enable);
            let isEnabled = enableWidget ? !!enableWidget.value : false;
            tab.classList.add(isEnabled ? 'enabled' : 'disabled');
            if (groupName === this.activeTab) {
                tab.classList.add('active');
            }
            const group = OPERATION_GROUPS[groupName];
            tab.textContent = `${group.icon} ${groupName}`;
            tab.dataset.groupName = groupName;
            tab.addEventListener('click', () => this.switchTab(groupName));
            tabsContainer.appendChild(tab);
        });

        this.container.appendChild(tabsContainer);
        console.log("Tabs created");
    }

    createSections() {
        Object.keys(OPERATION_GROUPS).forEach(groupName => {
            const section = document.createElement('div');
            section.className = 'postprocess-section';
            const sanitizedId = this.sanitizeId(groupName);
            section.id = `section-${sanitizedId}`;
            if (groupName === this.activeTab) {
                section.classList.add('active');
            }

            const header = document.createElement('div');
            header.className = 'section-header';

            const collapseBtn = document.createElement('button');
            collapseBtn.className = 'collapse-button';
            collapseBtn.textContent = '▼';

            const titleEl = document.createElement('h3');
            titleEl.className = 'section-title';
            titleEl.textContent = `${OPERATION_GROUPS[groupName].icon} ${groupName}`;

            const controls = document.createElement('div');
            controls.style.display = 'flex';
            controls.style.alignItems = 'center';

            const toggle = document.createElement('button');
            toggle.className = 'section-toggle';
            const enableWidget = this.findWidget(OPERATION_GROUPS[groupName].enable);
            let isEnabled = enableWidget ? !!enableWidget.value : false;
            toggle.textContent = isEnabled ? 'ON' : 'OFF';
            toggle.classList.toggle('disabled', !isEnabled);
            toggle.addEventListener('click', () => this.toggleSection(groupName, toggle));

            if (enableWidget) {
                enableWidget.serialize = true;
            }

            const resetBtn = document.createElement('button');
            resetBtn.className = 'reset-button';
            resetBtn.textContent = 'Reset';
            resetBtn.addEventListener('click', () => this.resetSection(groupName));

            controls.appendChild(toggle);
            controls.appendChild(resetBtn);
            header.appendChild(collapseBtn);
            header.appendChild(titleEl);
            header.appendChild(controls);

            const paramsContainer = document.createElement('div');
            paramsContainer.className = 'parameters-grid';
            paramsContainer.classList.toggle('disabled-section', !isEnabled);

            collapseBtn.addEventListener('click', () => {
                paramsContainer.style.display = paramsContainer.style.display === 'none' ? 'block' : 'none';
                collapseBtn.textContent = paramsContainer.style.display === 'none' ? '▶' : '▼';
                this.updateNodeSize();
            });

            if (groupName === this.activeTab) {
                paramsContainer.style.display = 'block';
                collapseBtn.textContent = '▼';
            } else {
                paramsContainer.style.display = 'none';
                collapseBtn.textContent = '▶';
            }

            this.createParameterControls(groupName, paramsContainer);
            section.appendChild(header);
            section.appendChild(paramsContainer);
            this.container.appendChild(section);
        });
        console.log("Sections created");
    }

    createParameterControls(groupName, container) {
        const group = OPERATION_GROUPS[groupName];
        const params = group.params || [];
        const subgroups = group.subgroups || {};

        if (params.length) {
            const paramGroup = document.createElement('div');
            paramGroup.className = 'parameter-group';
            this.createParameterRows(params, paramGroup);
            container.appendChild(paramGroup);
        }

        Object.keys(subgroups).forEach(subgroupName => {
            const paramGroup = document.createElement('div');
            paramGroup.className = 'parameter-group';
            const header = document.createElement('div');
            header.className = 'parameter-group-header';
            header.textContent = subgroupName;
            paramGroup.appendChild(header);
            this.createParameterRows(subgroups[subgroupName], paramGroup);
            container.appendChild(paramGroup);
        });
        console.log(`Parameter controls created for group: ${groupName}`);
    }

    createParameterRows(params, container) {
        params.forEach((paramName, index) => {
            const widget = this.findWidget(paramName);
            if (!widget) {
                console.warn(`Widget not found for parameter: ${paramName}`);
                return;
            }
            widget.serialize = true;

            const row = document.createElement('div');
            row.className = 'parameter-row';
            row.dataset.param = paramName;
            row.style.setProperty('--order', index);

            const label = document.createElement('div');
            label.className = 'parameter-label';
            label.textContent = PARAM_LABELS[paramName] || paramName;

            const control = document.createElement('div');
            control.className = 'parameter-control';

            if (paramName === 'glare_type' || paramName === 'upscale_model_path' || paramName === 'rescale_method' || paramName === 'precision' || paramName === 'radial_blur_type') {
                const select = document.createElement('select');
                select.className = 'dropdown-select';
                let options = [];
                if (paramName === 'glare_type') {
                    options = ["star_4", "star_6", "star_8", "anamorphic_h"];
                } else if (paramName === 'upscale_model_path') {
                    options = widget.options?.values || [];
                    if (options.length === 0) {
                        options = ["None"];
                    }
                } else if (paramName === 'rescale_method') {
                    options = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"];
                } else if (paramName === 'precision') {
                    options = ["auto", "32", "16", "bfloat16"];
                } else if (paramName === 'radial_blur_type') {
                    options = ["zoom", "spin"];
                }

                if (!options.includes(widget.value) && options.length > 0) {
                    console.log(`Setting default value for ${paramName}: ${options[0]}`);
                    widget.value = options[0];
                }

                options.forEach(option => {
                    const optionEl = document.createElement('option');
                    optionEl.value = option;
                    optionEl.textContent = option.replace(/_/g, ' ').toUpperCase();
                    select.appendChild(optionEl);
                });

                select.value = widget.value || options[0];
                select.addEventListener('change', (e) => {
                    widget.value = e.target.value;
                    row.classList.toggle('modified', widget.value !== (DEFAULTS[paramName] || options[0]));
                    this.node.setDirtyCanvas(true, true);
                });

                control.appendChild(select);
            } else {
                const slider = document.createElement('input');
                slider.type = 'range';
                slider.className = 'parameter-slider';
                let min = widget.options?.min ?? -10;
                let max = widget.options?.max ?? 10;
                let step = 0.001;

                if (paramName === 'ca_strength') {
                    min = 0.0;
                    max = 0.1;
                } else if (paramName === 'downscale_by') {
                    min = 0.25;
                    max = 1.0;
                    step = 0.05;
                } else if (paramName === 'glare_intensity') {
                    min = 0.0;
                    max = 100.0;
                } else if (paramName === 'glare_length') {
                    min = 0.1;
                    max = 100.0;
                } else if (paramName === 'glare_angle') {
                    min = -180.0;
                    max = 180.0;
                    step = 1.0;
                } else if (paramName === 'glare_threshold') {
                    min = 0.0;
                    max = 1.0;
                } else if (paramName === 'glare_quality') {
                    min = 8;
                    max = 32;
                    step = 1;
                } else if (paramName === 'radial_blur_samples') {
                    min = 8;
                    max = 64;
                    step = 1;
                }

                slider.min = min;
                slider.max = max;
                slider.step = step;
                slider.value = widget.value !== undefined ? parseFloat(widget.value) : (DEFAULTS[paramName] || 0);

                const input = document.createElement('input');
                input.type = 'number';
                input.className = 'parameter-input';
                input.min = min;
                input.max = max;
                input.step = step;
                input.value = widget.value !== undefined ? parseFloat(widget.value) : (DEFAULTS[paramName] || 0);

                const updateValue = (newValue) => {
                    const parsedValue = parseFloat(newValue);
                    if (!isNaN(parsedValue)) {
                        const roundedValue = parseFloat(parsedValue.toFixed(3));
                        widget.value = roundedValue;
                        slider.value = roundedValue;
                        input.value = roundedValue.toFixed(3);
                        row.classList.toggle('modified', roundedValue !== DEFAULTS[paramName]);
                        this.node.setDirtyCanvas(true, true);
                    }
                };

                slider.addEventListener('input', (e) => updateValue(e.target.value));
                input.addEventListener('change', (e) => updateValue(e.target.value));

                control.appendChild(slider);
                control.appendChild(input);
            }

            row.appendChild(label);
            row.appendChild(control);
            container.appendChild(row);
            row.classList.toggle('modified', widget.value !== (DEFAULTS[paramName] || (widget.options?.values ? widget.options.values[0] : undefined)));
        });
    }

    syncUIWithWidgets() {
        Object.keys(OPERATION_GROUPS).forEach(groupName => {
            const section = this.container.querySelector(`#section-${this.sanitizeId(groupName)}`);
            if (!section) return;

            const paramsContainer = section.querySelector('.parameters-grid');
            if (!paramsContainer) return;

            const group = OPERATION_GROUPS[groupName];
            const params = group.params || Object.values(group.subgroups || {}).flat();
            params.forEach(paramName => {
                const row = paramsContainer.querySelector(`.parameter-row[data-param="${paramName}"]`);
                if (row) {
                    const widget = this.findWidget(paramName);
                    if (widget && widget.value !== undefined) {
                        console.log(`Syncing ${paramName}: widget.value=${widget.value}, default=${DEFAULTS[paramName]}`);
                        const slider = row.querySelector('.parameter-slider');
                        const input = row.querySelector('.parameter-input');
                        const select = row.querySelector('.dropdown-select');
                        if (slider && input) {
                            const value = parseFloat(widget.value);
                            if (!isNaN(value)) {
                                slider.value = value;
                                input.value = value.toFixed(3);
                                row.classList.toggle('modified', value !== DEFAULTS[paramName]);
                            } else {
                                console.warn(`Invalid numeric value for ${paramName}: ${widget.value}`);
                            }
                        } else if (select) {
                            if (widget.options?.values && !widget.options.values.includes(widget.value)) {
                                console.warn(`Invalid dropdown value for ${paramName}: ${widget.value}, setting to ${widget.options.values[0]}`);
                                widget.value = widget.options.values[0];
                            }
                            select.value = widget.value;
                            row.classList.toggle('modified', widget.value !== (DEFAULTS[paramName] || (widget.options?.values ? widget.options.values[0] : undefined)));
                        }
                    } else {
                        console.warn(`Widget ${paramName} has undefined value`);
                    }
                }
            });

            const enableWidget = this.findWidget(group.enable);
            if (enableWidget) {
                const toggleButton = section.querySelector('.section-toggle');
                let isEnabled = !!enableWidget.value;
                if (toggleButton) {
                    toggleButton.textContent = isEnabled ? 'ON' : 'OFF';
                    toggleButton.classList.toggle('disabled', !isEnabled);
                }
                paramsContainer.classList.toggle('disabled-section', !isEnabled);
                if (isEnabled) {
                    this.enabledSections.add(groupName);
                } else {
                    this.enabledSections.delete(groupName);
                }

                const tab = this.container.querySelector(`.postprocess-tab[data-group-name="${groupName}"]`);
                if (tab) {
                    tab.classList.remove('enabled', 'disabled');
                    tab.classList.add(isEnabled ? 'enabled' : 'disabled');
                }
            }
        });
        console.log("UI synced with widgets, enabled sections:", Array.from(this.enabledSections));
    }

    switchTab(tabName) {
        this.activeTab = tabName;

        this.container.querySelectorAll('.postprocess-tab').forEach(tab => {
            const groupName = tab.dataset.groupName;
            if (groupName === tabName) {
                tab.classList.add('active');
            } else {
                tab.classList.remove('active');
            }
            tab.classList.remove('enabled', 'disabled');
            tab.classList.add(this.enabledSections.has(groupName) ? 'enabled' : 'disabled');
        });

        this.container.querySelectorAll('.postprocess-section').forEach(section => {
            section.classList.remove('active');
        });

        const sanitizedId = this.sanitizeId(tabName);
        const activeSection = this.container.querySelector(`#section-${sanitizedId}`);
        if (activeSection) {
            activeSection.classList.add('active');
            const paramsContainer = activeSection.querySelector('.parameters-grid');
            if (paramsContainer) {
                paramsContainer.style.display = 'block';
                const collapseBtn = activeSection.querySelector('.collapse-button');
                if (collapseBtn) collapseBtn.textContent = '▼';
            }
            this.updateNodeSize();
        } else {
            console.warn(`Section not found for tab: ${tabName}, ID: section-${sanitizedId}`);
        }
    }

    updateNodeSize() {
        const activeSection = this.container.querySelector('.postprocess-section.active');
        if (activeSection) {
            setTimeout(() => {
                const paramsContainer = activeSection.querySelector('.parameters-grid');
                const isCollapsed = paramsContainer && paramsContainer.style.display === 'none';
                const paramsHeight = isCollapsed ? 0 : (paramsContainer?.scrollHeight || 100);
                const headerHeight = activeSection.querySelector('.section-header')?.offsetHeight || 50;
                const tabsHeight = this.container.querySelector('.postprocess-tabs')?.offsetHeight || 50;
                const titleHeight = this.container.querySelector('.postprocess-title')?.offsetHeight || 40;
                const presetControlsHeight = this.container.querySelector('.preset-controls')?.offsetHeight || 40;
                const padding = 110;
                const newHeight = paramsHeight + headerHeight + tabsHeight + titleHeight + presetControlsHeight + padding;

                console.log(`updateNodeSize: paramsHeight=${paramsHeight}, headerHeight=${headerHeight}, tabsHeight=${tabsHeight}, titleHeight=${titleHeight}, presetControlsHeight=${presetControlsHeight}, newHeight=${newHeight}`);

                this.node.setSize([650, Math.max(220, newHeight)]);
                this.node.setDirtyCanvas(true, true);
            }, 50);
        } else {
            console.warn("No active section found during updateNodeSize");
        }
    }

	toggleSection(groupName, toggleButton) {
		const enableWidget = this.findWidget(OPERATION_GROUPS[groupName].enable);
		if (!enableWidget) {
			console.warn(`Primary enable widget not found for group: ${groupName}`);
			return;
		}

		const isEnabled = !enableWidget.value;
		enableWidget.value = isEnabled;
		toggleButton.textContent = isEnabled ? 'ON' : 'OFF';
		toggleButton.classList.toggle('disabled', !isEnabled);

		if (groupName === "GLOWS") {
			const enableLargeGlowWidget = this.findWidget("enable_large_glow");
			if (enableLargeGlowWidget) {
				enableLargeGlowWidget.value = isEnabled;
				enableLargeGlowWidget.serialize = true; // ensure this is serialized
				console.log(`Set enable_large_glow to ${isEnabled} for GLOWS group`);
			} else {
				console.error("enable_large_glow widget not found; large glow will not be enabled/disabled");
			}
			console.log(`GLOWS group toggled: enable_small_glow=${enableWidget.value}, enable_large_glow=${enableLargeGlowWidget?.value}`);
		}

		const sanitizedId = this.sanitizeId(groupName);
		const section = this.container.querySelector(`#section-${sanitizedId}`);
		const paramsContainer = section?.querySelector('.parameters-grid');
		if (paramsContainer) {
			paramsContainer.classList.toggle('disabled-section', !isEnabled);
			if (isEnabled) {
				this.enabledSections.add(groupName);
			} else {
				this.enabledSections.delete(groupName);
			}
		}

		this.container.querySelectorAll('.postprocess-tab').forEach(tab => {
			if (tab.dataset.groupName === groupName) {
				tab.classList.remove('enabled', 'disabled');
				tab.classList.add(isEnabled ? 'enabled' : 'disabled');
			}
		});

		this.node.setDirtyCanvas(true, true);
		if (groupName === this.activeTab) {
			this.switchTab(groupName);
		}
	}

    resetSection(groupName) {
        const group = OPERATION_GROUPS[groupName];
        const params = group.params || Object.values(group.subgroups || {}).flat();
        params.forEach(paramName => {
            const widget = this.findWidget(paramName);
            if (widget) {
                let value;
                if (paramName === 'upscale_model_path' && widget.options?.values && widget.options.values.length > 0) {
                    value = widget.options.values[0];
                    console.log(`Resetting ${paramName} to first valid option: ${value}`);
                } else if (DEFAULTS[paramName] !== undefined) {
                    const isNumber = typeof DEFAULTS[paramName] === 'number';
                    value = isNumber ? parseFloat(DEFAULTS[paramName].toFixed(3)) : DEFAULTS[paramName];
                } else {
                    console.warn(`No default value defined for ${paramName}, skipping reset`);
                    return;
                }

                widget.value = value;

                const row = this.container.querySelector(`.parameter-row[data-param="${paramName}"]`);
                if (row) {
                    const slider = row.querySelector('.parameter-slider');
                    const input = row.querySelector('.parameter-input');
                    const select = row.querySelector('.dropdown-select');
                    if (slider) slider.value = value;
                    if (input) input.value = typeof value === 'number' ? value.toFixed(3) : value;
                    if (select) select.value = value;
                    row.classList.remove('modified');
                }
            }
        });
        this.node.setDirtyCanvas(true, true);
        if (groupName === this.activeTab) {
            this.switchTab(groupName);
        }
    }

    refreshSectionControls(groupName, specificParam = null) {
        const sanitizedId = this.sanitizeId(groupName);
        const section = this.container.querySelector(`#section-${sanitizedId}`);
        if (!section) {
            console.warn(`Section not found for group: ${groupName}`);
            return;
        }

        const rows = section.querySelectorAll('.parameter-row');
        rows.forEach(row => {
            const label = row.querySelector('.parameter-label')?.textContent.toLowerCase().replace(/\s+/g, '_');
            const paramName = Object.keys(PARAM_LABELS).find(key =>
                PARAM_LABELS[key].toLowerCase().replace(/\s+/g, '_') === label
            );

            if (paramName && (!specificParam || paramName === specificParam)) {
                const widget = this.findWidget(paramName);
                if (widget && widget.value !== undefined) {
                    const slider = row.querySelector('.parameter-slider');
                    const input = row.querySelector('.parameter-input');
                    const select = row.querySelector('.dropdown-select');

                    if (slider && input) {
                        const value = parseFloat(widget.value);
                        slider.value = value;
                        input.value = value.toFixed(3);
                        row.classList.toggle('modified', value !== DEFAULTS[paramName]);
                    } else if (select) {
                        select.value = widget.value;
                        row.classList.toggle('modified', widget.value !== (DEFAULTS[paramName] || (widget.options?.values ? widget.options.values[0] : undefined)));
                    }
                }
            }
        });
    }

    findWidget(name) {
        const widget = this.node.widgets?.find(w => w.name === name) || null;
        if (!widget) {
            console.warn(`Widget "${name}" not found`);
        }
        return widget;
    }

    updateEnabledSections() {
        this.enabledSections.clear();
        Object.keys(OPERATION_GROUPS).forEach(groupName => {
            const enableWidget = this.findWidget(OPERATION_GROUPS[groupName].enable);
            let isEnabled = enableWidget && !!enableWidget.value;
            if (isEnabled) {
                this.enabledSections.add(groupName);
            }
        });
        console.log("Enabled sections updated:", Array.from(this.enabledSections));
    }
}

app.registerExtension({
    name: "Comfy.ProfessionalPostProcess.UI",
    async nodeCreated(node) {
        if (node.comfyClass === "CRT Post-Process Suite") {
            console.log("CRT Post-Process Suite node created:", node);
            node.color = "#000000";
            node.bgcolor = "#000000";
            if (!node.postProcessUI) {
                node.postProcessUI = new ProfessionalPostProcessUI(node);
            } else {
                console.log("postProcessUI already exists on node");
            }
        }
    },
    async setup() {
        console.log("ProfessionalPostProcess extension setup started");
        injectStyles();
        console.log("ProfessionalPostProcess extension setup completed");
    }
});
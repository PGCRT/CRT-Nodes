import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("FluxAIO_UI.js: Complete Professional UI loading...");

// =================================================================
// START: Integrated FluxKnob Module
// Reusable interactive knob UI component
// =================================================================
function createFluxKnob({
  name, label, min, max, step, precision, default: defaultValue, color = '#7700ff', onChange
}) {
  const container = document.createElement('div');
  container.className = `flux-knob-container-${name}`;
  container.style.display = 'flex';
  container.style.flexDirection = 'column';
  container.style.alignItems = 'center';

  const labelEl = document.createElement('div');
  labelEl.className = 'flux-knob-label';
  labelEl.textContent = label;

  const knob = document.createElement('div');
  knob.className = 'flux-knob';
  knob.tabIndex = 0;
  knob.style.setProperty('--knob-glow-color', color);
  knob.style.setProperty('--knob-color', color);

  const base = document.createElement('div');
  base.className = 'flux-knob-base';

  const glow = document.createElement('div');
  glow.className = 'flux-knob-glow-ring';

  const dial = document.createElement('div');
  dial.className = 'flux-knob-dial';

  const output = document.createElement('output');
  dial.appendChild(output);

  knob.appendChild(base);
  knob.appendChild(glow);
  knob.appendChild(dial);

  let value = defaultValue;

  const updateVisuals = (val) => {
    const pct = ((val - min) / (max - min)) * 100;
    knob.style.setProperty('--knob-value-percent', Math.max(0, Math.min(100, pct)));
    output.textContent = val.toFixed(precision);

    // FIX: Add non-default class for green glow
    const isDefault = Math.abs(val - defaultValue) < (step || 0.01) / 2;
    knob.classList.toggle('non-default', !isDefault);
  };

  const applyValue = (newVal) => {
    newVal = Math.max(min, Math.min(max, Math.round(newVal / step) * step));
    if (newVal.toFixed(precision) !== value.toFixed(precision)) {
      value = newVal;
      updateVisuals(value);
      if (typeof onChange === 'function') onChange(value);
    }
  };

  // Drag handling
  knob.addEventListener('mousedown', (e) => {
    e.preventDefault();
    const startY = e.clientY;
    const startVal = value;
    const sensitivity = (max - min) / 200;

    const onMove = (eMove) => {
      const delta = startY - eMove.clientY;
      applyValue(startVal + delta * sensitivity);
    };

    const onUp = () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  });

  // Wheel handling
  knob.addEventListener('wheel', (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -1 : 1;
    applyValue(value + delta * step);
  });

  updateVisuals(value);

  container.appendChild(labelEl);
  container.appendChild(knob);

  return { container, updateVisuals };
}
// =================================================================
// END: Integrated FluxKnob Module
// =================================================================

const CSS_STYLES_FLUXAIO = `
@font-face {
    font-family: 'Orbitron';
    src: url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&display=swap') format('woff2'),
         local('Orbitron');
}

:root {
    --primary-accent: #7700ff;
    --primary-accent-light: #9a70ff;
    --primary-accent-rgb: 119, 0, 255;
    --active-green: #2ecc71;
    --inactive-red: #d11d0a;
    --inactive-gray: #555555;
    --bright-orange: #ff8c00;
    --text-white: #ffffff;
    --text-gray: #555555;
    --background-black: #000000;
    --translucent-light: rgba(255, 255, 255, 0.1);
    --model-blue: #3498db;
    --performance-cyan: #1abc9c;
    --conditioning-purple: #9b59b6;
    --prompt-green: #27ae60;
    --lora-gold: #f39c12;
    --inference-red: #e74c3c;
    --postprocess-pink: #e91e63;
    --gallery-teal: #26a69a;
    --save-brown: #8d6e63;
    --preset-indigo: #6c5ce7;
}

/* Styles for the live preview in the gallery */
.live-preview-container {
    background: #000000;
    border: 1px solid var(--primary-accent);
    border-radius: 8px;
    padding: 8px;
    margin-bottom: 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    min-height: 300px;
    justify-content: center;
}

.live-preview-title {
    color: var(--primary-accent-light);
    font-family: 'Orbitron', monospace;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
}

.live-preview-image {
    max-width: 100%;
    max-height: 300px;
    border-radius: 4px;
    background-color: #000000;
    object-fit: contain;
    aspect-ratio: 1 / 1; /* Default aspect ratio */
}

.live-preview-container.empty .live-preview-image {
    display: none;
}

.live-preview-container.empty .live-preview-placeholder {
    display: block;
    color: var(--text-gray);
    font-size: 11px;
}

.live-preview-placeholder {
    display: none;
}


@layer properties {
  @property --knob-value-percent {
    syntax: "<number>";
    inherits: true;
    initial-value: 0;
  }
}

/* Make all text non-selectable */
.fluxaio-container * {
    user-select: none !important;
    -webkit-user-select: none !important;
    -moz-user-select: none !important;
    -ms-user-select: none !important;
}
/* Allow text selection on inputs */
.fluxaio-container input, .fluxaio-container textarea, .fluxaio-container select {
    user-select: text !important;
    -webkit-user-select: text !important;
    -moz-user-select: text !important;
    -ms-user-select: text !important;
}

.fluxaio-container.collapsed {
    min-height: 0;
    height: 120px; /* Visual height for collapsed UI */
    padding-bottom: 0;
    overflow: hidden; /* Hide overflowing content */
}

/* Main container */
.fluxaio-node-custom-widget {
    position: relative !important;
    box-sizing: border-box !important;
    width: 100% !important;
    min-height: 50px !important;
    padding: auto;
    margin: auto;
    overflow: visible !important;
    display: block !important;
    visibility: visible !important;
    z-index: 1 !important;
    /* container height offset */
    top: -10px;
}

.fluxaio-container {
    background: #000000;
    border: 2px solid var(--primary-accent);
    border-radius: 16px;
    padding: 16px;
    margin: 0;
    width: 100%;
    max-width: 900px;
    box-sizing: border-box;
    min-height: 50px;
    height: auto;
    box-shadow: 
        0 8px 32px rgba(119, 0, 255, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    display: flex;
    flex-direction: column;
    align-items: stretch;
    position: relative;
    overflow: hidden;
}

/* Enhanced title */
.fluxaio-title {
    position: sticky;
    top: 0;
    z-index: 10;
    color: var(--primary-accent);
    user-select: none;
    font-family: 'Orbitron', monospace;
    font-size: 18px;
    font-weight: 700;
    text-align: center;
    margin: 0 0 12px 0;
    padding: 8px 0 4px 0;
    text-shadow: 
        1px 1px 2px rgba(0, 0, 0, 0.8), 
        0 0 25px var(--primary-accent), 
        0 0 5px var(--primary-accent-light);
    background: #000000;
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

/* Top bar for Presets and Theme */
.fluxaio-top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    margin-bottom: 12px;
    flex-wrap: wrap;
}

/* Presets section improved theming */
.fluxaio-presets {
    display: flex;
    gap: 8px;
    justify-content: flex-start;
    padding: 8px;
    background: rgba(var(--primary-accent-rgb, 119, 0, 255), 0.1);
    border-radius: 8px;
    border: 1px solid rgba(var(--primary-accent-rgb, 119, 0, 255), 0.2);
    align-items: center;
    flex-grow: 1;
    flex-wrap: wrap;
}

.preset-select, .preset-input {
    background: #000000;
    border: 1px solid var(--primary-accent);
    border-radius: 6px;
    color: var(--primary-accent);
    padding: 4px 8px;
    font-size: 11px;
    font-family: 'Orbitron', monospace;
    font-weight: 500;
    transition: all 0.3s ease;
    cursor: pointer;
    height: 28px;
}

.preset-input {
    width: 120px;
}

.preset-button {
    background: #000000 !important;
    border: 1px solid var(--primary-accent);
    border-radius: 6px;
    color: var(--primary-accent);
    padding: 4px 10px;
    font-size: 10px;
    font-family: 'Orbitron', monospace;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    text-transform: uppercase;
    height: 28px;
}

.preset-button:hover, .preset-select:hover, .preset-input:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(var(--primary-accent-rgb, 119, 0, 255), 0.4);
    background: rgba(var(--primary-accent-rgb, 119, 0, 255), 0.1) !important;
    border-color: var(--primary-accent-light);
}

/* Theme color picker */
.theme-picker-container {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
}

.theme-picker-label {
    color: var(--primary-accent);
    font-family: 'Orbitron', monospace;
    font-size: 10px;
    font-weight: 500;
    transition: color 0.3s ease;
}

.theme-color-picker {
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    border: 2px solid var(--primary-accent-light);
    cursor: pointer;
    background-color: var(--primary-accent);
    box-shadow: 0 0 10px var(--primary-accent);
    transition: all 0.3s ease;
}

.theme-color-picker::-webkit-color-swatch-wrapper {
    padding: 0;
}
.theme-color-picker::-webkit-color-swatch {
    border: none;
    border-radius: 50%;
}
.theme-color-picker::-moz-color-swatch {
    border: none;
    border-radius: 50%;
}
.theme-color-picker:hover {
    transform: scale(1.1);
    box-shadow: 0 0 15px var(--primary-accent-light);
}

/* Tabs with reset buttons */
.fluxaio-tabs {
    display: flex;
    flex-wrap: wrap;
    gap: 2px;
    margin-bottom: 8px;
    padding: 4px;
    justify-content: center;
    background: #000000;
    border-radius: 8px;
    border: #000000;
    position: relative;
    z-index: 5;
}

.fluxaio-tab {
    background: #000000;
    color: #cccccc;
    border: 1px solid transparent;
    padding: 4px 8px;
    border-radius: 12px;
    cursor: pointer;
    font-size: 8px;
    font-weight: 600;
    font-family: 'Orbitron', monospace;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    white-space: nowrap;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    position: relative;
    min-width: 35px;
    text-align: center;
    flex: 1;
    max-width: 85px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.tab-reset-btn {
    background: var(--inactive-red);
    color: white;
    border: none;
    border-radius: 50%;
    width: 12px;
    height: 12px;
    font-size: 8px;
    cursor: pointer;
    margin-left: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0;
    transition: opacity 0.3s ease;
}

.fluxaio-tab.active .tab-reset-btn,
.fluxaio-tab:hover .tab-reset-btn {
    opacity: 1;
}

.fluxaio-tab:hover {
    background: linear-gradient(145deg, var(--primary-accent), var(--primary-accent-light));
    color: var(--text-white);
    border-color: var(--primary-accent-light);
    transform: translateY(-1px) scale(1.02);
    box-shadow: 0 4px 12px rgba(var(--primary-accent-rgb), 0.3);
}

.fluxaio-tab.active {
    background: linear-gradient(145deg, var(--primary-accent), var(--primary-accent-light));
    color: var(--text-white);
    border-color: var(--active-green);
    transform: translateY(-1px) scale(1.05);
    box-shadow: 
        0 4px 12px rgba(var(--primary-accent-rgb), 0.4),
        0 0 15px rgba(46, 204, 113, 0.3);
}

/* Content Container - Dynamic height, no scrollbars */
.fluxaio-content-container {
    flex: 1;
    min-height: 200px;
    height: auto;
    position: relative;
    overflow: hidden;
    transition: height 0.4s ease, opacity 0.4s ease;
}

.fluxaio-content-container.collapsed {
    height: 50px;
    opacity: 0;
    min-height: 50px;
    overflow-y: hidden;
}

/* Section styles - Dynamic height for all tabs */
.fluxaio-section {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    /* REMOVED min-height to allow shrinking */
    height: auto;
    padding: 12px;
    background: #000000;
    border-radius: 8px;
    border: #000000;
    opacity: 0;
    visibility: hidden;
    transition: all 0.3s ease;
    overflow-y: auto;
    box-sizing: border-box;
}

.fluxaio-section.visible {
    opacity: 1;
    visibility: visible;
    position: relative;
}

/* Section titles */
.section-title {
    color: var(--primary-accent-light);
    font-family: 'Orbitron', monospace;
    font-size: 14px;
    font-weight: 600;
    text-align: center;
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 1px;
    text-shadow: 0 0 10px var(--primary-accent-light);
}

/* Pass containers for inference tab */
.pass-container {
    border: 2px solid var(--primary-accent);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
    background: rgba(255, 255, 255, 0.02);
}

.pass-title {
    color: var(--primary-accent-light);
    font-family: 'Orbitron', monospace;
    font-size: 12px;
    font-weight: 600;
    text-align: center;
    margin-bottom: 12px;
    text-transform: uppercase;
}

/* Grid layouts */
.control-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}

.control-grid-1 {
    display: grid;
    grid-template-columns: 1fr;
    gap: 12px;
    margin-bottom: 12px;
}

.control-grid-2 {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-bottom: 12px;
}

.control-grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 12px;
}

.control-grid-4 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-bottom: 12px;
}

.control-grid-5 {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin-bottom: 12px;
}

/* Center single items in grids */
.control-grid > *:only-child,
.control-grid-2 > *:last-child:nth-child(odd),
.control-grid-3 > *:last-child:nth-child(3n-1),
.control-grid-4 > *:last-child:nth-child(4n-2),
.control-grid-4 > *:last-child:nth-child(4n-1) {
    grid-column: 1 / -1;
    justify-self: center;
    max-width: 300px;
}

/* Control groups */
.control-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 10px;
    background: #000000;
    border-radius: 8px;
    border: #000000;
    align-items: center;
}

.control-group.center-content {
    align-items: center;
}

.control-group-title {
    color: var(--primary-accent-light);
    font-family: 'Orbitron', monospace;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
    text-align: center;
}

/* Seed controls improvements */
.seed-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 6px;
    background: #000000;
    border-radius: 8px;
    border: #000000;
    flex-wrap: nowrap;
}

.seed-input {
    background: #000000;
    border: 1px solid var(--primary-accent-light);
    border-radius: 6px;
    color: var(--text-white);
    padding: 4px 8px;
    font-size: 11px;
    font-family: 'Orbitron', monospace;
    width: 80px;
    text-align: center;
    height: 26px;
    box-sizing: border-box;
}

.seed-mode-select {
    background: #000000;
    border: 1px solid var(--primary-accent-light);
    border-radius: 6px;
    color: var(--primary-accent-light);
    padding: 4px 6px;
    font-size: 10px;
    font-family: 'Orbitron', monospace;
    cursor: pointer;
    height: 26px;
    min-width: 70px;
    box-sizing: border-box;
}

.seed-randomize-btn {
    background: var(--bright-orange);
    border: none;
    border-radius: 4px;
    color: white;
    padding: 4px 6px;
    font-size: 10px;
    cursor: pointer;
    font-family: 'Orbitron', monospace;
    height: 26px;
    min-width: 26px;
    box-sizing: border-box;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* Dropdown styles */
.param-select {
    background: #000000;
    border: 1px solid var(--primary-accent-light);
    border-radius: 6px;
    color: var(--primary-accent-light);
    padding: 8px 12px;
    font-size: 11px;
    font-family: 'Orbitron', monospace;
    font-weight: 500;
    transition: all 0.3s ease;
    cursor: pointer;
    width: 100%;
    outline: none;
    height: 32px; /* Slimmer dropdown */
}

.param-select:hover {
    border-color: var(--primary-accent);
    color: var(--primary-accent);
    background: #000000;
}

.param-select:focus {
    border-color: var(--active-green);
    color: var(--active-green);
    box-shadow: 0 0 10px rgba(46, 204, 113, 0.3);
}

/* Enhanced slider styles with theme support */
.param-slider-container {
    margin: 4px 0;
    width: 100%;
}

.param-slider-label {
    color: var(--primary-accent-light);
    font-size: 10px;
    font-weight: 600;
    font-family: 'Orbitron', monospace;
    text-align: center;
    margin-bottom: 6px;
    text-transform: uppercase;
    transition: color 0.3s;
}

.param-slider-row {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
}

.param-slider {
    -webkit-appearance: none;
    appearance: none;
    flex: 1;
    height: 6px;
    background: linear-gradient(90deg, 
        var(--primary-accent), 
        rgba(var(--primary-accent-rgb), 0.3));
    border-radius: 3px;
    outline: none;
    transition: all 0.3s ease;
    min-width: 150px;
}

.param-slider.non-default {
    background: linear-gradient(90deg, var(--active-green), rgba(46, 204, 113, 0.5));
}
.param-slider.non-default::-webkit-slider-thumb {
    background: radial-gradient(circle, var(--active-green), #9affc8);
    box-shadow: 0 2px 6px var(--active-green);
}
.param-slider.non-default + .param-value {
    color: var(--active-green);
    border-color: var(--active-green);
    background: rgba(46, 204, 113, 0.1);
}
.param-slider-container:has(.non-default) .param-slider-label {
    color: var(--active-green);
}

.param-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 24px;
    height: 12px;
    background: radial-gradient(circle, var(--primary-accent), var(--primary-accent-light));
    border-radius: 50%;
    border: 2px solid #ffffff;
    cursor: grab;
    transition: all 0.2s ease;
    box-shadow: 0 2px 6px var(--primary-accent);
}

.param-slider::-webkit-slider-thumb:hover {
    transform: scale(1.1);
    box-shadow: 0 4px 12px var(--primary-accent);
}

.param-value {
    color: var(--primary-accent);
    font-size: 10px;
    font-family: 'Orbitron', monospace;
    font-weight: 600;
    min-width: 50px;
    text-align: center;
    background: rgba(var(--primary-accent-rgb), 0.1);
    padding: 4px 8px;
    border-radius: 4px;
    border: 1px solid var(--primary-accent);
    transition: all 0.3s;
}

/* Toggle styles */
.param-toggle-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 8px;
    background: #000000;
    border-radius: 6px;
    border: 1px solid rgba(var(--primary-accent-rgb), 0.2);
    cursor: pointer;
    transition: all 0.3s ease;
    min-height: 16px;
}

.param-toggle-container:hover {
    border-color: var(--primary-accent);
    background: rgba(var(--primary-accent-rgb), 0.1);
}
.param-toggle-container.active {
    border-color: var(--active-green);
    background: rgba(46, 204, 113, 0.1);
}

.param-toggle-label {
    color: var(--primary-accent-light);
    font-size: 10px;
    font-weight: 600;
    font-family: 'Orbitron', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    flex: 1;
    transition: color 0.3s;
}
.param-toggle-container.active .param-toggle-label {
    color: var(--active-green);
}

.param-toggle-led {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background-color: var(--inactive-red);
    box-shadow: 0 0 6px rgba(209, 29, 10, 0.5);
    transition: all 0.3s ease;
    border: 1px solid #ff4444;
    position: relative;
    flex-shrink: 0;
}

.param-toggle-led.active {
    background-color: var(--active-green);
    box-shadow: 0 0 10px 2px rgba(46, 204, 113, 0.7);
    border: 1px solid #9affc8;
}

.param-toggle-led::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.8);
    opacity: 0;
    transition: all 0.3s ease;
}

.param-toggle-led.active::before {
    opacity: 1;
}

.lora-select-list {
    border: 1px solid var(--primary-accent);
    border-radius: 6px;
    background: #000;
    max-height: 150px;
    overflow-y: auto;
    min-height: 100px;
    padding: 4px;
    margin-bottom: 8px;
}

.lora-select-item {
    padding: 6px 8px;
    color: var(--primary-accent-light);
    font-family: 'Orbitron', monospace;
    font-size: 10px;
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.2s ease;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.lora-select-item:hover {
    background: rgba(var(--primary-accent-rgb), 0.2);
    color: var(--text-white);
}

.lora-select-item.selected {
    background: var(--primary-accent);
    color: var(--text-white);
}

.lora-search-input {
    width: 100%;
    background: #000;
    border: 1px solid var(--primary-accent);
    border-radius: 4px;
    color: var(--text-white);
    padding: 6px 8px;
    font-size: 10px;
    font-family: 'Orbitron', monospace;
    margin-bottom: 8px;
}

.lora-search-input:focus {
    outline: none;
    border-color: var(--active-green);
    box-shadow: 0 0 8px rgba(46, 204, 113, 0.3);
}

/* LoRA Grid Styles - Enhanced vertical sliders */
.lora-blocks-container {
    background: rgba(var(--primary-accent-rgb), 0.05);
    border: 1px solid rgba(var(--primary-accent-rgb), 0.2);
    border-radius: 8px;
    padding: 12px;
    margin-top: 12px;
}

.lora-blocks-tabs {
    display: flex;
    gap: 5px;
    margin-bottom: 10px;
    justify-content: center;
}

.lora-blocks-tab {
    background: #000000;
    color: var(--text-gray);
    border: 1px solid var(--primary-accent);
    padding: 4px 12px;
    border-radius: 15px;
    cursor: pointer;
    font-size: 10px;
    font-weight: 500;
    font-family: 'Orbitron', monospace;
    transition: all 0.3s ease;
}

.lora-blocks-tab.active {
    background: var(--primary-accent);
    color: var(--text-white);
}

.lora-blocks-controls {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-bottom: 12px;
    flex-wrap: wrap;
    align-items: center;
}

.lora-blocks-control-group {
    display: flex;
    align-items: center;
    gap: 4px;
    background: rgba(var(--primary-accent-rgb), 0.1);
    border: 1px solid rgba(var(--primary-accent-rgb), 0.3);
    border-radius: 6px;
    padding: 4px 8px;
}
.lora-blocks-control-group > span {
    color: var(--primary-accent);
    font-size: 10px;
    font-family: 'Orbitron', monospace;
}

.lora-blocks-input {
    background: #000000;
    border: 1px solid var(--primary-accent);
    border-radius: 4px;
    color: var(--text-white);
    font-size: 10px;
    font-family: 'Orbitron', monospace;
    width: 50px;
    height: 24px;
    text-align: center;
    padding: 2px;
}

.lora-blocks-button {
    background: var(--primary-accent);
    border: none;
    border-radius: 4px;
    color: var(--text-white);
    font-size: 9px;
    font-family: 'Orbitron', monospace;
    font-weight: 600;
    cursor: pointer;
    padding: 4px 8px;
    height: 24px;
    text-transform: uppercase;
    transition: all 0.3s ease;
}

.lora-blocks-button:hover {
    background: var(--primary-accent-light);
    transform: scale(1.05);
}

.lora-blocks-button.danger {
    background: var(--inactive-red);
}

.lora-blocks-button.danger:hover {
    background: #c0392b;
}

.lora-blocks-grid {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 8px;
    background: #000000;
    border-radius: 8px;
    min-height: 120px;
}

.lora-blocks-row {
    display: flex;
    justify-content: center;
    gap: 4px;
    align-items: flex-end;
}

.lora-block-slider-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 24px;
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    border: 1px solid transparent;
    transition: all 0.2s ease;
}

.lora-block-slider-container.selected {
    border-color: var(--active-green);
    background: rgba(46, 204, 113, 0.1);
}

/* UPDATED LoRA SLIDER STYLE */
.lora-block-slider {
    writing-mode: vertical-lr;
    direction: rtl;
    height: 60px;
    /* Copied from .param-slider */
    width: 6px;
    background: linear-gradient(180deg, var(--primary-accent), rgba(var(--primary-accent-rgb), 0.3));
    border-radius: 3px;
    outline: none;
    transition: all 0.3s ease;
}

.lora-block-slider.non-default {
    background: linear-gradient(180deg, var(--active-green), rgba(46, 204, 113, 0.5));
}
.lora-block-slider.non-default::-webkit-slider-thumb {
    background: radial-gradient(circle, var(--active-green), #9affc8);
    box-shadow: 0 0 6px var(--active-green);
}

.lora-block-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    /* Copied from .param-slider */
    width: 16px;
    height: 16px;
    background: radial-gradient(circle, var(--primary-accent), var(--primary-accent-light));
    border-radius: 50%;
    border: 2px solid #ffffff;
    cursor: grab;
    transition: all 0.2s ease;
    box-shadow: 0 2px 6px var(--primary-accent);
}

.lora-block-slider::-webkit-slider-thumb:hover {
    transform: scale(1.1);
    box-shadow: 0 4px 12px var(--primary-accent);
}
/* END UPDATED LoRA SLIDER STYLE */


.lora-block-label {
    color: var(--primary-accent);
    font-size: 8px;
    font-weight: 500;
    text-align: center;
    font-family: 'Orbitron', monospace;
    margin-top: 4px;
    text-shadow: 0 0 3px var(--primary-accent);
    transition: all 0.2s ease;
    user-select: none;
    min-width: 12px;
}

.lora-block-label.active-label {
    color: var(--active-green);
    text-shadow: 0 0 3px var(--active-green);
}

/* LoRA Stack Styles */
.lora-stack-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px;
    background: rgba(var(--primary-accent-rgb), 0.1);
    border: 1px solid rgba(var(--primary-accent-rgb), 0.3);
    border-radius: 6px;
    margin-bottom: 4px;
    transition: all 0.3s ease;
}

.lora-stack-item:hover {
    background: rgba(var(--primary-accent-rgb), 0.15);
    border-color: rgba(var(--primary-accent-rgb), 0.5);
}

.lora-stack-item input[type="checkbox"] {
    width: 16px;
    height: 16px;
    accent-color: var(--primary-accent);
}

.lora-stack-item input[type="number"] {
    width: 60px;
    background: #000;
    border: 1px solid var(--primary-accent);
    border-radius: 4px;
    color: var(--text-white);
    font-size: 10px;
    padding: 2px 4px;
    font-family: 'Orbitron', monospace;
}

.lora-stack-item button {
    background: var(--inactive-red);
    border: none;
    border-radius: 3px;
    color: white;
    cursor: pointer;
    font-size: 12px;
    width: 20px;
    height: 20px;
    transition: all 0.2s ease;
}

.lora-stack-item button:hover {
    background: #e74c3c;
    transform: scale(1.1);
}

.lora-slot-container:hover {
    background: rgba(var(--primary-accent-rgb), 0.15) !important;
    border-color: rgba(var(--primary-accent-rgb), 0.5) !important;
    transform: translateY(-1px);
}

.lora-slot-container input[type="number"] {
    background: #000;
    border: 1px solid var(--primary-accent);
    border-radius: 4px;
    color: var(--text-white);
    font-size: 10px;
    padding: 4px 6px;
    font-family: 'Orbitron', monospace;
    text-align: center;
    width: 60px;
}

.lora-slot-container input[type="number"]:focus {
    border-color: var(--active-green);
    box-shadow: 0 0 8px rgba(46, 204, 113, 0.3);
}

/* Text input styles */
.param-text-input {
    background: #000000;
    border: 1px solid var(--primary-accent-light);
    border-radius: 6px;
    color: var(--text-white);
    padding: 8px 12px;
    font-size: 11px;
    font-family: 'Orbitron', monospace;
    width: 100%;
    outline: none;
    transition: all 0.3s ease;
    resize: none;
    min-height: 80px;
}

.param-text-input:focus {
    border-color: var(--active-green);
    box-shadow: 0 0 10px rgba(46, 204, 113, 0.3);
}

.param-text-input::placeholder {
    color: var(--text-gray);
    font-style: italic;
}

/* Post-process section styles */
.postprocess-tabs {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 12px;
    padding: 8px 0;
    justify-content: center;
    border-bottom: 1px solid rgba(var(--primary-accent-rgb), 0.2);
}

.postprocess-tab {
    background: #000000;
    color: #cccccc;
    border: 1px solid transparent;
    padding: 4px 8px;
    border-radius: 12px;
    cursor: pointer;
    font-size: 9px;
    font-weight: 600;
    font-family: 'Orbitron', monospace;
    transition: all 0.3s ease;
    white-space: nowrap;
    text-transform: uppercase;
}

.postprocess-tab:hover {
    background: linear-gradient(145deg, var(--postprocess-pink), #f06292);
    color: var(--text-white);
    transform: translateY(-1px) scale(1.02);
}

.postprocess-tab.active {
    background: linear-gradient(145deg, var(--primary-accent), var(--primary-accent-light));
    color: var(--text-white);
    transform: translateY(-1px) scale(1.05);
    box-shadow: 0 4px 12px rgba(var(--primary-accent-rgb), 0.4);
}

.postprocess-tab.enabled {
    background: linear-gradient(145deg, var(--active-green), #27ae60);
    color: var(--text-white);
}

.postprocess-section {
    display: none;
    opacity: 0;
    transform: translateY(10px);
    transition: all 0.3s ease;
}

.postprocess-section.active {
    display: block;
    opacity: 1;
    transform: translateY(0);
}

.section-header {
    display: flex;
    justify-content: center; /* Centered content */
    align-items: center;
    margin-bottom: 12px;
    padding: 8px;
    background: linear-gradient(145deg, var(--inactive-gray), #333);
    border-radius: 12px;
    border: 1px solid rgba(var(--primary-accent-rgb), 0.2);
    min-height: 32px;
    cursor: pointer; /* Make the whole bar clickable */
    transition: all 0.3s ease;
    user-select: none;
}

.section-header:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.section-header.enabled {
    background: linear-gradient(145deg, var(--active-green), #27ae60);
}

.section-header.enabled:hover {
    box-shadow: 0 4px 12px rgba(46, 204, 113, 0.4);
}

/* New style for the title text inside the header */
.section-header-title {
    color: var(--text-white);
    font-family: 'Orbitron', monospace;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.section-toggle {
    background: linear-gradient(145deg, var(--inactive-gray), #333);
    color: #aaa;
    border: none;
    padding: 4px 8px;
    border-radius: 12px;
    cursor: pointer;
    font-size: 9px;
    font-weight: 600;
    font-family: 'Orbitron', monospace;
    transition: all 0.3s ease;
    min-width: 40px;
    text-transform: uppercase;
    height: 26px;
}

.section-toggle.enabled {
    background: linear-gradient(145deg, var(--active-green), #27ae60);
    color: white;
}

.section-toggle:hover {
    transform: scale(1.05);
}
.section-toggle.enabled:hover {
    box-shadow: 0 4px 12px rgba(46, 204, 113, 0.3);
}


.reset-button, .gallery-btn, .export-btn {
    background: linear-gradient(145deg, var(--primary-accent), #333);
    border: 1px solid var(--primary-accent);
    border-radius: 6px;
    color: var(--primary-accent);
    padding: 4px 10px;
    font-size: 10px;
    font-family: 'Orbitron', monospace;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    text-transform: uppercase;
    height: 26px;
}

.reset-button:hover, .gallery-btn:hover, .export-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(var(--primary-accent-rgb), 0.4);
    background: rgba(var(--primary-accent-rgb), 0.2);
    border-color: var(--primary-accent-light);
    color: var(--primary-accent-light);
}

.reset-button {
    background: linear-gradient(145deg, var(--inactive-red), #c0392b);
    color: white;
    border: none;
}


/* Gallery Styles */
.gallery-container {
    background: rgba(255, 255, 255, 0.02);
    border: #000000;
    border-radius: 8px;
    padding: 12px;
    min-height: 200px;
}

.gallery-controls {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-bottom: 12px;
    flex-wrap: wrap;
}

.gallery-btn.active {
    background: linear-gradient(145deg, var(--active-green), #1e8449);
    border-color: var(--active-green);
    color: white;
    box-shadow: 0 0 10px rgba(46, 204, 113, 0.5);
}

.gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 8px;
    min-height: 120px;
    border: 1px dashed rgba(255, 255, 255, 0.2);
    border-radius: 6px;
    padding: 8px;
}

.gallery-grid.comparison-mode {
    grid-template-columns: 1fr 1fr;
}

.gallery-image-container {
    position: relative;
    border: #000000;
    border-radius: 4px;
    overflow: hidden;
    background: rgba(0, 0, 0, 0.3);
    aspect-ratio: 1;
    display: flex;
    align-items: center;
    justify-content: center;
}

.gallery-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.gallery-placeholder {
    color: var(--text-gray);
    font-family: 'Orbitron', monospace;
    font-size: 10px;
    text-align: center;
    grid-column: 1 / -1;
    padding: 30px;
}

/* Loading spinner */
.loading-spinner {
    border: 2px solid rgba(var(--primary-accent-rgb), 0.2);
    border-top: 2px solid var(--primary-accent);
    border-radius: 50%;
    width: 16px;
    height: 16px;
    animation: spin 1s linear infinite;
    margin: 8px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Collapse button */
.fluxaio-collapse-btn {
    width: 28px;
    height: 28px;
    background: rgba(var(--primary-accent-rgb), 0.2);
    border: 1px solid var(--primary-accent);
    border-radius: 6px;
    color: var(--primary-accent);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    font-weight: bold;
    transition: all 0.3s ease;
    padding: 0;
}

.fluxaio-collapse-btn:hover {
    background: var(--primary-accent);
    color: var(--text-white);
}

/* Better scrollbar styling for sections */
.fluxaio-section::-webkit-scrollbar {
    width: 6px;
}

.fluxaio-section::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
}

.fluxaio-section::-webkit-scrollbar-thumb {
    background: var(--primary-accent);
    border-radius: 3px;
}

.fluxaio-section::-webkit-scrollbar-thumb:hover {
    background: var(--primary-accent-light);
}

/* NEW: Inference Sub-tabs */
.inference-sub-tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
    justify-content: center;
    border-bottom: 1px solid rgba(var(--primary-accent-rgb), 0.2);
    padding-bottom: 8px;
}

.inference-sub-tab {
    background: #1a1a1a;
    color: var(--primary-accent-light);
    border: 1px solid var(--primary-accent);
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 11px;
    font-weight: 600;
    font-family: 'Orbitron', monospace;
    transition: all 0.3s ease;
    text-transform: uppercase;
	transition: color 0.3s ease, border-color 0.3s ease;
}

.inference-sub-section {
    display: none;
    padding: 8px;
    border: 1px dashed rgba(var(--primary-accent-rgb), 0.2);
    border-radius: 8px;
    margin-top: 8px;
}

.inference-sub-section.active {
    display: block;
}

/* --- NEW KNOB STYLES --- */
.flux-knob-label {
    color: #aaa;
    font-family: 'Orbitron', monospace;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
    text-align: center;
}

.flux-knob {
    --knob-size: 160px;
    --knob-color: #00ffe7;
    --knob-bg: radial-gradient(circle at 50% 50%, #1e1e1e, #0d0d0d);
    --glow-color: var(--knob-color);
    --tick-color: rgba(255, 255, 255, 0.1);
    --knob-value-percent: 30;

    width: var(--knob-size);
    height: var(--knob-size);
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--knob-bg);
    border-radius: 50%;
    box-shadow:
        inset 0 0 10px rgba(0, 0, 0, 0.4),
        0 0 10px rgba(0, 0, 0, 0.5);
    cursor: grab;
    user-select: none;
    transition: transform 0.2s ease;
}

.flux-knob:hover {
    transform: scale(1.05);
}

.flux-knob:active {
    cursor: grabbing;
    transform: scale(1.07);
}

/* Glow Ring */
.flux-knob-glow-ring {
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 100%;
    border-radius: 50%;
    background: conic-gradient(
        from 180deg,
        transparent 0deg,
        var(--glow-color) calc(var(--knob-value-percent) * 3.6deg),
        transparent calc(var(--knob-value-percent) * 3.6deg + 2deg)
    );
    filter: blur(4px);
    opacity: 0.6;
    z-index: 0;
}

/* Tick Marks */
.flux-knob-ticks {
    position: absolute;
    width: 90%;
    height: 90%;
    border-radius: 50%;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
    z-index: 1;
}

.flux-knob-tick {
    position: absolute;
    width: 2px;
    height: 8px;
    background: var(--tick-color);
    top: 0;
    left: 50%;
    transform-origin: bottom center;
}

/* Output Display */
.flux-knob output {
    font-family: 'Orbitron', monospace;
    font-size: 1.4em;
    color: white;
    z-index: 3;
    text-shadow: 0 0 4px black;
    pointer-events: none;
    user-select: none;
    text-align: center;
}

/* FIX: Green glow for non-default knobs */
.flux-knob.non-default .flux-knob-glow-ring {
    --knob-glow-color: var(--active-green);
}
.flux-knob.non-default .flux-knob-label {
    color: var(--active-green);
}

/* Radial Blur Control Pad Styles */
.radial-blur-pad-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 10px;
}

/* FIX: Re-layout for Radial Blur controls */
.radial-blur-layout {
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 16px;
    align-items: center;
}

.control-pad {
    width: 200px;
    height: 200px;
    background: rgba(var(--primary-accent-rgb), 0.1);
    border: 1px solid var(--primary-accent);
    border-radius: 8px;
    position: relative;
    cursor: crosshair;
}

.control-pad-handle {
    width: 12px;
    height: 12px;
    background: var(--active-green);
    border-radius: 50%;
    border: 2px solid white;
    position: absolute;
    transform: translate(-50%, -50%);
    pointer-events: none;
    box-shadow: 0 0 8px var(--active-green);
}

/* --- NEW THREE-STATE SWITCH --- */
.three-state-switch {
    display: flex;
    align-items: center;
    justify-content: center;
    background: #000;
    border-radius: 15px;
    padding: 4px;
    border: 1px solid var(--primary-accent);
    position: relative;
    width: 210px;
    height: 30px;
    cursor: pointer;
    margin: 10px auto;
}
.three-state-switch .switch-label {
    flex: 1;
    text-align: center;
    font-family: 'Orbitron', monospace;
    font-size: 10px;
    font-weight: 600;
    z-index: 2;
    transition: color 0.3s ease;
    color: var(--text-gray);
    text-transform: uppercase;
}
.three-state-switch .switch-label.active {
    color: var(--text-white);
}
.three-state-switch .switch-thumb {
    position: absolute;
    top: 2px;
    bottom: 2px;
    width: calc(33.33% - 4px);
    background: var(--inactive-gray);
    border-radius: 12px;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), background-color 0.3s ease;
    z-index: 1;
}
.three-state-switch.state-compile .switch-thumb {
    transform: translateX(-100%);
    background: var(--performance-cyan);
}
.three-state-switch.state-off .switch-thumb {
    transform: translateX(0%);
    background: var(--inactive-gray);
}
.three-state-switch.state-teacache .switch-thumb {
    transform: translateX(100%);
    background: var(--lora-gold);
}
select[data-param-name="precision"],
select[data-param-name="tiles"] {
    align-self: end;
}
.perf-options-container {
    display: none;
    padding: 10px;
    margin-top: 10px;
    border: 1px dashed rgba(var(--primary-accent-rgb), 0.3);
    border-radius: 8px;
    background: rgba(0,0,0,0.2);
}

.inference-sub-tab {
    padding: 8px 16px;
    border: 2px solid var(--border-color);
    background: var(--bg-color);
    color: var(--text-color);
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.3s ease;
    font-size: 13px;
    font-weight: 500;
    margin-right: 8px;
}

.inference-sub-tab:hover {
    background: var(--hover-color);
    border-color: var(--active-color);
}

.inference-sub-tab.active {
    background: var(--active-color);
    color: white;
    border-color: var(--active-color);
}

.inference-sub-tab.enabled {
    color: var(--active-green) !important;
    border-color: var(--active-green) !important;
    box-shadow: 0 0 8px rgba(46, 204, 113, 0.3);
}

.inference-sub-tab.active.enabled {
    background: var(--active-green) !important;
    border-color: var(--active-green) !important;
    color: white !important;
}

.full-width-control {
    grid-column: 1 / -1;
}

select[data-param-name="precision"] {
    align-self: end;
}
`;

// Model paths configuration
const MODEL_PATHS = {
    flux_models: "diffusion_models",
    vae_models: "vae", 
    text_encoders: "text_encoders",
    upscale_models: "upscale_models",
    style_models: "style_models",
    clip_vision: "clip_vision",
    loras: "loras"
};

// Complete sampler and scheduler options
const SAMPLERS = [
    "euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", 
    "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", 
    "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", 
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ddim", "uni_pc", 
    "uni_pc_bh2", "deis"
];

const SCHEDULERS = [
    "normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", 
    "beta", "linear", "cosine"
];

// Seed modes
const SEED_MODES = [
    "fixed", "increment", "decrement", "randomize"
];

const RESOLUTION_PRESETS = [
    "896x1152 (3:4 Portrait)", "768x1344 (9:16 Portrait)", "832x1216 (2:3 Portrait)", 
    "1024x1024 (1:1 Square)", "1152x896 (4:3 Landscape)", "1344x768 (16:9 Widescreen)", 
    "1216x832 (3:2 Landscape)", "1536x640 (21:9 CinemaScope)"
];

// Parameter options mapping
const PARAMETER_OPTIONS = {
    'compile_backend': ["inductor", "aot_eager", "cudagraphs"],
    'compile_mode': ["default", "max-autotune", "reduce-overhead"],
    'patch_order': ["weight_patch_first", "object_patch_first"],
    'full_load': ["auto", "enabled", "disabled"],
    'florence2_model': [
        'microsoft/Florence-2-base', 'microsoft/Florence-2-base-ft',
        'microsoft/Florence-2-large', 'microsoft/Florence-2-large-ft',
        'HuggingFaceM4/Florence-2-DocVQA', 'thwri/CogFlorence-2.1-Large',
        'thwri/CogFlorence-2.2-Large', 'gokaygokay/Florence-2-SD3-Captioner',
        'gokaygokay/Florence-2-Flux-Large', 'MiaoshouAI/Florence-2-base-PromptGen-v1.5',
        'MiaoshouAI/Florence-2-large-PromptGen-v1.5', 'MiaoshouAI/Florence-2-base-PromptGen-v2.0',
        'MiaoshouAI/Florence-2-large-PromptGen-v2.0'
    ],
    'florence2_task': [
        'region_caption', 'dense_region_caption', 'region_proposal', 'caption',
        'detailed_caption', 'more_detailed_caption', 'caption_to_phrase_grounding',
        'referring_expression_segmentation', 'ocr', 'ocr_with_region', 'docvqa',
        'prompt_gen_tags', 'prompt_gen_mixed_caption', 'prompt_gen_analyze',
        'prompt_gen_mixed_caption_plus'
    ],
    'florence2_precision': ["fp16", "bf16", "fp32"],
    'florence2_attention': ["flash_attention_2", "sdpa", "eager"],
    'strength_type': ["multiply", "attn_bias"],
    'crop': ["center", "none"],
    'rescale_method': ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
    'precision': ["auto", "fp32", "fp16", "bf16"],
    'radial_blur_type': ["spin", "zoom"],
    'glare_type': ["star_4", "star_6", "star_8", "anamorphic_h"],
    'normalize_injected_noise': ["enable", "disable"]
};

// Post-processing operation groups
const POSTPROCESS_OPERATIONS = {
    "LEVELS": {
        icon: "📊",
        params: [
            { name: "exposure", label: "Exposure", min: -3, max: 3, step: 0.01, default: 0 },
            { name: "gamma", label: "Gamma", min: 0.1, max: 3, step: 0.01, default: 1 },
            { name: "brightness", label: "Brightness", min: -1, max: 1, step: 0.01, default: 0 },
            { name: "contrast", label: "Contrast", min: 0, max: 3, step: 0.01, default: 1 },
            { name: "saturation", label: "Saturation", min: 0, max: 3, step: 0.01, default: 1 },
            { name: "vibrance", label: "Vibrance", min: -1, max: 1, step: 0.01, default: 0 }
        ],
        enable: "enable_levels"
    },
    "COLOR": {
        icon: "🎨",
        subgroups: {
            "Lift": [
                { name: "lift_r", label: "Red", min: -1, max: 1, step: 0.01, default: 0 },
                { name: "lift_g", label: "Green", min: -1, max: 1, step: 0.01, default: 0 },
                { name: "lift_b", label: "Blue", min: -1, max: 1, step: 0.01, default: 0 }
            ],
            "Gamma": [
                { name: "gamma_r", label: "Red", min: 0.1, max: 3, step: 0.01, default: 1 },
                { name: "gamma_g", label: "Green", min: 0.1, max: 3, step: 0.01, default: 1 },
                { name: "gamma_b", label: "Blue", min: 0.1, max: 3, step: 0.01, default: 1 }
            ],
            "Gain": [
                { name: "gain_r", label: "Red", min: 0, max: 3, step: 0.01, default: 1 },
                { name: "gain_g", label: "Green", min: 0, max: 3, step: 0.01, default: 1 },
                { name: "gain_b", label: "Blue", min: 0, max: 3, step: 0.01, default: 1 }
            ]
        },
        enable: "enable_color_wheels"
    },
    "TEMPERATURE": {
        icon: "🌡️",
        params: [
            { name: "temperature", label: "Temperature", min: -100, max: 100, step: 1, default: 0 },
            { name: "tint", label: "Tint", min: -100, max: 100, step: 1, default: 0 }
        ],
        enable: "enable_temp_tint"
    },
    "SHARPEN": {
        icon: "⚡",
        params: [
            { name: "sharpen_strength", label: "Strength", min: 0, max: 3, step: 0.01, default: 2.5 },
            { name: "sharpen_radius", label: "Radius", min: 0.1, max: 5, step: 0.1, default: 1.85 },
            { name: "sharpen_threshold", label: "Threshold", min: 0, max: 1, step: 0.01, default: 0.015 }
        ],
        enable: "enable_sharpen"
    },
    "SMALL_GLOW": {
        icon: "✨",
        params: [
            { name: "small_glow_intensity", label: "Intensity", min: 0, max: 2, step: 0.01, default: 0.015 },
            { name: "small_glow_radius", label: "Radius", min: 0, max: 10, step: 0.1, default: 0.1 },
            { name: "small_glow_threshold", label: "Threshold", min: 0, max: 1, step: 0.01, default: 0.25 }
        ],
        enable: "enable_small_glow"
    },
    "LARGE_GLOW": {
        icon: "🌟",
        params: [
            { name: "large_glow_intensity", label: "Intensity", min: 0, max: 2, step: 0.01, default: 0.25 },
            { name: "large_glow_radius", label: "Radius", min: 30, max: 100, step: 0.5, default: 50 },
            { name: "large_glow_threshold", label: "Threshold", min: 0, max: 1, step: 0.01, default: 0.3 }
        ],
        enable: "enable_large_glow"
    },
    "GLARE": {
        icon: "⭐",
        params: [
            { name: "glare_intensity", label: "Intensity", min: 0, max: 100, step: 0.01, default: 0.65 },
            { name: "glare_length", label: "Length", min: 0.1, max: 100, step: 0.01, default: 1.5 },
            { name: "glare_angle", label: "Angle", min: -180, max: 180, step: 1, default: 0 },
            { name: "glare_threshold", label: "Threshold", min: 0, max: 1, step: 0.01, default: 0.95 },
            { name: "glare_quality", label: "Quality", min: 4, max: 32, step: 4, default: 16 },
            { name: "glare_ray_width", label: "Ray Width", min: 0.1, max: 5, step: 0.1, default: 1 }
        ],
        selectors: [
            { name: "glare_type", label: "Glare Type", options: ["star_4", "star_6", "star_8", "anamorphic_h"] }
        ],
        enable: "enable_glare"
    },
    "CHROMATIC": {
        icon: "🌈",
        params: [
            { name: "ca_strength", label: "Strength", min: 0, max: 0.1, step: 0.001, default: 0.005 },
            { name: "ca_edge_falloff", label: "Edge Falloff", min: 0, max: 2, step: 0.01, default: 2 },
            { name: "ca_hue_shift_degrees", label: "Hue Shift", min: -180, max: 180, step: 1, default: 0 }
        ],
        toggles: [
            { name: "enable_ca_hue_shift", label: "Enable Hue Shift" }
        ],
        enable: "enable_chromatic_aberration"
    },
    "VIGNETTE": {
        icon: "🔘",
        params: [
            { name: "vignette_strength", label: "Strength", min: 0, max: 2, step: 0.01, default: 0.5 },
            { name: "vignette_radius", label: "Radius", min: 0.1, max: 3, step: 0.01, default: 0.7 },
            { name: "vignette_softness", label: "Softness", min: 0, max: 4, step: 0.01, default: 2 }
        ],
        enable: "enable_vignette"
    },
    "RADIAL_BLUR": {
        icon: "🌀",
        params: [
            { name: "radial_blur_strength", label: "Strength", min: 0, max: 0.5, step: 0.005, default: 0.02 },
            { name: "radial_blur_center_x", label: "Center X", min: 0, max: 1, step: 0.01, default: 0.5 },
            { name: "radial_blur_center_y", label: "Center Y", min: 0, max: 1, step: 0.01, default: 0.25 },
            { name: "radial_blur_falloff", label: "Falloff", min: 0.001, max: 1, step: 0.01, default: 0.25 },
            { name: "radial_blur_samples", label: "Samples", min: 8, max: 64, step: 8, default: 16 }
        ],
        selectors: [
            { name: "radial_blur_type", label: "Blur Type", options: ["spin", "zoom"] }
        ],
        enable: "enable_radial_blur"
    },
    "FILM_GRAIN": {
        icon: "📽️",
        params: [
            { name: "grain_intensity", label: "Intensity", min: 0, max: 0.15, step: 0.01, default: 0.02 },
            { name: "grain_size", label: "Size", min: 0.25, max: 4, step: 0.05, default: 1 },
            { name: "grain_color_amount", label: "Color Amount", min: 0, max: 1, step: 0.01, default: 0 }
        ],
        enable: "enable_film_grain"
    },
    "LENS_DISTORTION": {
        icon: "🔍",
        params: [
            { name: "barrel_distortion", label: "Barrel Distortion", min: -0.5, max: 0.5, step: 0.001, default: 0 }
        ],
        enable: "enable_lens_distortion"
    }
};

class FluxAIOUI {
    constructor(node) {
        console.log(`[FluxAIOUI] Constructor called for node ${node.id}`);
		this.filteredLoraList = [];
		this.selectedLoraIndex = -1;
		this.finalImageDisplayed = false;
        this.node = node;
        this.container = null;
        this.activeTab = "inference";
        this.activePostProcessTab = "LEVELS";
        this.activeLoRABlocksTab = "double_blocks";
        this.activeInferenceSubTab = 'first_pass';
        this.numberInputs = {};
        this.retryCount = 0;
        this.maxRetries = 5;
        this.isInitialized = false;
        this.resizeObserver = null;
        this.isCollapsed = false;
        this.contentContainer = null;
        this.animationFrameId = null;
        this.loraList = [];
        this.loraStack = [];
        this.resultImages = {};
        this.comparisonMode = false;
        this.presets = this.loadPresets();
        this.domWidget = null;
        this.currentTheme = this.loadTheme();
        this.seedCounter = 0;
        this.loraBlockValues = {
            double_blocks: Array(19).fill(1.0),
            single_blocks: Array(38).fill(1.0)
        };
        this.loraBlocksSettings = {
            double_blocks: { min: -2, max: 2 },
            single_blocks: { min: -2, max: 2 }
        };
        this.selectedLoRABlock = { type: null, index: -1 };
		this.additionalPromptText = '';
		this.originalBasePrompt = null;
        
        this.livePreviewImage = null; // To hold the live preview image element
        this.livePreviewContainer = null; // To hold the container for the preview
        this.currentPreviewImageURL = null; // To manage blob URLs

        // Bind methods
        this.handleResize = this.handleResize.bind(this);
        
        // Add to global debug tracker
        if (window.FluxAIOUI_Debug) {
            window.FluxAIOUI_Debug.logInstance(this);
        }
        
		// Use requestAnimationFrame for faster initialization
		requestAnimationFrame(() => this.initializeWithRetry());
	}
    
	updatePreview(imageBlob) {
		if (!this.livePreviewContainer || !this.livePreviewImage) {
			console.error("[FluxAIOUI] Live preview UI elements not ready.");
			return;
		}

		if (this.finalImageDisplayed) {
			return;
		}

		if (this.currentPreviewImageURL) {
			URL.revokeObjectURL(this.currentPreviewImageURL);
		}

		this.currentPreviewImageURL = URL.createObjectURL(imageBlob);
		this.livePreviewImage.src = this.currentPreviewImageURL;

		this.livePreviewContainer.classList.remove('empty');
		
		const title = this.livePreviewContainer.querySelector('.live-preview-title');
		if (title) {
			title.textContent = 'Live Preview (Generating...)';
		}
	}
	
    updateConditionalUIVisibility() {
        if (!this.container) return;

        const isImg2ImgEnabled = this.getWidgetValue('enable_img2img', false);
        const img2imgDenoiseSlider = this.container.querySelector('.param-slider-container:has(input[type="range"][data-param-name="img2img_denoise"])');
        if (img2imgDenoiseSlider) {
            img2imgDenoiseSlider.style.display = isImg2ImgEnabled ? 'block' : 'none';
        }

        const isStyleModelEnabled = this.getWidgetValue('enable_style_model', false);
        const styleStrengthSlider = this.container.querySelector('.param-slider-container:has(input[type="range"][data-param-name="style_strength"])');
        if(styleStrengthSlider) {
            styleStrengthSlider.style.display = isStyleModelEnabled ? 'block' : 'none';
        }
    }	
	
	setWidgetValue(name, value, skipCanvasDirty = false) {
		const widget = this.node.widgets?.find(w => w.name === name);
		if (widget) {
			let validatedValue = value;
			
			// Handle LoRA block weights specifically
			if (name.includes('lora_block_') && name.includes('_weight')) {
				if (value === '' || value === null || value === undefined || value === 'None' || isNaN(parseFloat(value))) {
					validatedValue = 1.0;
					console.log(`[FluxAIOUI] Fixed ${name}: empty/invalid -> 1.0`);
				} else {
					let numValue = parseFloat(value);
					validatedValue = Math.max(-2.0, Math.min(2.0, numValue));
				}
			}
			// Handle LoRA clip strength parameters
			else if (name.includes('lora_') && name.includes('_clip_strength')) {
				if (value === '' || value === null || value === undefined || value === 'None' || isNaN(parseFloat(value))) {
					validatedValue = 1.0;
					console.log(`[FluxAIOUI] Fixed ${name}: empty/None -> 1.0`);
				} else {
					validatedValue = parseFloat(value);
				}
			}
			// Handle LoRA strength parameters
			else if (name.includes('lora_') && name.includes('_strength') && !name.includes('clip')) {
				if (value === '' || value === null || value === undefined || value === 'None' || isNaN(parseFloat(value))) {
					validatedValue = 1.0;
					console.log(`[FluxAIOUI] Fixed ${name}: empty/None -> 1.0`);
				} else {
					validatedValue = parseFloat(value);
				}
			}
			// Handle specific problematic parameters
			else if (name === 'barrel_distortion') {
				if (value === '' || value === null || value === undefined || isNaN(parseFloat(value))) {
					validatedValue = 0.0;
				} else {
					let numValue = parseFloat(value);
					validatedValue = Math.max(-0.5, Math.min(0.5, numValue));
				}
			}
			else if (name === 'downscale_by') {
				if (value === '' || value === null || value === undefined || value === 'nearest-exact' || isNaN(parseFloat(value))) {
					validatedValue = 0.5;
					console.log(`[FluxAIOUI] Fixed ${name}: invalid -> 0.5`);
				} else {
					let numValue = parseFloat(value);
					validatedValue = Math.max(0.25, Math.min(1.0, numValue));
				}
			}
			else if (name === 'steps' || name === 'steps_2ND') {
				if (value === '' || value === null || value === undefined || value === 'deis' || isNaN(parseFloat(value))) {
					validatedValue = name === 'steps' ? 28 : 18;
					console.log(`[FluxAIOUI] Fixed ${name}: invalid -> ${validatedValue}`);
				} else {
					validatedValue = Math.max(1, Math.min(100, parseInt(parseFloat(value))));
				}
			}
			// Handle seed parameters
			else if (name.includes('seed')) {
				if (value === '' || value === null || value === undefined || isNaN(parseFloat(value))) {
					validatedValue = name === 'seed' ? 1 : 0;
				} else {
					validatedValue = parseInt(parseFloat(value));
				}
			}
			// Handle range-limited parameters
			else if (name === 'sharpen_radius') {
				if (value === '' || value === null || value === undefined || isNaN(parseFloat(value))) {
					validatedValue = 1.85;
				} else {
					validatedValue = Math.max(0.1, Math.min(5.0, parseFloat(value)));
				}
			}
			else if (name === 'batch_size_2ND') {
				if (value === '' || value === null || value === undefined || isNaN(parseFloat(value))) {
					validatedValue = 1;
				} else {
					validatedValue = Math.max(1, Math.min(100, parseInt(parseFloat(value))));
				}
			}
			else if (name === 'glare_quality') {
				if (value === '' || value === null || value === undefined || isNaN(parseFloat(value))) {
					validatedValue = 16;
				} else {
					validatedValue = Math.max(4, Math.min(32, parseInt(parseFloat(value))));
				}
			}
			else if (name === 'radial_blur_samples') {
				if (value === '' || value === null || value === undefined || isNaN(parseFloat(value))) {
					validatedValue = 16;
				} else {
					validatedValue = Math.max(8, Math.min(64, parseInt(parseFloat(value))));
				}
			}
			else if (name === 'large_glow_radius') {
				if (value === '' || value === null || value === undefined || isNaN(parseFloat(value))) {
					validatedValue = 50.0;
				} else {
					validatedValue = Math.max(30.0, Math.min(100.0, parseFloat(value)));
				}
			}
			else if (name === 'gamma') {
				if (value === '' || value === null || value === undefined || isNaN(parseFloat(value))) {
					validatedValue = 1.0;
				} else {
					validatedValue = Math.max(0.1, Math.min(3.0, parseFloat(value)));
				}
			}
			// Handle dropdown parameters with validation
			else if (name === 'sampler_name') {
				const validSamplers = ["euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", 
									  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", 
									  "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", 
									  "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ddim", "uni_pc", 
									  "uni_pc_bh2", "deis"];
				if (!validSamplers.includes(value)) {
					validatedValue = 'deis';
					console.log(`[FluxAIOUI] Fixed ${name}: invalid '${value}' -> 'deis'`);
				}
			}
			else if (name === 'scheduler') {
				const validSchedulers = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", 
										"beta", "linear", "cosine"];
				if (!validSchedulers.includes(value)) {
					validatedValue = 'beta';
					console.log(`[FluxAIOUI] Fixed ${name}: invalid '${value}' -> 'beta'`);
				}
			}
			// Handle general string emptiness
			else if (typeof value === 'string' && value.trim() === '') {
				if (name.includes('weight') || name.includes('strength') || name.includes('scale')) {
					validatedValue = 1.0;
				} else if (name.includes('seed')) {
					validatedValue = 1;
				} else if (name.includes('steps')) {
					validatedValue = 20;
				} else {
					validatedValue = 0.0;
				}
			}
			
			widget.value = validatedValue;
			
			if (!skipCanvasDirty) {
				this.node.setDirtyCanvas(true, true);
			}
			return true;
		} else {
			console.warn(`[FluxAIOUI] Widget not found: ${name}`);
			return false;
		}
	}

	forceSyncToBackend() {
		this.updateCombinedPrompt();
		this.resetPreviewForNewGeneration();
		console.log("[FluxAIOUI] Force syncing all UI values to backend widgets before execution...");
		
		const criticalParams = ['seed', 'steps', 'seed_shift_2ND', 'steps_2ND', 'sampler_name', 'scheduler'];
		
		criticalParams.forEach(paramName => {
			const input = this.container.querySelector(`input[data-param="${paramName}"], select[data-param="${paramName}"]`);
			if (input) {
				let value = input.value;
				
				// Additional validation for critical parameters
				if (paramName === 'steps' || paramName === 'steps_2ND') {
					if (value === '' || value === 'deis' || isNaN(parseFloat(value))) {
						value = paramName === 'steps' ? 28 : 18;
						input.value = value;
						console.log(`[FluxAIOUI] Fixed ${paramName} in UI: invalid -> ${value}`);
					} else {
						value = Math.max(1, Math.min(100, parseInt(parseFloat(value))));
						input.value = value;
					}
				} else if (paramName.includes('seed')) {
					if (value === '' || isNaN(parseFloat(value))) {
						value = paramName === 'seed' ? 1 : 0;
						input.value = value;
						console.log(`[FluxAIOUI] Fixed ${paramName} in UI: invalid -> ${value}`);
					} else {
						value = parseInt(parseFloat(value));
						input.value = value;
					}
				} else if (paramName === 'sampler_name') {
					if (!SAMPLERS.includes(value)) {
						value = 'deis';
						input.value = value;
						console.log(`[FluxAIOUI] Fixed ${paramName} in UI: invalid -> ${value}`);
					}
				} else if (paramName === 'scheduler') {
					if (!SCHEDULERS.includes(value)) {
						value = 'beta';
						input.value = value;
						console.log(`[FluxAIOUI] Fixed ${paramName} in UI: invalid -> ${value}`);
					}
				}
				
				// Convert to proper type before setting
				if (input.type === 'number') {
					value = parseFloat(value) || (paramName.includes('seed') ? 1 : 0);
				}
				
				this.setWidgetValue(paramName, value, true);
				console.log(`[FluxAIOUI] Force synced ${paramName}: ${value}`);
			}
		});
		
		// Additional validation for LoRA parameters
		for (let i = 1; i <= 6; i++) {
			['_strength', '_clip_strength'].forEach(suffix => {
				const paramName = `lora_${i}${suffix}`;
				const input = this.container.querySelector(`input[data-param="${paramName}"]`);
				if (input) {
					let value = input.value;
					if (value === '' || value === 'None' || isNaN(parseFloat(value))) {
						value = 1.0;
						input.value = value;
						console.log(`[FluxAIOUI] Fixed ${paramName} in UI: invalid -> 1.0`);
					}
					this.setWidgetValue(paramName, parseFloat(value), true);
				}
			});
		}
		
		// Validate LoRA block weights
		for (let i = 0; i < 38; i++) {
			const paramName = `lora_block_${i}_weight`;
			const widget = this.node.widgets?.find(w => w.name === paramName);
			if (widget && (widget.value === '' || widget.value === null || widget.value === undefined)) {
				widget.value = 1.0;
				console.log(`[FluxAIOUI] Fixed ${paramName}: empty -> 1.0`);
			}
		}
		
		for (let i = 0; i < 19; i++) {
			const paramName = `lora_block_${i}_double_weight`;
			const widget = this.node.widgets?.find(w => w.name === paramName);
			if (widget && (widget.value === '' || widget.value === null || widget.value === undefined)) {
				widget.value = 1.0;
				console.log(`[FluxAIOUI] Fixed ${paramName}: empty -> 1.0`);
			}
		}
		
		this.node.setDirtyCanvas(true, true);
	}	
	
    displayFinalImage(imgData) {
        if (!this.livePreviewContainer || !this.livePreviewImage) {
            console.error("[FluxAIOUI] Live preview UI elements not ready.");
            return;
        }

        this.finalImageDisplayed = true;
        
        if (this.currentPreviewImageURL) {
            URL.revokeObjectURL(this.currentPreviewImageURL);
            this.currentPreviewImageURL = null;
        }

        // FIX: Handle both file-based and base64-based previews
        if (imgData.source && imgData.format) { // This is a fast preview
            this.livePreviewImage.src = `data:image/${imgData.format};base64,${imgData.source}`;
            console.log("[FluxAIOUI] Displaying base64 fast preview.");
        } else { // This is a standard file-based preview
            const url = new URL(api.api_url);
            url.pathname = "/view";
            url.searchParams.set("filename", imgData.filename);
            url.searchParams.set("subfolder", imgData.subfolder);
            url.searchParams.set("type", imgData.type);
            this.livePreviewImage.src = url.toString();
            console.log("[FluxAIOUI] Displaying file-based final image preview.");
        }

        this.livePreviewContainer.classList.remove('empty');
        
        const title = this.livePreviewContainer.querySelector('.live-preview-title');
        if (title) {
            title.textContent = 'Final Result';
        }
    }

	resetPreviewForNewGeneration() {
		this.finalImageDisplayed = false;
		
		if (this.livePreviewContainer) {
			this.livePreviewContainer.classList.add('empty');
		}
		if (this.livePreviewImage) {
			this.livePreviewImage.src = '';
		}
		
		const title = this.livePreviewContainer?.querySelector('.live-preview-title');
		if (title) {
			title.textContent = 'Live Preview';
		}
		
		if (this.currentPreviewImageURL) {
			URL.revokeObjectURL(this.currentPreviewImageURL);
			this.currentPreviewImageURL = null;
		}
		
		console.log("[FluxAIOUI] Preview reset for new generation");
	}
	
    addFinalImageToGallery(imgData) {
        const galleryGrid = this.container.querySelector('#gallery-grid');
        if (!galleryGrid) return;
    
        const placeholder = galleryGrid.querySelector('.gallery-placeholder');
        if (placeholder) placeholder.remove();
    
        const container = document.createElement('div');
        container.className = 'gallery-image-container';
    
        const img = document.createElement('img');
        img.className = 'gallery-image';
        
        // FIX: Handle both preview types
        if (imgData.source && imgData.format) {
            img.src = `data:image/${imgData.format};base64,${imgData.source}`;
        } else {
            const url = new URL(api.api_url);
            url.pathname = "/view";
            url.searchParams.set("filename", imgData.filename);
            url.searchParams.set("subfolder", imgData.subfolder);
            url.searchParams.set("type", imgData.type);
            img.src = url.toString();
        }
        
        container.appendChild(img);
        galleryGrid.appendChild(container);
    }

    loadPresets() {
        const presets = localStorage.getItem('fluxAIOPresets');
        return presets ? JSON.parse(presets) : {};
    }

    savePresets() {
        localStorage.setItem('fluxAIOPresets', JSON.stringify(this.presets));
    }

    loadTheme() {
        return localStorage.getItem('fluxAIOTheme') || '#7700ff';
    }

    saveTheme() {
        localStorage.setItem('fluxAIOTheme', this.currentTheme);
    }
	
    updateCombinedPrompt() {
        const positivePromptWidget = this.node.widgets?.find(w => w.name === 'positive_prompt');
        let basePrompt = '';
        
        if (positivePromptWidget && positivePromptWidget.value) {
            basePrompt = positivePromptWidget.value.toString().trim();
        }
        
        if (this.originalBasePrompt === null && basePrompt) {
            this.originalBasePrompt = basePrompt;
        }
        
        let finalPrompt = basePrompt;
        if (this.additionalPromptText && this.additionalPromptText.trim()) {
            if (finalPrompt) {
                finalPrompt += ', ' + this.additionalPromptText.trim();
            } else {
                finalPrompt = this.additionalPromptText.trim();
            }
        }
        
        if (positivePromptWidget) {
            positivePromptWidget.value = finalPrompt;
        }
        
        console.log(`[FluxAIOUI] Combined prompt: "${finalPrompt}"`);
    }
	
	updateSeedDisplays() {
		const seedInput = this.container?.querySelector('input[data-param="seed"]');
		const seedShiftInput = this.container?.querySelector('input[data-param="seed_shift_2ND"]');
		
		if (seedInput) {
			const currentSeed = this.getWidgetValue('seed', 1);
			if (parseInt(seedInput.value) !== currentSeed) {
				seedInput.value = currentSeed;
				console.log(`[FluxAIOUI] Updated main seed display: ${currentSeed}`);
			}
		}
		
		if (seedShiftInput) {
			const currentSeedShift = this.getWidgetValue('seed_shift_2ND', 1);
			if (parseInt(seedShiftInput.value) !== currentSeedShift) {
				seedShiftInput.value = currentSeedShift;
				console.log(`[FluxAIOUI] Updated seed shift display: ${currentSeedShift}`);
			}
		}
	}
    
    async initializeWithRetry() {
        if (this.isInitialized) return;
        
        console.log(`[FluxAIOUI] Attempting to initialize UI for node ${this.node.id}, retry ${this.retryCount}`);
        
        if (!this.node.widgets || this.node.widgets.length === 0) {
            if (this.retryCount < this.maxRetries) {
                this.retryCount++;
                console.log(`[FluxAIOUI] Widgets not ready, retrying in ${this.retryCount * 2}ms`);
                setTimeout(() => this.initializeWithRetry(), this.retryCount * 2);
                return;
            } else {
                console.error(`[FluxAIOUI] Failed to initialize after ${this.maxRetries} retries - no widgets found`);
                return;
            }
        }
        
        try {
            this.setWidgetDefaults();
            await this.fetchModelLists();
            await this.fetchLoRAList();
            this.initializeUI();
            this.isInitialized = true;
            console.log(`[FluxAIOUI] UI initialized successfully for node ${this.node.id}`);
        } catch (err) {
            console.error(`[FluxAIOUI] Error initializing for node ${this.node.id}:`, err);
            if (this.retryCount < this.maxRetries) {
                this.retryCount++;
                setTimeout(() => this.initializeWithRetry(), 10);
            }
        }
    }

		async fetchModelLists() {
			try {
				let modelData = null;
				
				try {
					const response = await fetch('/object_info/FluxAIO_CRT');
					if (response.ok) {
						const data = await response.json();
						if (data.input && data.input.required) {
							const inputs = data.input.required;
							modelData = {
								flux_models: inputs.flux_model_name?.[0] || [],
								vae_models: inputs.vae_name?.[0] || [],
								text_encoders: inputs.clip_l_name?.[0] || [],
								upscale_models: inputs.upscale_model_name?.[0] || [],
								style_models: inputs.style_model_name?.[0] || [],
								clip_vision: inputs.clip_vision_name?.[0] || [],
								// --- CORRECTED KEYS START HERE ---
								controlnet_models: inputs.flux_cnet_upscaler_model?.[0] || [],
								bbox_models: inputs.face_bbox_model?.[0] || [],
								segm_models: inputs.face_segm_model?.[0] || []
								// --- CORRECTED KEYS END HERE ---
							};
						}
					}
				} catch (e) {
					console.log('[FluxAIOUI] Primary endpoint failed, trying alternatives:', e);
				}

			if (!modelData) {
				try {
					const response = await fetch('/object_info');
					if (response.ok) {
						const allData = await response.json();
						const fluxData = allData['FluxAIO_CRT'];
						if (fluxData && fluxData.input && fluxData.input.required) {
							const inputs = fluxData.input.required;
							modelData = {
								flux_models: inputs.flux_model_name?.[0] || [],
								vae_models: inputs.vae_name?.[0] || [],
								text_encoders: inputs.clip_l_name?.[0] || [],
								upscale_models: inputs.upscale_model_name?.[0] || [],
								style_models: inputs.style_model_name?.[0] || [],
								clip_vision: inputs.clip_vision_name?.[0] || [],
								// --- CORRECTED KEYS START HERE ---
								controlnet_models: inputs.flux_cnet_upscaler_model?.[0] || [],
								bbox_models: inputs.face_bbox_model?.[0] || [],
								segm_models: inputs.face_segm_model?.[0] || []
								// --- CORRECTED KEYS END HERE ---
							};
						}
					}
				} catch (e) {
					console.log('[FluxAIOUI] Secondary endpoint failed:', e);
				}
			}

			this.modelLists = modelData || {
				flux_models: ['flux1-dev.safetensors', 'flux1-schnell.safetensors'],
				vae_models: ['ae.safetensors'],
				text_encoders: ['clip_l.safetensors', 't5xxl_fp16.safetensors'],
				upscale_models: ['4x_foolhardy_Remacri.pth'],
				style_models: ['flux1-redux-dev.safetensors'],
				clip_vision: ['sigclip_vision_patch14_384.safetensors'],
				controlnet_models: ['None'],
				bbox_models: ['None'],
				segm_models: ['None']
			};

			console.log(`[FluxAIOUI] Final model lists:`, this.modelLists);

			if (this.isInitialized) {
				this.syncAllSelects();
			}

		} catch (error) {
			console.error('[FluxAIOUI] Failed to fetch model lists:', error);
			this.modelLists = {
				flux_models: ['flux1-dev.safetensors', 'flux1-schnell.safetensors'],
				vae_models: ['ae.safetensors'],
				text_encoders: ['clip_l.safetensors', 't5xxl_fp16.safetensors'],
				upscale_models: ['4x_foolhardy_Remacri.pth'],
				style_models: ['flux1-redux-dev.safetensors'],
				clip_vision: ['sigclip_vision_patch14_384.safetensors'],
				controlnet_models: ['None'],
				bbox_models: ['None'],
				segm_models: ['None']
			};
		}
	}

	async fetchLoRAList() {
			try {
				console.log('[FluxAIOUI] Attempting to fetch LoRA list...');
				let foundLoras = [];
				
				try {
					const response = await fetch('/loras');
					if (response.ok) {
						const data = await response.json();
						if (Array.isArray(data) && data.length > 0) {
							foundLoras = data;
							console.log(`[FluxAIOUI] Fetched ${foundLoras.length} LoRAs from /loras endpoint.`);
						}
					}
				} catch (e) {
					console.log('[FluxAIOUI] /loras endpoint failed, trying alternatives:', e.message);
				}

				if (foundLoras.length === 0) {
					try {
						const response = await fetch('/object_info');
						if (response.ok) {
							const data = await response.json();
							const loraSet = new Set();
							const potentialLoraNodes = [
								'LoraLoader', 'LoraLoaderModelOnly', 'LoRALoader', 'LoraLoader|pysssss',
								'LoraLoaderSimple', 'FluxAIO_CRT', 'Power Lora Loader (rgthree)'
							];

							for (const nodeName in data) {
								const nodeInfo = data[nodeName];
								if (potentialLoraNodes.some(name => nodeName.includes(name)) || nodeName.toLowerCase().includes('lora')) {
									if (nodeInfo.input?.required) {
										Object.entries(nodeInfo.input.required).forEach(([paramName, paramData]) => {
											if (paramName.toLowerCase().includes('lora') && Array.isArray(paramData) && Array.isArray(paramData[0])) {
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
								foundLoras = Array.from(loraSet).sort();
								console.log(`[FluxAIOUI] Found ${foundLoras.length} LoRAs from object_info.`);
							}
						}
					} catch (e) {
						console.log('[FluxAIOUI] object_info fallback failed:', e.message);
					}
				}
				
				if (foundLoras.length > 0) {
					this.loraList = foundLoras;
				} else {
					console.warn('[FluxAIOUI] All LoRA fetching strategies failed, using fallback list');
					this.loraList = ['No LoRAs found - Check your loras folder'];
				}

			} catch (error) {
				console.error('[FluxAIOUI] Critical error in fetchLoRAList:', error);
				this.loraList = ['Error loading LoRAs'];
			} finally {
				this.filteredLoraList = [...this.loraList];
				this.updateLoRAListDisplay();
			}
		}

	updateLoRAListDisplay() {
		const listContainer = this.container?.querySelector('.lora-select-list');
		if (!listContainer) return;
		
		listContainer.innerHTML = '';
		
		this.filteredLoraList.forEach((lora, index) => {
			if (lora && typeof lora === 'string' && lora.trim() && 
				!lora.includes('Error') && !lora.includes('No LoRAs found')) {
				
				const item = document.createElement('div');
				item.className = 'lora-select-item';
				item.textContent = lora;
				item.dataset.index = index;
				
				if (index === this.selectedLoraIndex) {
					item.classList.add('selected');
				}
				
				item.addEventListener('click', () => {
					this.addLoRAToStaticSlot(lora);
				});
				
				listContainer.appendChild(item);
			}
		});
		
		console.log(`[FluxAIOUI] Updated LoRA list display with ${this.filteredLoraList.length} items`);
	}
	
	filterLoRAList(searchTerm) {
		if (!searchTerm || searchTerm.trim() === '') {
			this.filteredLoraList = [...this.loraList];
		} else {
			this.filteredLoraList = this.loraList.filter(lora => 
				lora.toLowerCase().includes(searchTerm.toLowerCase())
			);
		}
		this.selectedLoraIndex = this.filteredLoraList.length > 0 ? 0 : -1;
		this.updateLoRAListDisplay();
	}

	handleLoRAKeyboard(event) {
		if (!this.filteredLoraList.length) return;
		
		switch (event.key) {
			case 'ArrowDown':
				event.preventDefault();
				this.selectedLoraIndex = Math.min(this.selectedLoraIndex + 1, this.filteredLoraList.length - 1);
				this.updateLoRAListDisplay();
				break;
			case 'ArrowUp':
				event.preventDefault();
				this.selectedLoraIndex = Math.max(this.selectedLoraIndex - 1, 0);
				this.updateLoRAListDisplay();
				break;
			case 'Enter':
				event.preventDefault();
				if (this.selectedLoraIndex >= 0 && this.selectedLoraIndex < this.filteredLoraList.length) {
					this.addLoRAToStaticSlot(this.filteredLoraList[this.selectedLoraIndex]);
				}
				break;
		}
	}

    setWidgetDefaults() {
        if (!this.node.widgets || this.node.widgets.length === 0) {
            console.log(`[FluxAIOUI] No widgets found on node ${this.node.id}`);
            return;
        }
        
        console.log(`[FluxAIOUI] Setting widget defaults for node ${this.node.id}`);
        
        const controlDefaults = {
            'flux_guidance': 2.7, 'steps': 28, 'steps_2ND': 18,
            'sampler_name': 'deis', 'scheduler': 'beta', 'lora_stack': '[]',
            'seed': 1, 'seed_shift_2ND': 0,
            'enable_florence2': false, 'enable_img2img': false,
            'enable_style_model': false, 'enable_upscale_with_model': false, 'enable_tiling': false,
            'enable_torch_compile': false, 'enable_2nd_pass': false, 'enable_lora_block_patcher': false,
            'resolution_preset': '832x1216 (2:3 Portrait)',
            'downscale_by': 0.5, 'img2img_denoise': 0.8,
        };
        
        Object.entries(controlDefaults).forEach(([param, defaultValue]) => {
            const widget = this.node.widgets.find(w => w.name === param);
            if (widget) {
                if(widget.value === undefined || widget.value === null) {
                    widget.value = defaultValue;
                }
            }
        });
        
        this.node.setDirtyCanvas(true, true);
    }

    initializeUI() {
        console.log(`[FluxAIOUI] Starting UI initialization for node ${this.node.id}`);
        
        try {
            this.injectStyles();
            this.hideOriginalWidgets();
            this.createCustomDOM();
            this.setupResizeObserver();
            this.applyTheme(this.currentTheme);
            
            if (!window.fluxAIOPreviewNodes) {
                window.fluxAIOPreviewNodes = new Map();
            }
            window.fluxAIOPreviewNodes.set(this.node.id, this);
            
            setTimeout(() => {
                this.setNodeSizeInstantly();
            }, 10);
            
            this.node.setDirtyCanvas(true, true);
			this.validateAllLoRABlockWidgets();
			this.initializeLoRABlockWidgets();
            
            console.log(`[FluxAIOUI] UI initialization completed successfully for node ${this.node.id}`);
        } catch (error) {
            console.error(`[FluxAIOUI] Error during UI initialization for node ${this.node.id}:`, error);
            throw error;
        }
    }

    injectStyles() {
        if (!document.getElementById('fluxaio-styles')) {
            const styleSheet = document.createElement('style');
            styleSheet.id = 'fluxaio-styles';
            styleSheet.textContent = CSS_STYLES_FLUXAIO;
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
        if (!this.node.widgets || this.node.widgets.length === 0) return;
        
        const keepVisible = ['positive_prompt', 'img2img_image', 'style_image', 'florence2_image'];
        
        this.node.widgets.forEach((widget) => {
            if (widget.name && !keepVisible.includes(widget.name)) {
                widget.computeSize = () => [0, -4];
                if (widget.element) widget.element.classList.add('widget-hidden');
            }
        });
    }

    createCustomDOM() {
        console.log(`[FluxAIOUI] Creating custom DOM for node ${this.node.id}`);
        
        if (this.container) this.container.remove();

        this.container = document.createElement('div');
        this.container.className = 'fluxaio-container';

        const title = document.createElement('div');
        title.className = 'fluxaio-title';
        title.textContent = 'FluxAIO_CRT';
        this.container.appendChild(title);

        const topBar = document.createElement('div');
        topBar.className = 'fluxaio-top-bar';
        this.createPresetsSection(topBar);
        this.createThemePicker(topBar);
        this.container.appendChild(topBar);
        
        this.createTabs();

        const contentContainer = document.createElement('div');
        contentContainer.className = 'fluxaio-content-container';
        this.contentContainer = contentContainer;

        this.createModelsSection(contentContainer);
        this.createPerformanceSection(contentContainer);
        this.createPromptSection(contentContainer);
        this.createLoRASection(contentContainer);
        this.createInferenceSection(contentContainer);
        this.createPostProcessSection(contentContainer);
        this.createGallerySection(contentContainer);
        this.createSaveSection(contentContainer);
        
        this.container.appendChild(contentContainer);

        this.updateTabVisibility();
        this.syncAllUIToWidgetState();

        const widgetWrapper = document.createElement('div');
        widgetWrapper.className = 'fluxaio-node-custom-widget';
        widgetWrapper.appendChild(this.container);
        
        try {
            const domWidget = this.node.addDOMWidget('fluxaio_ui', 'div', widgetWrapper, {
                serialize: false, hideOnZoom: false,
                onDraw: (ctx, node, widget_width, y, widget_height) => {}
            });
            
            if (domWidget) {
                this.domWidget = domWidget;
                domWidget.fluxAIOInstance = this;
            } else {
                console.error(`[FluxAIOUI] addDOMWidget returned null/undefined for node ${this.node.id}`);
            }
        } catch (error) {
            console.error(`[FluxAIOUI] Error calling addDOMWidget for node ${this.node.id}:`, error);
            throw error;
        }
    }

    createThemePicker(parent) {
        const themeSection = document.createElement('div');
        themeSection.className = 'theme-picker-container';

        const label = document.createElement('span');
        label.className = 'theme-picker-label';
        label.textContent = 'Theme:';
        this.themeLabel = label;
        themeSection.appendChild(label);

        const colorPicker = document.createElement('input');
        colorPicker.type = 'color';
        colorPicker.className = 'theme-color-picker';
        colorPicker.value = this.currentTheme;
        colorPicker.addEventListener('input', (e) => {
            this.currentTheme = e.target.value;
            this.applyTheme(this.currentTheme);
            this.saveTheme();
        });
        themeSection.appendChild(colorPicker);

        parent.appendChild(themeSection);
    }

    applyTheme(color) {
        if (!this.container) return;
        const root = this.container;
        const hexToRgb = (hex) => {
            const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
            return result ? {
                r: parseInt(result[1], 16), g: parseInt(result[2], 16), b: parseInt(result[3], 16)
            } : null;
        };
        
        const rgb = hexToRgb(color);
        if (rgb) {
            const lighterR = Math.min(255, rgb.r + 40);
            const lighterG = Math.min(255, rgb.g + 40);
            const lighterB = Math.min(255, rgb.b + 40);
            const lighterColor = `#${lighterR.toString(16).padStart(2, '0')}${lighterG.toString(16).padStart(2, '0')}${lighterB.toString(16).padStart(2, '0')}`;
            
            root.style.setProperty('--primary-accent', color);
            root.style.setProperty('--primary-accent-light', lighterColor);
            root.style.setProperty('--primary-accent-rgb', `${rgb.r}, ${rgb.g}, ${rgb.b}`);
        }
    }

    createPresetsSection(parent) {
        const presetsSection = document.createElement('div');
        presetsSection.className = 'fluxaio-presets';

        const presetSelect = document.createElement('select');
        presetSelect.className = 'preset-select';
        this.presetSelect = presetSelect;
        this.updatePresetDropdown();
        presetsSection.appendChild(presetSelect);

        const loadBtn = document.createElement('button');
        loadBtn.className = 'preset-button';
        loadBtn.textContent = 'Load';
        loadBtn.addEventListener('click', () => this.loadPreset());
        presetsSection.appendChild(loadBtn);

        const presetInput = document.createElement('input');
        presetInput.className = 'preset-input';
        presetInput.type = 'text';
        presetInput.placeholder = 'Preset Name';
        this.presetInput = presetInput;
        presetsSection.appendChild(presetInput);

        const saveBtn = document.createElement('button');
        saveBtn.className = 'preset-button';
        saveBtn.textContent = 'Save';
        saveBtn.addEventListener('click', () => this.savePreset());
        presetsSection.appendChild(saveBtn);
        
        const collapseBtn = document.createElement('div');
        collapseBtn.className = 'fluxaio-collapse-btn';
        collapseBtn.textContent = '−';
        collapseBtn.title = 'Collapse/Expand';
        collapseBtn.addEventListener('click', () => this.toggleCollapse());
        presetsSection.appendChild(collapseBtn);

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'preset-button';
        deleteBtn.textContent = 'Delete';
        deleteBtn.addEventListener('click', () => this.deletePreset());
        presetsSection.appendChild(deleteBtn);

        parent.appendChild(presetsSection);
    }

    updatePresetDropdown() {
        if (!this.presetSelect) return;
        
        this.presetSelect.innerHTML = '';
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'Select Preset';
        this.presetSelect.appendChild(defaultOption);

        Object.keys(this.presets).sort().forEach(presetName => {
            const option = document.createElement('option');
            option.value = presetName;
            option.textContent = presetName;
            this.presetSelect.appendChild(option);
        });
    }

	savePreset() {
		const presetName = this.presetInput.value.trim();
		if (!presetName) {
			alert('Please enter a preset name');
			return;
		}

		const presetData = {};
		
        // Save all widget values
        if (this.node.widgets) {
            this.node.widgets.forEach(widget => {
                if (widget.name && widget.value !== undefined) {
                    presetData[widget.name] = widget.value;
                }
            });
        }

        // Save custom UI state
		presetData.fluxAIOUIData = this.serialize();

		this.presets[presetName] = presetData;
		this.savePresets();
		this.updatePresetDropdown();
		this.presetSelect.value = presetName;
		this.presetInput.value = '';
		
		console.log(`[FluxAIOUI] Saved preset: ${presetName}`);
	}

	loadPreset() {
		const presetName = this.presetSelect.value;
		if (!presetName || !this.presets[presetName]) {
			alert('Please select a valid preset');
			return;
		}

		const presetData = this.presets[presetName];
		console.log(`[FluxAIOUI] Loading preset: ${presetName}`);
		
        // Restore all widget values
		Object.entries(presetData).forEach(([paramName, value]) => {
            if (paramName !== 'fluxAIOUIData') {
                this.setWidgetValue(paramName, value, true);
            }
		});

        // Restore custom UI state
        if (presetData.fluxAIOUIData) {
            this.deserialize(presetData.fluxAIOUIData);
        }

        // Final sync to ensure UI reflects all changes
		setTimeout(() => {
			this.syncAllUIToWidgetState();
			this.node.setDirtyCanvas(true, true);
			console.log(`[FluxAIOUI] Preset loaded and UI synced: ${presetName}`);
		}, 50);
	}

    deletePreset() {
        const presetName = this.presetSelect.value;
        if (!presetName) {
            alert('Please select a preset to delete');
            return;
        }

        if (confirm(`Delete preset "${presetName}"?`)) {
            delete this.presets[presetName];
            this.savePresets();
            this.updatePresetDropdown();
            console.log(`[FluxAIOUI] Deleted preset: ${presetName}`);
        }
    }

    createTabs() {
        const tabsContainer = document.createElement('div');
        tabsContainer.className = 'fluxaio-tabs';
        
        const tabs = [
            { id: 'models', label: 'MODELS' }, { id: 'performance', label: 'PERF' },
            { id: 'prompt', label: 'PROMPT' }, { id: 'lora', label: 'LORA' },
            { id: 'inference', label: 'INFERENCE' }, { id: 'postprocess', label: 'POST' },
            { id: 'gallery', label: 'GALLERY' }, { id: 'save', label: 'SAVE' }
        ];
        
        tabs.forEach(({ id, label }) => {
            const tab = document.createElement('button');
            tab.className = 'fluxaio-tab';
            
            const tabText = document.createElement('span');
            tabText.textContent = label;
            tab.appendChild(tabText);
            
            const resetBtn = document.createElement('button');
            resetBtn.className = 'tab-reset-btn';
            resetBtn.textContent = '↻';
            resetBtn.title = `Reset ${label} tab`;
            resetBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.resetTab(id);
            });
            tab.appendChild(resetBtn);
            
            tab.dataset.type = id;
            
            if (id === this.activeTab) tab.classList.add('active');
            
            tab.addEventListener('click', () => this.switchTab(id));
            tabsContainer.appendChild(tab);
        });
        
        this.container.appendChild(tabsContainer);
    }

    resetTab(tabId) {
        const resets = {
            'models': this.resetModelsTab.bind(this),
            'performance': this.resetPerformanceTab.bind(this),
            'prompt': this.resetPromptTab.bind(this),
            'lora': this.resetLoRATab.bind(this),
            'inference': this.resetInferenceTab.bind(this),
            'postprocess': this.resetPostProcessTab.bind(this),
            'gallery': this.resetGalleryTab.bind(this),
            'save': this.resetSaveTab.bind(this)
        };
        if (resets[tabId]) resets[tabId]();
    }

	resetInferenceTab() {
		const defaults = {
			'flux_guidance': 2.5,
			'resolution_preset': '832x1216 (2:3 Portrait)',
			'sampler_name': 'deis',
			'scheduler': 'beta',
            'steps': 28,
            
			'enable_img2img': false,
			'enable_style_model': false,
			'img2img_denoise': 0.8,
			'style_strength': 1.0,
			'strength_type': 'multiply',
			'crop': 'none',

			'enable_latent_injection': true,
			'injection_point': 0.75,
			'injection_seed_offset': 1,
			'injection_strength': 0.3,
			'normalize_injected_noise': 'enable',

			'enable_upscale_with_model': true,
			'enable_2nd_pass': true,
			'downscale_by': 0.5,
			'precision': 'bf16',
			'steps_2ND': 20,
			'denoise_2ND': 0.3,
			'seed_shift_2ND': 1,
            
			'enable_tiling': true,
			'tiles': '2x2 (4)',
			'tile_padding': 32,
			'mask_blur': 16,

			'enable_face_enhancement': false,
            'face_resize_back': false,
            'face_initial_resize': 4096,
            'face_upscale_res': 1536,
            'face_bbox_threshold': 0.5,
            'face_segm_threshold': 0.5,
            'face_mask_expand': 16,
            'face_mask_blur': 16.0,
            'face_steps': 20,
            'face_seed_shift': 1,
            'face_padding': 64,
            'face_cnet_strength': 0.7,
            'face_cnet_end': 0.7,
            'face_color_match_strength': 1.0,
            'enable_image_mix': false,
            'image_mix_factor': 0.5,
		};

		Object.entries(defaults).forEach(([param, value]) => this.setWidgetValue(param, value));
		
        this.syncAllUIToWidgetState();
	}

	resetModelsTab() {
		const defaults = {
			'flux_model_name': "flux1-dev-fp8.safetensors",
			'vae_name': "ae.safetensors",
			'clip_l_name': "clip_l.safetensors",
			't5_name': "t5xxl_fp16.safetensors",
			'cnet_model_name': 'Flux.1-dev-Controlnet-Upscaler.safetensors',
			'style_model_name': 'flux1-redux-dev.safetensors',
			'clip_vision_name': 'sigclip_vision_patch14_384.safetensors',
			'upscale_model_name': '4x_foolhardy_Remacri.pth',
			'face_bbox_model': 'bbox/face_yolov8m.pt',
			'face_segm_model': 'segm/face_yolov8n-seg2_60.pt'
		};
		Object.entries(defaults).forEach(([param, value]) => {
			const modelList = this.getModelList(param);
			if (modelList.includes(value)) {
				this.setWidgetValue(param, value);
			} else if (modelList.length > 0) {
				this.setWidgetValue(param, modelList[0]);
			}
		});
		this.syncAllUIToWidgetState();
	}

    resetPerformanceTab() {
        const defaults = {
            'enable_torch_compile': false, 'enable_teacache': true,
            'enable_lora_block_patcher': false, 'compile_backend': 'inductor',
            'full_load': 'auto', 'enable_sage_attention': true, 'sage_attention_mode': 'auto',
            'teacache_rel_l1_thresh': 0.3, 'teacache_start_percent': 0.2, 'teacache_end_percent': 1.0,
            'teacache_2nd_rel_l1_thresh': 0.3, 'teacache_2nd_start_percent': 0.0, 'teacache_2nd_end_percent': 1.0,
            'compile_mode': 'default', 'compile_fullgraph': false, 'compile_dynamic': false
        };
        Object.entries(defaults).forEach(([param, value]) => this.setWidgetValue(param, value));
        
        if(this.perfSwitch) {
            this.perfSwitch.setState('teacache', true);
        }
        this.syncAllUIToWidgetState();
    }

	resetPromptTab() {
		const defaults = {
			'enable_florence2': false, 'florence2_model': 'microsoft/Florence-2-base',
			'florence2_task': 'caption', 'florence2_precision': 'fp16',
			'florence2_attention': 'sdpa', 'florence2_max_new_tokens': 1024,
			'florence2_num_beams': 3, 'florence2_seed': 1, 'florence2_do_sample': true
		};
		Object.entries(defaults).forEach(([param, value]) => this.setWidgetValue(param, value));
		
		this.additionalPromptText = '';
		this.originalBasePrompt = null;
		if (this.additionalPromptTextarea) {
			this.additionalPromptTextarea.value = '';
		}
		
		this.syncAllUIToWidgetState();
	}

	resetLoRATab() {
		for (let i = 1; i <= 6; i++) {
			this.setWidgetValue(`lora_${i}_name`, 'None');
			this.setWidgetValue(`lora_${i}_strength`, 1.0);
			this.setWidgetValue(`lora_${i}_clip_strength`, 1.0);
		}
		
		this.setWidgetValue('enable_lora_stack', false);
		
		this.updateLoRAStaticSlotsDisplay();
		
		this.loraBlockValues = {
			double_blocks: Array(19).fill(1.0), 
			single_blocks: Array(38).fill(1.0),
		};
		this.setLoRABlocks('double_blocks', 1.0);
		this.setLoRABlocks('single_blocks', 1.0);
		this.updateLoRABlocksUI();
		this.setWidgetValue('enable_lora_block_patcher', false);
		this.syncAllUIToWidgetState();
	}

    resetPostProcessTab() {
        Object.values(POSTPROCESS_OPERATIONS).forEach(op => {
            this.setWidgetValue(op.enable, false);
            if (op.params) op.params.forEach(p => this.setWidgetValue(p.name, p.default));
            if (op.subgroups) Object.values(op.subgroups).flat().forEach(p => this.setWidgetValue(p.name, p.default));
        });
        this.syncAllUIToWidgetState();
    }

    resetGalleryTab() {
        this.clearGallery();
        this.setGalleryMode('gallery');
    }

    resetSaveTab() {
        this.setWidgetValue('enable_save_image', true);
        this.setWidgetValue('filename_prefix', 'flux_aio/image');
        this.syncAllUIToWidgetState();
    }

    createModelsSection(parent) {
        const section = document.createElement('div');
        section.id = 'models-section';
        section.className = 'fluxaio-section';

        const title = document.createElement('div');
        title.className = 'section-title';
        title.textContent = 'Model Configuration';
        section.appendChild(title);

        const grid = document.createElement('div');
        grid.className = 'control-grid-2';

		const modelSelectors = [
			{ name: 'flux_model_name', label: 'Flux Model' },
			{ name: 'vae_name', label: 'VAE Model' },
			{ name: 'clip_l_name', label: 'CLIP L Model' },
			{ name: 't5_name', label: 'T5-XXL Model' },
			{ name: 'style_model_name', label: 'Style Model (Redux)' },
			{ name: 'clip_vision_name', label: 'CLIP Vision Model' },
			{ name: 'upscale_model_name', label: 'Upscale Model' },
			{ name: 'flux_cnet_upscaler_model', label: 'Flux ControlNet Upscaler' },
			{ name: 'face_bbox_model', label: 'Face BBox Model' },
			{ name: 'face_segm_model', label: 'Face Segmentation Model' }
		];
		
        modelSelectors.forEach(selector => {
            const group = this.createControlGroup(selector.label);
            const select = this.createSelect(selector.name, this.getModelOptions(selector.name));
            group.appendChild(select);
            grid.appendChild(group);
        });
        
        section.appendChild(grid);
        parent.appendChild(section);
    }
    
    syncAllSelects() {
        this.container.querySelectorAll('select.param-select').forEach(select => {
            const paramName = select.dataset.paramName;
            if (paramName) {
                const options = this.getModelOptions(paramName);
                const currentValue = select.value;
                select.innerHTML = '';
                options.forEach(opt => {
                    const optionEl = document.createElement('option');
                    optionEl.value = opt;
                    optionEl.textContent = opt;
                    select.appendChild(optionEl);
                });
                if (options.includes(currentValue)) {
                    select.value = currentValue;
                } else if (options.length > 0) {
                    select.value = options[0];
                    this.setWidgetValue(paramName, options[0]);
                }
            }
        });
    }

	getModelList(paramName) {
		if (!this.modelLists) return ['Loading...'];
		const mapping = {
			'flux_model_name': 'flux_models', 
			'vae_name': 'vae_models',
			'clip_l_name': 'text_encoders', 
			't5_name': 'text_encoders',
			'style_model_name': 'style_models', 
			'clip_vision_name': 'clip_vision',
			'upscale_model_name': 'upscale_models',
			'flux_cnet_upscaler_model': 'controlnet_models',
			'face_bbox_model': 'bbox_models',
			'face_segm_model': 'segm_models'
		};
		return this.modelLists[mapping[paramName]] || ['No models found'];
	}

    getModelOptions(paramName) {
        // First, check for static, non-model options
        const staticOptions = {
            'sampler_name': SAMPLERS,
            'scheduler': SCHEDULERS,
            'resolution_preset': RESOLUTION_PRESETS
        };
        if(staticOptions[paramName]) return staticOptions[paramName];

        // Next, check for model-related options by calling getModelList.
        // This is the key part of the fix.
        const modelList = this.getModelList(paramName);
        if (modelList && !(modelList.length === 1 && modelList[0].includes('No models found'))) {
            return modelList;
        }

        // Finally, fall back to other parameter options
        return PARAMETER_OPTIONS[paramName] || ['No models found'];
    }

	createPerformanceSection(parent) {
		const section = document.createElement('div');
		section.id = 'performance-section';
		section.className = 'fluxaio-section';

		const title = document.createElement('div');
		title.className = 'section-title';
		title.textContent = 'Performance & Advanced Options';
		section.appendChild(title);
		
		const mainControlGroup = this.createControlGroup('Optimization Mode');
		this.perfSwitch = this.createThreeStateSwitch(
			'enable_torch_compile', 
			'enable_teacache',
			['Compile', 'Off', 'TeaCache']
		);
		mainControlGroup.appendChild(this.perfSwitch.element);
		section.appendChild(mainControlGroup);

		// --- Sage Attention Options (Visible in Off & TeaCache states) ---
		const sageOptions = document.createElement('div');
		sageOptions.id = 'sage-options-container';
		sageOptions.className = 'perf-options-container';
		const sageGrid = document.createElement('div');
		sageGrid.className = 'control-grid-2';
		sageGrid.appendChild(this.createToggle('enable_sage_attention', 'Sage Attention', { default: true }));
		const sageModeGroup = this.createControlGroup('Sage Mode');
		sageModeGroup.appendChild(this.createSelect('sage_attention_mode', ['auto', 'sageattn_qk_int8_pv_fp16_cuda', 'sageattn_qk_int8_pv_fp16_triton', 'sageattn_qk_int8_pv_fp8_cuda']));
		sageGrid.appendChild(sageModeGroup);
		sageOptions.appendChild(sageGrid);
		section.appendChild(sageOptions);

		// --- Torch Compile Options (Visible only in Compile state) ---
		const compileOptions = document.createElement('div');
		compileOptions.id = 'compile-options-container';
		compileOptions.className = 'perf-options-container';
		const compileGrid = document.createElement('div');
		compileGrid.className = 'control-grid-2';
		
		const backendGroup = this.createControlGroup('Backend');
		backendGroup.appendChild(this.createSelect('compile_backend', PARAMETER_OPTIONS['compile_backend']));
		compileGrid.appendChild(backendGroup);
		
		const modeGroup = this.createControlGroup('Mode');
		modeGroup.appendChild(this.createSelect('compile_mode', PARAMETER_OPTIONS['compile_mode']));
		compileGrid.appendChild(modeGroup);

		compileGrid.appendChild(this.createToggle('compile_fullgraph', 'Full Graph', {default: false}));
		compileGrid.appendChild(this.createToggle('compile_dynamic', 'Dynamic', {default: false}));
		
		const fullLoadGroup = this.createControlGroup('Full Load (Compile)');
		fullLoadGroup.appendChild(this.createSelect('full_load', PARAMETER_OPTIONS['full_load']));
		compileGrid.appendChild(fullLoadGroup);

		compileOptions.appendChild(compileGrid);
		section.appendChild(compileOptions);

		// --- TeaCache Options (Visible only in TeaCache state) ---
		const teacacheOptions = document.createElement('div');
		teacacheOptions.id = 'teacache-options-container';
		teacacheOptions.className = 'perf-options-container';

		const firstPassTea = this.createControlGroup('First Pass TeaCache');
		firstPassTea.appendChild(this.createSlider({ name: 'teacache_rel_l1_thresh', label: 'Threshold', min: 0, max: 1, step: 0.01, default: 0.3, precision: 2 }));
		firstPassTea.appendChild(this.createSlider({ name: 'teacache_start_percent', label: 'Start %', min: 0, max: 1, step: 0.01, default: 0.2, precision: 2 }));
		firstPassTea.appendChild(this.createSlider({ name: 'teacache_end_percent', label: 'End %', min: 0, max: 1, step: 0.01, default: 1.0, precision: 2 }));
		teacacheOptions.appendChild(firstPassTea);

		const secondPassTea = this.createControlGroup('2ND Pass TeaCache');
		secondPassTea.appendChild(this.createSlider({ name: 'teacache_2nd_rel_l1_thresh', label: 'Threshold', min: 0, max: 1, step: 0.01, default: 0.3, precision: 2 }));
		secondPassTea.appendChild(this.createSlider({ name: 'teacache_2nd_start_percent', label: 'Start %', min: 0, max: 1, step: 0.01, default: 0.0, precision: 2 }));
		secondPassTea.appendChild(this.createSlider({ name: 'teacache_2nd_end_percent', label: 'End %', min: 0, max: 1, step: 0.01, default: 1.0, precision: 2 }));
		teacacheOptions.appendChild(secondPassTea);
		
		section.appendChild(teacacheOptions);
		
		parent.appendChild(section);

		this.perfSwitch.syncState();
	}
    
    createThreeStateSwitch(widget1, widget2, labels) {
        const switchContainer = document.createElement('div');
        switchContainer.className = 'three-state-switch';
    
        const thumb = document.createElement('div');
        thumb.className = 'switch-thumb';
    
        const [label1, label2, label3] = labels.map(text => {
            const label = document.createElement('span');
            label.className = 'switch-label';
            label.textContent = text;
            return label;
        });
    
        switchContainer.appendChild(label1);
        switchContainer.appendChild(label2);
        switchContainer.appendChild(label3);
        switchContainer.appendChild(thumb);
    
        let currentState = 'off';
    
        const setState = (newState, supressEvent = false) => {
            currentState = newState;
            switchContainer.classList.remove('state-compile', 'state-off', 'state-teacache');
            switchContainer.classList.add(`state-${newState}`);
    
            label1.classList.toggle('active', newState === 'compile');
            label2.classList.toggle('active', newState === 'off');
            label3.classList.toggle('active', newState === 'teacache');

            if (!supressEvent) {
                this.setWidgetValue(widget1, newState === 'compile');
                this.setWidgetValue(widget2, newState === 'teacache');
            }
            
            const compileOptions = this.container.querySelector('#compile-options-container');
            const teacacheOptions = this.container.querySelector('#teacache-options-container');
            const sageOptions = this.container.querySelector('#sage-options-container');

            if (compileOptions) compileOptions.style.display = (newState === 'compile') ? 'block' : 'none';
            if (teacacheOptions) teacacheOptions.style.display = (newState === 'teacache') ? 'block' : 'none';
            if (sageOptions) sageOptions.style.display = (newState === 'compile') ? 'none' : 'block';
        };
    
        switchContainer.addEventListener('click', () => {
            const nextState = {
                'compile': 'teacache',
                'teacache': 'off',
                'off': 'compile'
            }[currentState];
            setState(nextState);
        });
    
        const syncState = () => {
            const val1 = this.getWidgetValue(widget1, false);
            const val2 = this.getWidgetValue(widget2, false);
            if (val1) setState('compile', true);
            else if (val2) setState('teacache', true);
            else setState('off', true);
        };
    
        return { element: switchContainer, setState, syncState };
    }
    
    createPromptSection(parent) {
        const section = document.createElement('div');
        section.id = 'prompt-section';
        section.className = 'fluxaio-section';

        const title = document.createElement('div');
        title.className = 'section-title';
        title.textContent = 'Prompt Management (Florence-2)';
        section.appendChild(title);

        const florenceControlsContainer = document.createElement('div');
        
        const florenceGrid = document.createElement('div');
        florenceGrid.className = 'control-grid-5';
        
        const florenceToggle = this.createToggle('enable_florence2', 'Enable', {default: false});
        florenceGrid.appendChild(florenceToggle);
        
        const florenceModelGroup = this.createControlGroup('Model');
        florenceModelGroup.appendChild(this.createSelect('florence2_model', this.getModelOptions('florence2_model')));
        florenceGrid.appendChild(florenceModelGroup);
        
        const florenceTaskGroup = this.createControlGroup('Task');
        florenceTaskGroup.appendChild(this.createSelect('florence2_task', this.getModelOptions('florence2_task')));
        florenceGrid.appendChild(florenceTaskGroup);
        
        const florencePrecisionGroup = this.createControlGroup('Precision');
        florencePrecisionGroup.appendChild(this.createSelect('florence2_precision', this.getModelOptions('florence2_precision')));
        florenceGrid.appendChild(florencePrecisionGroup);
        
        const florenceAttentionGroup = this.createControlGroup('Attention');
        florenceAttentionGroup.appendChild(this.createSelect('florence2_attention', this.getModelOptions('florence2_attention')));
        florenceGrid.appendChild(florenceAttentionGroup);
        
        florenceControlsContainer.appendChild(florenceGrid);

        const florenceParamsGrid = document.createElement('div');
        florenceParamsGrid.className = 'control-grid-3';
        const florenceParams = [
            { name: 'florence2_max_new_tokens', label: 'Max Tokens', min: 1, max: 4096, step: 1, default: 1024, precision: 0 },
            { name: 'florence2_num_beams', label: 'Num Beams', min: 1, max: 64, step: 1, default: 3, precision: 0 },
            { name: 'florence2_seed', label: 'Seed', min: 0, max: 999999, step: 1, default: 1, precision: 0 }
        ];
        florenceParams.forEach(param => florenceParamsGrid.appendChild(this.createSlider(param)));
        
        florenceControlsContainer.appendChild(florenceParamsGrid);
        section.appendChild(florenceControlsContainer);
        parent.appendChild(section);
    }
    
	createLoRASection(parent) {
		const section = document.createElement('div');
		section.id = 'lora-section';
		section.className = 'fluxaio-section';
		
		const title = document.createElement('div');
		title.className = 'section-title';
		title.textContent = 'LoRA Management';
		section.appendChild(title);
		
		const stackContainer = this.createControlGroup('');
		stackContainer.style.marginBottom = '3px';
		
		const enableToggle = this.createToggle('enable_lora_stack', 'Enable LoRA Stack', {default: false});
		stackContainer.appendChild(enableToggle);
		
		const searchInput = document.createElement('input');
		searchInput.type = 'text';
		searchInput.className = 'lora-search-input';
		searchInput.placeholder = 'Search LoRAs...';
		searchInput.addEventListener('input', (e) => {
			this.filterLoRAList(e.target.value);
		});
		searchInput.addEventListener('keydown', (e) => {
			this.handleLoRAKeyboard(e);
		});
		stackContainer.appendChild(searchInput);

		const loraListContainer = document.createElement('div');
		loraListContainer.className = 'lora-select-list';
		stackContainer.appendChild(loraListContainer);

        const refreshBtn = document.createElement('button');
        refreshBtn.className = 'preset-button';
        refreshBtn.textContent = '🔄 Refresh LoRA List';
        refreshBtn.title = 'Refresh LoRA List';
        refreshBtn.style.width = '100%';
        refreshBtn.style.marginBottom = '12px';
        refreshBtn.addEventListener('click', async () => {
            refreshBtn.textContent = '⟳ Loading...';
            await this.fetchLoRAList();
            refreshBtn.textContent = '🔄 Refresh LoRA List';
        });
        stackContainer.appendChild(refreshBtn);
		
		const slotsContainer = document.createElement('div');
		slotsContainer.id = 'lora-slots-container';
		slotsContainer.style.minHeight = '60px'; 
		slotsContainer.style.border = '1px dashed var(--primary-accent)';
		slotsContainer.style.borderRadius = '6px'; 
		slotsContainer.style.padding = '8px';
		slotsContainer.style.background = 'rgba(255, 255, 255, 0.02)';
		this.loraStackContainer = slotsContainer;
		stackContainer.appendChild(slotsContainer);
		
		section.appendChild(stackContainer);
		
		const blockPatcherSection = document.createElement('div');
		blockPatcherSection.className = 'lora-blocks-container';
		const blockPatcherTitle = document.createElement('div');
		blockPatcherTitle.className = 'control-group-title';
		blockPatcherTitle.textContent = 'LoRA Block Patcher';
		blockPatcherSection.appendChild(blockPatcherTitle);
		blockPatcherSection.appendChild(this.createToggle('enable_lora_block_patcher', 'Enable Block Patcher', {default: false}));
		
		const blockTabs = document.createElement('div');
		blockTabs.className = 'lora-blocks-tabs';
		const doubleTab = document.createElement('button');
		doubleTab.className = 'lora-blocks-tab active'; 
		doubleTab.textContent = 'DOUBLE (0-18)';
		doubleTab.addEventListener('click', () => this.switchLoRABlocksTab('double_blocks'));
		blockTabs.appendChild(doubleTab);
		const singleTab = document.createElement('button');
		singleTab.className = 'lora-blocks-tab'; 
		singleTab.textContent = 'SINGLE (0-37)';
		singleTab.addEventListener('click', () => this.switchLoRABlocksTab('single_blocks'));
		blockTabs.appendChild(singleTab);
		blockPatcherSection.appendChild(blockTabs);
		
		this.createLoRABlocksGrid(blockPatcherSection, 'double_blocks', 19);
		this.createLoRABlocksGrid(blockPatcherSection, 'single_blocks', 38);
		
		console.log('[FluxAIOUI] LoRA blocks grids created');
		section.appendChild(blockPatcherSection);
		parent.appendChild(section);
		
		this.updateLoRAStaticSlotsDisplay();
	}

	createLoRABlocksGrid(parent, type, count) {
		const controlsContainer = document.createElement('div');
		controlsContainer.className = 'lora-blocks-controls';
		controlsContainer.id = `lora-blocks-controls-${type}`;
		controlsContainer.style.display = type === 'double_blocks' ? 'flex' : 'none';
		
		const minGroup = this.createControlGroupNoTitle();
		minGroup.innerHTML = `<span>Min:</span>`;
		const minInput = document.createElement('input');
		minInput.className = 'lora-blocks-input'; 
		minInput.type = 'number';
		minInput.min = -2; 
		minInput.max = 2; 
		minInput.step = 0.1;
		minInput.value = this.loraBlocksSettings[type].min;
		minInput.addEventListener('input', (e) => this.loraBlocksSettings[type].min = parseFloat(e.target.value));
		minGroup.appendChild(minInput);
		controlsContainer.appendChild(minGroup);
		
		const maxGroup = this.createControlGroupNoTitle();
		maxGroup.innerHTML = `<span>Max:</span>`;
		const maxInput = document.createElement('input');
		maxInput.className = 'lora-blocks-input'; 
		maxInput.type = 'number';
		maxInput.min = -2; 
		maxInput.max = 2; 
		maxInput.step = 0.1;
		maxInput.value = this.loraBlocksSettings[type].max;
		maxInput.addEventListener('input', (e) => this.loraBlocksSettings[type].max = parseFloat(e.target.value));
		maxGroup.appendChild(maxInput);
		controlsContainer.appendChild(maxGroup);
		
		const selectedValueGroup = this.createControlGroupNoTitle();
		selectedValueGroup.innerHTML = `<span id="lora-block-selected-label-${type}">Selected Block: (-)</span>`;
		const valueInput = document.createElement('input');
		valueInput.id = `lora-block-value-input-${type}`; 
		valueInput.className = 'lora-blocks-input';
		valueInput.type = 'number'; 
		valueInput.step = 0.01; 
		valueInput.disabled = true;
		valueInput.addEventListener('change', (e) => {
			if(this.selectedLoRABlock.type === type && this.selectedLoRABlock.index !== -1) {
				const newValue = parseFloat(e.target.value);
				this.loraBlockValues[type][this.selectedLoRABlock.index] = newValue;
				this.updateLoRABlocksUI();
				this.syncLoRABlockToWidget(this.selectedLoRABlock.index, newValue, type);
			}
		});
		selectedValueGroup.appendChild(valueInput);
		controlsContainer.appendChild(selectedValueGroup);

		const randomBtn = document.createElement('button');
		randomBtn.className = 'lora-blocks-button'; 
		randomBtn.textContent = 'Random';
		randomBtn.addEventListener('click', () => this.randomizeLoRABlocks(type));
		controlsContainer.appendChild(randomBtn);
		
		const zeroBtn = document.createElement('button');
		zeroBtn.className = 'lora-blocks-button danger'; 
		zeroBtn.textContent = 'Zero';
		zeroBtn.addEventListener('click', () => this.setLoRABlocks(type, 0.0));
		controlsContainer.appendChild(zeroBtn);
		
		const resetBtn = document.createElement('button');
		resetBtn.className = 'lora-blocks-button'; 
		resetBtn.textContent = 'Reset';
		resetBtn.addEventListener('click', () => this.setLoRABlocks(type, 1.0));
		controlsContainer.appendChild(resetBtn);
		
		parent.appendChild(controlsContainer);
		
		const grid = document.createElement('div');
		grid.className = 'lora-blocks-grid';
		grid.id = `lora-blocks-${type}`;
		grid.style.display = type === 'double_blocks' ? 'flex' : 'none';
		
		if (type === 'double_blocks') {
			const row = document.createElement('div'); 
			row.className = 'lora-blocks-row';
			for (let i = 0; i < count; i++) {
				row.appendChild(this.createLoRABlockSlider(i, type));
			}
			grid.appendChild(row);
		} else {
			const row1 = document.createElement('div'); 
			row1.className = 'lora-blocks-row';
			for (let i = 0; i < 19; i++) {
				row1.appendChild(this.createLoRABlockSlider(i, type));
			}
			grid.appendChild(row1);
			
			const row2 = document.createElement('div'); 
			row2.className = 'lora-blocks-row';
			for (let i = 19; i < count; i++) {
				row2.appendChild(this.createLoRABlockSlider(i, type));
			}
			grid.appendChild(row2);
		}
		parent.appendChild(grid);
	}
    
	createLoRABlockSlider(index, type) {
		const sliderContainer = document.createElement('div');
		sliderContainer.className = 'lora-block-slider-container';
		sliderContainer.dataset.index = index;
		sliderContainer.dataset.type = type;

		const slider = document.createElement('input');
		slider.type = 'range'; 
		slider.className = 'lora-block-slider';
		slider.min = 0; 
		slider.max = 2; 
		slider.step = 0.01;
		slider.value = this.loraBlockValues[type][index] ?? 1.0;

		slider.addEventListener('input', (e) => {
			const value = parseFloat(e.target.value);
			this.loraBlockValues[type][index] = value;
			this.syncLoRABlockToWidget(index, value, type);
			this.updateLoRABlockVisualState(sliderContainer, slider);
			this.updateSelectedBlockValueInput(type, index, value);
		});
		
		sliderContainer.addEventListener('click', () => {
			this.setSelectedLoRABlock(type, index);
		});

		const label = document.createElement('span');
		label.className = 'lora-block-label';
		label.textContent = `${index}`;
		
		sliderContainer.appendChild(slider);
		sliderContainer.appendChild(label);
		
		return sliderContainer;
	}

	addLoRAToStaticSlot(loraName) {
		if (!loraName || loraName === "None" || loraName.trim() === '') {
			return;
		}
		if (loraName.includes('Error') || loraName.includes('No LoRAs found')) {
			return;
		}
		
		for (let i = 1; i <= 6; i++) {
			if (this.getWidgetValue(`lora_${i}_name`, 'None') === 'None') {
				this.setWidgetValue(`lora_${i}_name`, loraName, true);
				this.setWidgetValue(`lora_${i}_strength`, 1.0, true);
				this.setWidgetValue(`lora_${i}_clip_strength`, 1.0, true);
				this.setWidgetValue('enable_lora_stack', true);

				console.log(`[FluxAIOUI] Added LoRA "${loraName}" to slot ${i}`);
				
				this.updateLoRAStaticSlotsDisplay(); 
				this.numberInputs['enable_lora_stack'].update(true);
				return;
			}
		}
		alert('All LoRA slots are full!');
		const searchInput = this.container?.querySelector('.lora-search-input');
		if (searchInput) {
			searchInput.value = '';
			this.filterLoRAList('');
	    }
    }

	updateLoRAStaticSlotsDisplay() {
		const container = this.container.querySelector('#lora-slots-container');
		if (!container) return;

		let hasActiveLoRAs = false;
		const activeSlots = new Set();

		container.querySelectorAll('.lora-slot-container').forEach(el => {
			const slotIndex = parseInt(el.dataset.slotIndex, 10);
			const loraName = this.getWidgetValue(`lora_${slotIndex}_name`, 'None');

			if (loraName === 'None') {
				el.remove();
			} else {
				activeSlots.add(slotIndex);
				hasActiveLoRAs = true;
			}
		});

		for (let i = 1; i <= 6; i++) {
			if (!activeSlots.has(i)) {
				const loraName = this.getWidgetValue(`lora_${i}_name`, 'None');
				if (loraName !== 'None') {
					const newSlotElement = this.createLoRASlotElement(i, loraName);
					container.appendChild(newSlotElement);
					hasActiveLoRAs = true;
				}
			}
		}

		let emptyMsg = container.querySelector('.lora-empty-message');
		if (hasActiveLoRAs) {
			if (emptyMsg) emptyMsg.remove();
		} else {
			if (!emptyMsg) {
				emptyMsg = document.createElement('div');
				emptyMsg.className = 'lora-empty-message';
				emptyMsg.textContent = 'No LoRAs added yet - use the Add button above';
				Object.assign(emptyMsg.style, {
					textAlign: 'center', color: 'var(--text-gray)',
					fontStyle: 'italic', padding: '20px'
				});
				container.appendChild(emptyMsg);
			}
		}
	}

	createLoRASlotElement(slotIndex, loraName) {
		const slotContainer = document.createElement('div');
		slotContainer.className = 'lora-slot-container';
		slotContainer.dataset.slotIndex = slotIndex;
		slotContainer.style.display = 'grid';
		slotContainer.style.gridTemplateColumns = '40px 2fr 1fr 1fr 30px';
		slotContainer.style.gap = '8px';
		slotContainer.style.alignItems = 'center';
		slotContainer.style.padding = '8px';
		slotContainer.style.background = 'rgba(var(--primary-accent-rgb), 0.1)';
		slotContainer.style.border = '1px solid rgba(var(--primary-accent-rgb), 0.3)';
		slotContainer.style.borderRadius = '6px';
		slotContainer.style.marginBottom = '4px';
		slotContainer.style.transition = 'all 0.3s ease';
		
		const slotLabel = document.createElement('div');
		slotLabel.textContent = `${slotIndex}:`;
		slotLabel.style.color = 'var(--primary-accent)';
		slotLabel.style.fontSize = '12px';
		slotLabel.style.fontFamily = "'Orbitron', monospace";
		slotLabel.style.fontWeight = '600';
		slotLabel.style.textAlign = 'center';
		
		const nameDisplay = document.createElement('div');
		nameDisplay.textContent = loraName;
		nameDisplay.style.color = 'var(--primary-accent)';
		nameDisplay.style.fontSize = '11px';
		nameDisplay.style.fontFamily = "'Orbitron', monospace";
		nameDisplay.style.overflow = 'hidden';
		nameDisplay.style.textOverflow = 'ellipsis';
		nameDisplay.style.whiteSpace = 'nowrap';
		nameDisplay.title = loraName;
		
		const modelStrengthInput = document.createElement('input');
		modelStrengthInput.type = 'number';
		modelStrengthInput.className = 'lora-blocks-input';
		modelStrengthInput.min = -10;
		modelStrengthInput.max = 10;
		modelStrengthInput.step = 0.01;
		modelStrengthInput.value = this.getWidgetValue(`lora_${slotIndex}_strength`, 1.0);
		modelStrengthInput.placeholder = 'Model';
		modelStrengthInput.addEventListener('input', (e) => {
			this.setWidgetValue(`lora_${slotIndex}_strength`, parseFloat(e.target.value) || 1.0);
		});
		
		const clipStrengthInput = document.createElement('input');
		clipStrengthInput.type = 'number';
		clipStrengthInput.className = 'lora-blocks-input';
		clipStrengthInput.min = -10;
		clipStrengthInput.max = 10;
		clipStrengthInput.step = 0.01;
		clipStrengthInput.value = this.getWidgetValue(`lora_${slotIndex}_clip_strength`, 1.0);
		clipStrengthInput.placeholder = 'CLIP';
		clipStrengthInput.addEventListener('input', (e) => {
			this.setWidgetValue(`lora_${slotIndex}_clip_strength`, parseFloat(e.target.value) || 1.0);
		});
		
		const removeBtn = document.createElement('button');
		removeBtn.style.background = 'var(--inactive-red)';
		removeBtn.style.border = 'none';
		removeBtn.style.borderRadius = '3px';
		removeBtn.style.color = 'white';
		removeBtn.style.cursor = 'pointer';
		removeBtn.style.fontSize = '12px';
		removeBtn.style.width = '20px';
		removeBtn.style.height = '20px';
		removeBtn.style.transition = 'all 0.2s ease';
		removeBtn.textContent = '×';
		removeBtn.addEventListener('click', () => {
			this.setWidgetValue(`lora_${slotIndex}_name`, 'None', true);
			this.setWidgetValue(`lora_${slotIndex}_strength`, 1.0, true);
			this.setWidgetValue(`lora_${slotIndex}_clip_strength`, 1.0, true);

			this.updateLoRAStaticSlotsDisplay();
			
			let hasActiveLoRAs = false;
			for (let i = 1; i <= 6; i++) {
				if (this.getWidgetValue(`lora_${i}_name`, 'None') !== 'None') {
					hasActiveLoRAs = true;
					break;
				}
			}
			this.setWidgetValue('enable_lora_stack', hasActiveLoRAs);
			this.numberInputs['enable_lora_stack'].update(hasActiveLoRAs);
		});
		removeBtn.addEventListener('mouseenter', () => {
			removeBtn.style.background = '#e74c3c';
			removeBtn.style.transform = 'scale(1.1)';
		});
		removeBtn.addEventListener('mouseleave', () => {
			removeBtn.style.background = 'var(--inactive-red)';
			removeBtn.style.transform = 'scale(1)';
		});
		
		slotContainer.appendChild(slotLabel);
		slotContainer.appendChild(nameDisplay);
		slotContainer.appendChild(modelStrengthInput);
		slotContainer.appendChild(clipStrengthInput);
		slotContainer.appendChild(removeBtn);
		
		return slotContainer;
	}

	setSelectedLoRABlock(type, index) {
        this.selectedLoRABlock = { type, index };
        this.updateLoRABlocksUI(); // To update visual selection
        const value = this.loraBlockValues[type][index];
        this.updateSelectedBlockValueInput(type, index, value);
    }

    updateSelectedBlockValueInput(type, index, value) {
        const label = this.container.querySelector(`#lora-block-selected-label-${type}`);
        const input = this.container.querySelector(`#lora-block-value-input-${type}`);
        if(label && input) {
            label.textContent = `Selected Block: (${index})`;
            input.value = (value ?? 1.0).toFixed(2);
            input.disabled = false;
        }
    }
    
    randomizeLoRABlocks(type) {
        const min = this.loraBlocksSettings[type].min;
        const max = this.loraBlocksSettings[type].max;
        
        for (let i = 0; i < this.loraBlockValues[type].length; i++) {
            const randomValue = Math.random() * (max - min) + min;
            this.loraBlockValues[type][i] = parseFloat(randomValue.toFixed(3));
            this.syncLoRABlockToWidget(i, this.loraBlockValues[type][i], type);
        }
        this.updateLoRABlocksUI();
        if (this.selectedLoRABlock.type === type && this.selectedLoRABlock.index !== -1) {
            this.updateSelectedBlockValueInput(type, this.selectedLoRABlock.index, this.loraBlockValues[type][this.selectedLoRABlock.index]);
        }
    }
    
    setLoRABlocks(type, value) {
        for (let i = 0; i < this.loraBlockValues[type].length; i++) {
            this.loraBlockValues[type][i] = value;
            this.syncLoRABlockToWidget(i, value, type);
        }
        this.updateLoRABlocksUI();
        if (this.selectedLoRABlock.type === type && this.selectedLoRABlock.index !== -1) {
            this.updateSelectedBlockValueInput(type, this.selectedLoRABlock.index, value);
        }
    }
    
    switchLoRABlocksTab(type) {
        this.activeLoRABlocksTab = type;
        
        this.container.querySelectorAll('.lora-blocks-tab').forEach(tab => tab.classList.remove('active'));
        this.container.querySelector(`.lora-blocks-tab:nth-child(${type === 'double_blocks' ? 1 : 2})`).classList.add('active');
        
        this.container.querySelectorAll('.lora-blocks-controls').forEach(controls => controls.style.display = 'none');
        const activeControls = this.container.querySelector(`#lora-blocks-controls-${type}`);
        if (activeControls) activeControls.style.display = 'flex';
        
        this.container.querySelectorAll('.lora-blocks-grid').forEach(grid => grid.style.display = 'none');
        const activeGrid = this.container.querySelector(`#lora-blocks-${type}`);
        if (activeGrid) activeGrid.style.display = 'flex';
        
        this.updateLoRABlocksUI();
    }

    updateLoRABlocksUI() {
        if (!this.container || !this.activeLoRABlocksTab) return;
        const activeGrid = this.container.querySelector(`#lora-blocks-${this.activeLoRABlocksTab}`);
        if (!activeGrid) return;
        
        activeGrid.querySelectorAll('.lora-block-slider-container').forEach(container => {
            const slider = container.querySelector('.lora-block-slider');
            const index = parseInt(container.dataset.index);
            slider.value = this.loraBlockValues[this.activeLoRABlocksTab][index] ?? 1.0;
            this.updateLoRABlockVisualState(container, slider);

            const isSelected = this.selectedLoRABlock.type === this.activeLoRABlocksTab && this.selectedLoRABlock.index === index;
            container.classList.toggle('selected', isSelected);
        });
    }

    updateLoRABlockVisualState(container, sliderElement) {
        const value = parseFloat(sliderElement.value);
        const label = container.querySelector('.lora-block-label');
        const isDefault = Math.abs(value - 1.0) < 0.001;
        
        sliderElement.classList.toggle('non-default', !isDefault);
        if(label) label.classList.toggle('active-label', !isDefault);
    }

	syncLoRABlockToWidget(index, value, type) {
		const widgetName = type === 'double_blocks' ? `lora_block_${index}_double_weight` : `lora_block_${index}_weight`;
		
		let numValue = parseFloat(value);
		if (isNaN(numValue) || value === '' || value === null || value === undefined) {
			numValue = 1.0;
		}
		
		numValue = Math.max(-2.0, Math.min(2.0, numValue));
		
		const widget = this.node.widgets?.find(w => w.name === widgetName);
		if (widget) {
			widget.value = parseFloat(numValue.toFixed(3));
			this.node.setDirtyCanvas(true, true);
		} else {
			console.warn(`[FluxAIOUI] Widget ${widgetName} not found`);
		}
	}

    syncLoRABlocksToWidgets() {
        this.loraBlockValues.double_blocks.forEach((v, i) => this.syncLoRABlockToWidget(i, v, 'double_blocks'));
        this.loraBlockValues.single_blocks.forEach((v, i) => this.syncLoRABlockToWidget(i, v, 'single_blocks'));
    }

	createInferenceSection(parent) {
		const section = document.createElement('div');
		section.id = 'inference-section';
		section.className = 'fluxaio-section';

		const title = document.createElement('div');
		title.className = 'section-title';
		title.textContent = 'Inference Settings';
		section.appendChild(title);

		const globalGrid = document.createElement('div');
		globalGrid.className = 'control-grid-3';
		const resolutionGroup = this.createControlGroup('Resolution');
		resolutionGroup.appendChild(this.createSelect('resolution_preset', this.getModelOptions('resolution_preset')));
		globalGrid.appendChild(resolutionGroup);
		const samplerGroup = this.createControlGroup('Sampler');
		samplerGroup.appendChild(this.createSelect('sampler_name', this.getModelOptions('sampler_name')));
		globalGrid.appendChild(samplerGroup);
		const schedulerGroup = this.createControlGroup('Scheduler');
		schedulerGroup.appendChild(this.createSelect('scheduler', this.getModelOptions('scheduler')));
		globalGrid.appendChild(schedulerGroup);
		section.appendChild(globalGrid);

		section.appendChild(this.createSlider({ 
			name: 'flux_guidance', label: 'Flux Guidance', min: 0, max: 10, step: 0.1, default: 2.5, precision: 1 
		}));

		const subTabsContainer = document.createElement('div');
		subTabsContainer.className = 'inference-sub-tabs';
		const subTabContentContainer = document.createElement('div');
		subTabContentContainer.className = 'inference-sub-content';

		// Sub-tab configuration
		const subTabs = [
			{ id: 'first_pass', label: 'First Pass' },
			{ id: 'second_pass', label: 'Second Pass' },
			{ id: 'face_enhance', label: 'Face Enhance' }
		];
		
		subTabs.forEach(({ id, label }) => {
			const tab = document.createElement('button');
			tab.className = 'inference-sub-tab';
			tab.textContent = label;
			tab.dataset.tabId = id;
			if (id === this.activeInferenceSubTab) tab.classList.add('active');

			tab.addEventListener('click', () => {
				this.activeInferenceSubTab = id;
				subTabsContainer.querySelectorAll('.inference-sub-tab').forEach(t => t.classList.remove('active'));
				tab.classList.add('active');
				subTabContentContainer.querySelectorAll('.inference-sub-section').forEach(s => s.classList.remove('active'));
				const targetSection = subTabContentContainer.querySelector(`#${id}-sub-section`);
				if (targetSection) {
					targetSection.classList.add('active');
				}
			});
			subTabsContainer.appendChild(tab);
		});

		section.appendChild(subTabsContainer);
		section.appendChild(subTabContentContainer);

		// --- First Pass & Noise Injection Content ---
		const firstPassSection = document.createElement('div');
		firstPassSection.id = 'first_pass-sub-section';
		firstPassSection.className = 'inference-sub-section active';

		const firstPassToggles = document.createElement('div');
		firstPassToggles.className = 'control-grid-2';
		firstPassToggles.appendChild(this.createToggle('enable_img2img', 'Image to Image', {default: false}));
		firstPassToggles.appendChild(this.createToggle('enable_style_model', 'Redux Style', {default: false}));
		firstPassSection.appendChild(firstPassToggles);

		const knobsGrid1 = document.createElement('div');
		knobsGrid1.className = 'control-grid-2';
		knobsGrid1.appendChild(this.createSlider({ name: 'steps', label: 'Steps', min: 4, max: 50, step: 1, default: 28, precision: 0 }));
		knobsGrid1.appendChild(this.createSlider({ name: 'img2img_denoise', label: 'IMG2IMG Denoise', min: 0, max: 1, step: 0.01, default: 0.8, precision: 2 }));
		firstPassSection.appendChild(knobsGrid1);

		const reduxGrid = document.createElement('div');
		reduxGrid.className = 'control-grid-2';
		const strengthGroup = this.createControlGroup('Strength Type');
		strengthGroup.appendChild(this.createSelect('strength_type', this.getModelOptions('strength_type')));
		reduxGrid.appendChild(strengthGroup);
		const cropGroup = this.createControlGroup('Crop Mode');
		cropGroup.appendChild(this.createSelect('crop', this.getModelOptions('crop')));
		reduxGrid.appendChild(cropGroup);
		firstPassSection.appendChild(reduxGrid);
		firstPassSection.appendChild(this.createSlider({ name: 'style_strength', label: 'Style Strength', min: 0, max: 10, step: 0.01, precision: 2, default: 1.0}));
		
		const noiseInjectionSection = this.createControlGroup('Noise Injection');
		noiseInjectionSection.style.marginTop = '15px';
		const injectionHeaderRow = document.createElement('div');
		injectionHeaderRow.style.display = 'flex';
		injectionHeaderRow.style.justifyContent = 'space-around';
		injectionHeaderRow.style.width = '100%';
		injectionHeaderRow.style.marginBottom = '12px';
		injectionHeaderRow.appendChild(this.createToggle('enable_latent_injection', 'Enable Injection', { default: true }));
		
		const normalizeButton = document.createElement('button');
		normalizeButton.className = 'section-toggle';
		const updateNormalizeButton = (value) => {
			const isEnabled = value === 'enable';
			normalizeButton.textContent = `Normalize: ${isEnabled ? 'ON' : 'OFF'}`;
			normalizeButton.classList.toggle('enabled', isEnabled);
		};
		normalizeButton.addEventListener('click', () => {
			const currentValue = this.getWidgetValue('normalize_injected_noise', 'enable');
			const newValue = (currentValue === 'enable') ? 'disable' : 'enable';
			this.setWidgetValue('normalize_injected_noise', newValue);
			updateNormalizeButton(newValue);
		});
		updateNormalizeButton(this.getWidgetValue('normalize_injected_noise', 'enable'));
		this.numberInputs['normalize_injected_noise'] = { update: updateNormalizeButton, param: { default: 'enable' } };
		injectionHeaderRow.appendChild(normalizeButton);
		noiseInjectionSection.appendChild(injectionHeaderRow);
		
		const injectionGrid = document.createElement('div');
		injectionGrid.className = 'control-grid-2';
		injectionGrid.appendChild(this.createSlider({ name: 'injection_point', label: 'Point', min: 0.0, max: 1.0, step: 0.01, default: 0.75, precision: 2 }));
		injectionGrid.appendChild(this.createSlider({ name: 'injection_seed_offset', label: 'Seed Offset', min: -100, max: 100, step: 1, default: 1, precision: 0 }));
		noiseInjectionSection.appendChild(injectionGrid);
		noiseInjectionSection.appendChild(this.createSlider({ name: 'injection_strength', label: 'Strength', min: -2.0, max: 2.0, step: 0.01, default: 0.3, precision: 2 }));
		firstPassSection.appendChild(noiseInjectionSection);
		
		subTabContentContainer.appendChild(firstPassSection);

		// --- Second Pass Content ---
		const secondPassSection = document.createElement('div');
		secondPassSection.id = 'second_pass-sub-section';
		secondPassSection.className = 'inference-sub-section';

		const secondPassToggles = document.createElement('div');
		secondPassToggles.className = 'control-grid-2';
		secondPassToggles.appendChild(this.createToggle('enable_upscale_with_model', 'Enable Upscaling', {default: true}));
		secondPassToggles.appendChild(this.createToggle('enable_2nd_pass', 'Enable 2nd Pass', {default: true}));
		secondPassSection.appendChild(secondPassToggles);

		const upscaleGrid = document.createElement('div');
		upscaleGrid.className = 'control-grid-2';
		upscaleGrid.appendChild(this.createSlider({ name: 'downscale_by', label: 'Downscale By', min: 0.25, max: 1, step: 0.05, default: 0.5, precision: 2 }));
		
		upscaleGrid.appendChild(this.createSelect('precision', this.getModelOptions('precision')));
		secondPassSection.appendChild(upscaleGrid);

		const knobsGrid2 = document.createElement('div');
		knobsGrid2.className = 'control-grid-2';
		knobsGrid2.appendChild(this.createSlider({ name: 'steps_2ND', label: 'Steps (2nd Pass)', min: 4, max: 50, step: 1, default: 20, precision: 0 }));
		knobsGrid2.appendChild(this.createSlider({ name: 'denoise_2ND', label: 'Denoise (2nd Pass)', min: 0, max: 1, step: 0.01, default: 0.3, precision: 2 }));
		secondPassSection.appendChild(knobsGrid2);

		secondPassSection.appendChild(this.createSlider({ name: 'seed_shift_2ND', label: 'Seed Shift', min: -100, max: 100, step: 1, default: 1, precision: 0 }));

		const tilingSection = this.createControlGroup('Tiling Settings');
		tilingSection.appendChild(this.createToggle('enable_tiling', 'Enable Tiling', {default: true}));
		const tilingGrid = document.createElement('div');
		tilingGrid.className = 'control-grid-3';

		tilingGrid.appendChild(this.createSelect('tiles', ["2x2 (4)", "3x3 (9)", "4x4 (16)"]));
		
        tilingGrid.appendChild(this.createSlider({ name: 'tile_padding', label: 'Padding', min: 0, max: 256, step: 8, default: 32, precision: 0 }));
		tilingGrid.appendChild(this.createSlider({ name: 'mask_blur', label: 'Mask Blur', min: 0, max: 64, step: 1, default: 16, precision: 0 }));
		tilingSection.appendChild(tilingGrid);
		secondPassSection.appendChild(tilingSection);

		subTabContentContainer.appendChild(secondPassSection);

		// --- Face Enhance Content ---
		const faceEnhanceSection = document.createElement('div');
		faceEnhanceSection.id = 'face_enhance-sub-section';
		faceEnhanceSection.className = 'inference-sub-section';
		
		const faceHeader = document.createElement('div');
		faceHeader.className = 'control-grid-3'; // Changed to 3 columns
		faceHeader.appendChild(this.createToggle('enable_face_enhancement', 'Enable Face Enhancement', {default: false}));
		faceHeader.appendChild(this.createToggle('face_resize_back', 'Resize to Original', {default: false}));
		faceHeader.appendChild(this.createToggle('enable_image_mix', 'Enable Image Mix', {default: false})); // Added Toggle
		faceEnhanceSection.appendChild(faceHeader);

		const faceResolutionsGrid = document.createElement('div');
		faceResolutionsGrid.className = 'control-grid-2';
		faceResolutionsGrid.appendChild(this.createSlider({name: 'face_initial_resize', label: 'Initial Resize', min: 256, max: 8192, step: 64, default: 4096, precision: 0}));
		faceResolutionsGrid.appendChild(this.createSlider({name: 'face_upscale_res', label: 'Face Upscale Res', min: 512, max: 4096, step: 64, default: 1536, precision: 0}));
		faceEnhanceSection.appendChild(faceResolutionsGrid);

		const faceThresholdsGrid = document.createElement('div');
		faceThresholdsGrid.className = 'control-grid-2';
		faceThresholdsGrid.appendChild(this.createSlider({name: 'face_bbox_threshold', label: 'BBox Threshold', min: 0.1, max: 1.0, step: 0.01, default: 0.5, precision: 2}));
		faceThresholdsGrid.appendChild(this.createSlider({name: 'face_segm_threshold', label: 'Segm Threshold', min: 0.1, max: 1.0, step: 0.01, default: 0.5, precision: 2}));
		faceEnhanceSection.appendChild(faceThresholdsGrid);

		const faceMaskingGrid = document.createElement('div');
		faceMaskingGrid.className = 'control-grid-2';
		faceMaskingGrid.appendChild(this.createSlider({name: 'face_mask_expand', label: 'Mask Expand', min: -512, max: 512, step: 1, default: 16, precision: 0}));
		faceMaskingGrid.appendChild(this.createSlider({name: 'face_mask_blur', label: 'Mask Blur', min: 0, max: 256, step: 0.5, default: 16.0, precision: 1}));
		faceEnhanceSection.appendChild(faceMaskingGrid);
		
		const faceParamsGrid = document.createElement('div');
		faceParamsGrid.className = 'control-grid-3';
		faceParamsGrid.appendChild(this.createSlider({name: 'face_steps', label: 'Steps', min: 1, max: 100, step: 1, default: 20, precision: 0}));
		faceParamsGrid.appendChild(this.createSlider({name: 'face_seed_shift', label: 'Seed Shift', min: -1000, max: 1000, step: 1, default: 1, precision: 0}));
		faceParamsGrid.appendChild(this.createSlider({name: 'face_padding', label: 'Padding', min: 0, max: 512, step: 8, default: 64, precision: 0}));
		faceEnhanceSection.appendChild(faceParamsGrid);

		const faceStrengthGrid = document.createElement('div');
		faceStrengthGrid.className = 'control-grid-3';
		faceStrengthGrid.appendChild(this.createSlider({name: 'face_cnet_strength', label: 'CNet Strength', min: 0, max: 2, step: 0.05, default: 0.7, precision: 2}));
		faceStrengthGrid.appendChild(this.createSlider({name: 'face_cnet_end', label: 'CNet End %', min: 0, max: 1, step: 0.01, default: 0.7, precision: 2}));
		faceStrengthGrid.appendChild(this.createSlider({name: 'face_color_match_strength', label: 'Color Match', min: 0, max: 2, step: 0.05, default: 1.0, precision: 2}));
		faceEnhanceSection.appendChild(faceStrengthGrid);

        const faceMixGrid = document.createElement('div');
        faceMixGrid.className = 'control-grid-1';
        faceMixGrid.appendChild(this.createSlider({name: 'image_mix_factor', label: 'Face Mix', min: 0.0, max: 1.0, step: 0.01, default: 0.5, precision: 2}));
        faceEnhanceSection.appendChild(faceMixGrid);

		subTabContentContainer.appendChild(faceEnhanceSection);

		setTimeout(() => {
			if (this.updateTabColors) {
				this.updateTabColors();
			}
		}, 100);

		parent.appendChild(section);
	}
	
	createPostProcessSection(parent) {
		const section = document.createElement('div');
		section.id = 'postprocess-section';
		section.className = 'fluxaio-section';

		const title = document.createElement('div');
		title.className = 'section-title';
		title.textContent = 'Post-Processing Effects';
		section.appendChild(title);

		const tabsContainer = document.createElement('div');
		tabsContainer.className = 'postprocess-tabs';

		Object.entries(POSTPROCESS_OPERATIONS).forEach(([opKey, opData]) => {
			const tab = document.createElement('button');
			tab.className = 'postprocess-tab';
			tab.textContent = `${opData.icon} ${opKey}`;
			tab.dataset.operation = opKey;
			
			if (opKey === this.activePostProcessTab) {
				tab.classList.add('active');
			}
			
			tab.addEventListener('click', () => {
				this.activePostProcessTab = opKey;
				this.updatePostProcessTabs();
			});
			
			tabsContainer.appendChild(tab);
		});
		
		section.appendChild(tabsContainer);

		Object.entries(POSTPROCESS_OPERATIONS).forEach(([opKey, opData]) => {
			const opSection = document.createElement('div');
			opSection.className = 'postprocess-section';
			opSection.id = `postprocess-${opKey}`;
			
			if (opKey === this.activePostProcessTab) {
				opSection.classList.add('active');
			}

			const header = document.createElement('div');
			header.className = 'section-header';
			header.dataset.enableParam = opData.enable;

            const headerTitle = document.createElement('span');
            headerTitle.className = 'section-header-title';
            headerTitle.textContent = opKey.replace(/_/g, ' ');
            header.appendChild(headerTitle);

			header.addEventListener('click', () => {
				const paramName = header.dataset.enableParam;
				const newValue = !this.getWidgetValue(paramName, false);
				this.setWidgetValue(paramName, newValue);
				this.updatePostProcessVisuals();
			});
			opSection.appendChild(header);

			if (opData.params) {
				const grid = document.createElement('div');
                
                // --- MODIFICATION START ---
                // Use a 2-column grid for the TEMPERATURE section, and a 3-column grid for others.
                if (opKey === 'TEMPERATURE') {
                    grid.className = 'control-grid-2';
                } else {
                    grid.className = 'control-grid-3';
                }
                // --- MODIFICATION END ---

				opData.params.forEach(param => {
					const sliderContainer = this.createSlider(param);

                    if (param.name === 'barrel_distortion') {
                        sliderContainer.classList.add('full-width-control');
                    }

					grid.appendChild(sliderContainer);
				});
				opSection.appendChild(grid);
			}

			if (opData.subgroups) {
				Object.entries(opData.subgroups).forEach(([groupName, params]) => {
					const groupContainer = this.createControlGroup(groupName);
					const groupGrid = document.createElement('div');
					groupGrid.className = 'control-grid-3';
					params.forEach(param => {
						groupGrid.appendChild(this.createSlider(param));
					});
					groupContainer.appendChild(groupGrid);
					opSection.appendChild(groupContainer);
				});
			}

			if (opData.selectors) {
				const selectorGrid = document.createElement('div');
				selectorGrid.className = 'control-grid-2';
				opData.selectors.forEach(selector => {
					const group = this.createControlGroup(selector.label);
					group.appendChild(this.createSelect(selector.name, selector.options));
					selectorGrid.appendChild(group);
				});
				opSection.appendChild(selectorGrid);
			}

			if (opData.toggles) {
				const toggleGrid = document.createElement('div');
				toggleGrid.className = 'control-grid-2';
				opData.toggles.forEach(toggle => {
					toggleGrid.appendChild(this.createToggle(toggle.name, toggle.label, {default: false}));
				});
				opSection.appendChild(toggleGrid);
			}

			if (opKey === 'RADIAL_BLUR') {
				const radialBlurLayout = document.createElement('div');
				radialBlurLayout.className = 'radial-blur-layout';
				
				const controlsDiv = document.createElement('div');
				opSection.querySelectorAll('.param-slider-container').forEach(slider => {
					controlsDiv.appendChild(slider);
				});
				
				const padDiv = document.createElement('div');
				padDiv.appendChild(this.createRadialBlurControlPad());
				
				radialBlurLayout.appendChild(controlsDiv);
				radialBlurLayout.appendChild(padDiv);
				
				opSection.innerHTML = '';
				opSection.appendChild(header);
				opSection.appendChild(radialBlurLayout);
			}

			section.appendChild(opSection);
		});
        
        setTimeout(() => this.updatePostProcessVisuals(), 100);

		parent.appendChild(section);
	}
	
		updatePostProcessVisuals() {
			if (!this.container) return;
			
			Object.entries(POSTPROCESS_OPERATIONS).forEach(([opKey, opData]) => {
				const isEnabled = this.getWidgetValue(opData.enable, false);

				// Update the main tab's color
				const tab = this.container.querySelector(`.postprocess-tab[data-operation="${opKey}"]`);
				if (tab) {
					tab.classList.toggle('enabled', isEnabled);
				}

				// Update the clickable header bar's appearance
				const header = this.container.querySelector(`#postprocess-${opKey} .section-header`);
				if (header) {
					header.classList.toggle('enabled', isEnabled);
				}
			});
		}
		
		updateTabColors() {
			if (!this.container) return;
			
			const firstPassTab = this.container.querySelector('[data-tab-id="first_pass"]');
			const secondPassTab = this.container.querySelector('[data-tab-id="second_pass"]');
			const faceEnhanceTab = this.container.querySelector('[data-tab-id="face_enhance"]');
			
			// First pass is always green (always runs)
			if (firstPassTab) {
				firstPassTab.classList.add('enabled');
			}
			
			// Second pass based on setting
			if (secondPassTab) {
				const enabled = this.getWidgetValue('enable_2nd_pass', false);
				secondPassTab.classList.toggle('enabled', enabled);
			}
			
			// Face enhance based on setting
			if (faceEnhanceTab) {
				const enabled = this.getWidgetValue('enable_face_enhancement', false);
				faceEnhanceTab.classList.toggle('enabled', enabled);
			}
		}
		updatePostProcessTabs() {
			if (!this.container) return;
			
			// Update tab active states
			this.container.querySelectorAll('.postprocess-tab').forEach(tab => {
				tab.classList.toggle('active', tab.dataset.operation === this.activePostProcessTab);
			});
			
			// Update section visibility
			this.container.querySelectorAll('.postprocess-section').forEach(section => {
				section.classList.toggle('active', section.id === `postprocess-${this.activePostProcessTab}`);
			});
		}
	
    createGallerySection(parent) {
        const section = document.createElement('div');
        section.id = 'gallery-section';
        section.className = 'fluxaio-section';

        const title = document.createElement('div');
        title.className = 'section-title';
        title.textContent = 'Gallery & Preview Management';
        section.appendChild(title);
        
        const livePreviewContainer = document.createElement('div');
        livePreviewContainer.className = 'live-preview-container empty';
        this.livePreviewContainer = livePreviewContainer;
        
        const livePreviewTitle = document.createElement('div');
        livePreviewTitle.className = 'live-preview-title';
        livePreviewTitle.textContent = 'Live Preview';
        livePreviewContainer.appendChild(livePreviewTitle);

        const livePreviewImage = document.createElement('img');
        livePreviewImage.className = 'live-preview-image';
        this.livePreviewImage = livePreviewImage;
        livePreviewContainer.appendChild(livePreviewImage);

        const livePreviewPlaceholder = document.createElement('div');
        livePreviewPlaceholder.className = 'live-preview-placeholder';
        livePreviewPlaceholder.textContent = 'Preview will appear here during generation...';
        livePreviewContainer.appendChild(livePreviewPlaceholder);
        
        section.appendChild(livePreviewContainer);

        parent.appendChild(section);
    }

    createSaveSection(parent) {
        const section = document.createElement('div');
        section.id = 'save-section';
        section.className = 'fluxaio-section';
    
        const title = document.createElement('div');
        title.className = 'section-title';
        title.textContent = 'Save & Export Options';
        section.appendChild(title);
    
        const saveContainer = document.createElement('div');
        saveContainer.className = 'control-grid-2';
    
        const saveToggle = this.createToggle('enable_save_image', 'Enable Image Saving', {default: true});
        saveContainer.appendChild(saveToggle);
    
        const filenameGroup = this.createControlGroup('Filename Prefix');
        const filenameInput = document.createElement('input');
        filenameInput.type = 'text';
        filenameInput.className = 'param-text-input';
        filenameInput.style.minHeight = '32px';
        filenameInput.value = this.getWidgetValue('filename_prefix', 'flux_aio/image');
        filenameInput.placeholder = 'flux_aio/image';
        filenameInput.addEventListener('input', (e) => {
            this.setWidgetValue('filename_prefix', e.target.value);
        });
        filenameGroup.appendChild(filenameInput);
        saveContainer.appendChild(filenameGroup);
    
        section.appendChild(saveContainer);
    
        const exportSection = this.createControlGroup('Export Options');
        const exportGrid = document.createElement('div');
        exportGrid.className = 'control-grid-3';
    
        const exportBtn = document.createElement('button');
        exportBtn.className = 'export-btn';
        exportBtn.textContent = 'Export Settings';
        exportBtn.addEventListener('click', () => this.exportSettings());
    
        const importBtn = document.createElement('button');
        importBtn.className = 'export-btn';
        importBtn.textContent = 'Import Settings';
        importBtn.addEventListener('click', () => this.importSettings());
    
        const resetAllBtn = document.createElement('button');
        resetAllBtn.className = 'reset-button';
        resetAllBtn.textContent = 'Reset All';
        resetAllBtn.addEventListener('click', () => this.resetAllSettings());
    
        exportGrid.appendChild(exportBtn);
        exportGrid.appendChild(importBtn);
        exportGrid.appendChild(resetAllBtn);
        exportSection.appendChild(exportGrid);
        section.appendChild(exportSection);
    
        parent.appendChild(section);
    }

    setGalleryMode(mode) {
        this.comparisonMode = (mode === 'comparison');
        const galleryGrid = this.container.querySelector('#gallery-grid');
        const buttons = this.container.querySelectorAll('.gallery-controls .gallery-btn');
        
        if (galleryGrid) {
            galleryGrid.classList.toggle('comparison-mode', this.comparisonMode);
        }
        
        buttons.forEach(btn => {
            btn.classList.remove('active');
            if ((mode === 'gallery' && btn.textContent === 'Gallery') ||
                (mode === 'comparison' && btn.textContent === 'Compare')) {
                btn.classList.add('active');
            }
        });
    }

    clearGallery() {
        const galleryGrid = this.container.querySelector('#gallery-grid');
        if (galleryGrid) {
            galleryGrid.innerHTML = '<div class="gallery-placeholder">Final generated images will appear here</div>';
        }
        this.resultImages = {};
        
        if (this.livePreviewContainer) {
            this.livePreviewContainer.classList.add('empty');
            if (this.livePreviewImage) this.livePreviewImage.src = '';
            if (this.currentPreviewImageURL) {
                URL.revokeObjectURL(this.currentPreviewImageURL);
                this.currentPreviewImageURL = null;
            }
        }
    }

    exportSettings() {
        const settings = {};
        if (this.node.widgets) {
            this.node.widgets.forEach(widget => {
                if (widget.name && widget.value !== undefined) {
                    settings[widget.name] = widget.value;
                }
            });
        }
        settings.fluxAIOUIData = this.serialize();
        
        const dataStr = JSON.stringify(settings, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'fluxaio_settings.json';
        link.click();
        URL.revokeObjectURL(url);
    }

    importSettings() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        input.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    try {
                        const settings = JSON.parse(e.target.result);
                        this.applyImportedSettings(settings);
                    } catch (error) {
                        console.error('Failed to import settings:', error);
                        alert('Failed to import settings. Please check the file format.');
                    }
                };
                reader.readAsText(file);
            }
        });
        input.click();
    }

    applyImportedSettings(settings) {
        Object.entries(settings).forEach(([paramName, value]) => {
            if (paramName === 'fluxAIOUIData') {
                this.deserialize(value);
            } else {
                this.setWidgetValue(paramName, value);
            }
        });
        this.syncAllUIToWidgetState();
        this.node.setDirtyCanvas(true, true);

        console.log('[FluxAIOUI] Settings imported successfully');
    }

    initializeLoRABlockWidgets() {
        console.log('[FluxAIOUI] Initializing LoRA block widgets with safe defaults...');
        
        let fixedCount = 0;
        
        for (let i = 0; i < 19; i++) {
            const widgetName = `lora_block_${i}_double_weight`;
            const widget = this.node.widgets?.find(w => w.name === widgetName);
            if (widget) {
                if (widget.value === '' || widget.value === null || widget.value === undefined || isNaN(parseFloat(widget.value))) {
                    widget.value = 1.0;
                    fixedCount++;
                    console.log(`[FluxAIOUI] Fixed ${widgetName}: empty -> 1.0`);
                }
            }
        }
        
        for (let i = 0; i < 38; i++) {
            const widgetName = `lora_block_${i}_weight`;
            const widget = this.node.widgets?.find(w => w.name === widgetName);
            if (widget) {
                if (widget.value === '' || widget.value === null || widget.value === undefined || isNaN(parseFloat(widget.value))) {
                    widget.value = 1.0;
                    fixedCount++;
                    console.log(`[FluxAIOUI] Fixed ${widgetName}: empty -> 1.0`);
                }
            }
        }
        
        if (fixedCount > 0) {
            console.log(`[FluxAIOUI] Fixed ${fixedCount} LoRA block widgets`);
            this.node.setDirtyCanvas(true, true);
        }
        
        return fixedCount;
    }

    resetAllSettings() {
        if (confirm('Are you sure you want to reset all settings to defaults? This cannot be undone.')) {
            this.resetModelsTab();
            this.resetPerformanceTab();
            this.resetPromptTab();
            this.resetLoRATab();
            this.resetInferenceTab();
            this.resetPostProcessTab();
            this.resetGalleryTab();
            this.resetSaveTab();
            
            this.currentTheme = '#7700ff';
            this.applyTheme(this.currentTheme);
            const colorPicker = this.container.querySelector('.theme-color-picker');
            if (colorPicker) colorPicker.value = this.currentTheme;
            
            console.log('[FluxAIOUI] All settings reset to defaults');
        }
    }

    createControlGroup(title) {
        const group = document.createElement('div');
        group.className = 'control-group';
        if (title) {
            const titleEl = document.createElement('div');
            titleEl.className = 'control-group-title'; titleEl.textContent = title;
            group.appendChild(titleEl);
        }
        return group;
    }

    createControlGroupNoTitle() {
        const group = document.createElement('div');
        group.className = 'lora-blocks-control-group';
        return group;
    }
    
    createSlider(param) {
        const container = document.createElement('div');
        container.className = 'param-slider-container';
        const label = document.createElement('div');
        label.className = 'param-slider-label'; label.textContent = param.label;
        container.appendChild(label);
        const row = document.createElement('div');
        row.className = 'param-slider-row';
        const slider = document.createElement('input');
        slider.type = 'range'; slider.className = 'param-slider';
        slider.min = param.min; slider.max = param.max;
        slider.step = param.step || 0.01;
        const valueDisplay = document.createElement('div');
        valueDisplay.className = 'param-value';
        
        const update = (value) => {
            let numValue = Number(value);
            if(isNaN(numValue)) numValue = param.default;

            slider.value = numValue;
            valueDisplay.textContent = numValue.toFixed(param.precision ?? 2);
            const isDefault = Math.abs(numValue - param.default) < (param.step || 0.01) / 2;
            slider.classList.toggle('non-default', !isDefault);
        };
        
        slider.addEventListener('input', (e) => {
            let value = parseFloat(e.target.value);
            if (param.step >= 1) value = Math.round(value);
            update(value);
            this.setWidgetValue(param.name, value);
        });
        
        row.appendChild(slider); row.appendChild(valueDisplay);
        container.appendChild(row);
        
        this.numberInputs[param.name] = { update, param };
        update(this.getWidgetValue(param.name, param.default));
        return container;
    }
    
    createSelect(paramName, options) {
        const select = document.createElement('select');
        select.className = 'param-select';
        select.dataset.paramName = paramName;
        
        if (paramName === 'lora_selector') {
            const defaultOption = document.createElement('option');
            defaultOption.value = '';
            defaultOption.textContent = 'Select LoRA to add...';
            select.appendChild(defaultOption);
            
            options.slice(1).forEach(option => {
                if (option && typeof option === 'string' && option.trim()) {
                    const optionEl = document.createElement('option');
                    optionEl.value = option;
                    optionEl.textContent = option;
                    select.appendChild(optionEl);
                }
            });
            return select;
        }
        
        (options || []).forEach(option => {
            const optionEl = document.createElement('option');
            optionEl.value = option; 
            optionEl.textContent = option;
            select.appendChild(optionEl);
        });
        
        select.value = this.getWidgetValue(paramName, options?.[0]);
        select.addEventListener('change', (e) => this.setWidgetValue(paramName, e.target.value));
        return select;
    }

    createToggle(paramName, label, param) {
        const container = document.createElement('div');
        container.className = 'param-toggle-container';
        const labelEl = document.createElement('div');
        labelEl.className = 'param-toggle-label'; labelEl.textContent = label;
        const led = document.createElement('div');
        led.className = 'param-toggle-led';
        
        const update = (value) => {
            const active = !!value;
            led.classList.toggle('active', active);
            container.classList.toggle('active', active);
        };
        
	container.addEventListener('click', () => {
		const newValue = !this.getWidgetValue(paramName, param.default);
		this.setWidgetValue(paramName, newValue);
		update(newValue);
		
		// Update tab colors when relevant toggles change
		if (paramName === 'enable_2nd_pass' || paramName === 'enable_face_enhancement') {
			setTimeout(() => {
				if (this.updateTabColors) {
					this.updateTabColors();
				}
			}, 50);
		}
	});
        
        container.appendChild(labelEl); container.appendChild(led);
        this.numberInputs[paramName] = { update, param };
        update(this.getWidgetValue(paramName, param.default));
        return container;
    }

    createFluxKnob(param) {
        // This method now correctly returns an object with both the container and the update function
        const knobData = createFluxKnob({
            name: param.name,
            label: param.label,
            min: param.min,
            max: param.max,
            step: param.step,
            precision: param.precision,
            default: this.getWidgetValue(param.name, param.default),
            color: 'var(--primary-accent)',
            onChange: (val) => {
                this.setWidgetValue(param.name, val);
            }
        });

        // Store the update function for external syncing
        this.numberInputs[param.name] = { update: knobData.updateVisuals, param };
        return knobData; // Return the full object { container, updateVisuals }
    }

    createRadialBlurControlPad() {
        const container = document.createElement('div');
        container.className = 'radial-blur-pad-container';

        const pad = document.createElement('div');
        pad.className = 'control-pad';

        const handle = document.createElement('div');
        handle.className = 'control-pad-handle';

        pad.appendChild(handle);
        container.appendChild(pad);

        let isDragging = false;
        
        const update = (x, y) => {
            handle.style.left = `${x * 100}%`;
            handle.style.top = `${y * 100}%`;
            this.setWidgetValue('radial_blur_center_x', x);
            this.setWidgetValue('radial_blur_center_y', y);
        };

        const onMouseMove = (e) => {
            if (!isDragging) return;
            const rect = pad.getBoundingClientRect();
            let x = (e.clientX - rect.left) / rect.width;
            let y = (e.clientY - rect.top) / rect.height;
            x = Math.max(0, Math.min(1, x));
            y = Math.max(0, Math.min(1, y));
            update(x, y);
        };

        const onMouseUp = () => {
            isDragging = false;
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        };

        pad.addEventListener('mousedown', (e) => {
            isDragging = true;
            onMouseMove(e);
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        });
        
        const initialX = this.getWidgetValue('radial_blur_center_x', 0.5);
        const initialY = this.getWidgetValue('radial_blur_center_y', 0.5);
        update(initialX, initialY);
        
        this.numberInputs['radial_blur_center_x'] = {
            update: (val) => update(val, this.getWidgetValue('radial_blur_center_y'))
        };
        this.numberInputs['radial_blur_center_y'] = {
            update: (val) => update(this.getWidgetValue('radial_blur_center_x'), val)
        };

        return container;
    }

    switchTab(tabType) {
        this.activeTab = tabType;
        this.updateTabVisibility();
        
        setTimeout(() => this.setNodeSizeInstantly(), 50);
    }

    updateTabVisibility() {
        this.container.querySelectorAll('.fluxaio-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.type === this.activeTab);
        });
        
        this.container.querySelectorAll('.fluxaio-section').forEach(section => {
            const isVisible = section.id === `${this.activeTab}-section`;
            section.classList.toggle('visible', isVisible);
        });
    }

	toggleCollapse() {
		this.isCollapsed = !this.isCollapsed;
		
		if (this.container) {
			this.container.classList.toggle('collapsed', this.isCollapsed);
			const collapseBtn = this.container.querySelector('.fluxaio-collapse-btn');
			if (collapseBtn) {
				collapseBtn.textContent = this.isCollapsed ? '□' : '−';
				collapseBtn.title = this.isCollapsed ? 'Expand' : 'Collapse';
			}
		}
	}

    setupResizeObserver() {
        if (this.resizeObserver) this.resizeObserver.disconnect();
        
        this.resizeObserver = new ResizeObserver(() => {
            if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = requestAnimationFrame(this.handleResize);
        });
        
        if (this.container) this.resizeObserver.observe(this.container);
    }

    handleResize() {
        this.node.setDirtyCanvas(true, true);
    }

    setNodeSizeInstantly() {
        if (!this.node || !this.container) return;
        
        if (this.isCollapsed) {
            this.node.setSize([900, 150]);
            this.node.setDirtyCanvas(true, true);
            return;
        }

        let targetHeight = 180;
        
        const activeSection = this.container.querySelector('.fluxaio-section.visible');
        if (activeSection) {
            targetHeight += activeSection.scrollHeight + 40;
        } else {
            targetHeight += 500;
        }
        
        targetHeight = Math.max(150, Math.min(targetHeight, 50));
        
        this.node.size = [900, targetHeight]; 
        
        this.node.setDirtyCanvas(true, true);
    }
    
    syncAllUIToWidgetState() {
        if (!this.node.widgets || !this.isInitialized || !this.container) return;
        
        console.log("[FluxAIOUI] Performing full UI sync from widget state...");

        try {
            // Sync simple controls (sliders, toggles, dropdowns, text inputs)
            Object.entries(this.numberInputs).forEach(([name, input]) => {
                const value = this.getWidgetValue(name, input.param.default);
                if (input.update) {
                    input.update(value);
                }
            });

            this.container.querySelectorAll('select.param-select').forEach(select => {
                const paramName = select.dataset.paramName;
                if (paramName) {
                    select.value = this.getWidgetValue(paramName, select.options[0]?.value);
                }
            });
            
            this.container.querySelectorAll('.param-text-input, .preset-input, .seed-input').forEach(input => {
                 const paramName = input.dataset.param;
                 if(paramName) {
                     input.value = this.getWidgetValue(paramName, input.defaultValue || '');
                 }
            });

            // Sync complex components
            if (this.perfSwitch) {
                this.perfSwitch.syncState();
            }

            // Sync LoRA Block values FROM widgets TO the internal state, then update UI
            this.loraBlockValues.double_blocks.forEach((v, i) => {
                this.loraBlockValues.double_blocks[i] = this.getWidgetValue(`lora_block_${i}_double_weight`, 1.0);
            });
            this.loraBlockValues.single_blocks.forEach((v, i) => {
                this.loraBlockValues.single_blocks[i] = this.getWidgetValue(`lora_block_${i}_weight`, 1.0);
            });
            this.updateLoRAStaticSlotsDisplay();
            this.updateLoRABlocksUI();
            
            // Sync visual states that depend on widget values
            this.updatePostProcessVisuals();
            this.updateConditionalUIVisibility();

            // --- MOVED TO THE END ---
            // This ensures all toggles are visually updated BEFORE we check their state to color the tabs.
            this.updateTabColors(); 

        } catch (error) {
            console.error('[FluxAIOUI] Error during UI sync:', error);
        }
    }

    getWidgetValue(name, defaultValue = null) {
        const widget = this.node.widgets?.find(w => w.name === name);
        return (widget && widget.value !== undefined) ? widget.value : defaultValue;
    }

	validateAllLoRABlockWidgets() {
		console.log('[FluxAIOUI] Validating all LoRA block widgets...');
		let fixedCount = 0;
		
		for (let i = 0; i < 19; i++) {
			const widgetName = `lora_block_${i}_double_weight`;
			const widget = this.node.widgets?.find(w => w.name === widgetName);
			if (widget) {
				if (widget.value === '' || widget.value === null || widget.value === undefined || isNaN(parseFloat(widget.value))) {
					widget.value = 1.0;
					fixedCount++;
					console.log(`[FluxAIOUI] Fixed ${widgetName}: empty -> 1.0`);
				}
			}
		}
		
		for (let i = 0; i < 38; i++) {
			const widgetName = `lora_block_${i}_weight`;
			const widget = this.node.widgets?.find(w => w.name === widgetName);
			if (widget) {
				if (widget.value === '' || widget.value === null || widget.value === undefined || isNaN(parseFloat(widget.value))) {
					widget.value = 1.0;
					fixedCount++;
					console.log(`[FluxAIOUI] Fixed ${widgetName}: empty -> 1.0`);
				}
			}
		}
		
		if (fixedCount > 0) {
			console.log(`[FluxAIOUI] Fixed ${fixedCount} LoRA block widgets`);
			this.node.setDirtyCanvas(true, true);
		}
		
		return fixedCount;
	}

    destroy() {
        if (window.fluxAIOPreviewNodes) {
            window.fluxAIOPreviewNodes.delete(this.node.id);
        }
        
        if (this.currentPreviewImageURL) {
            URL.revokeObjectURL(this.currentPreviewImageURL);
        }

        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = null;
        }
        if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
        if (this.container && this.container.parentNode) this.container.parentNode.removeChild(this.container);
        this.container = null;
        console.log(`[FluxAIOUI] Destroyed UI instance for node ${this.node.id}`);
    }

	serialize() {
		return {
			activeTab: this.activeTab,
			activePostProcessTab: this.activePostProcessTab,
			activeLoRABlocksTab: this.activeLoRABlocksTab,
            activeInferenceSubTab: this.activeInferenceSubTab,
			loraBlockValues: this.loraBlockValues,
			currentTheme: this.currentTheme,
			isCollapsed: this.isCollapsed,
			additionalPromptText: this.additionalPromptText || ''
		};
	}

    deserialize(data) {
        if (!data) return;
        try {
            this.activeTab = data.activeTab || 'inference';
            this.activePostProcessTab = data.activePostProcessTab || 'LEVELS';
            this.activeLoRABlocksTab = data.activeLoRABlocksTab || 'double_blocks';
            this.activeInferenceSubTab = data.activeInferenceSubTab || 'first_pass';
            this.loraBlockValues = data.loraBlockValues || {
                double_blocks: Array(19).fill(1.0),
                single_blocks: Array(38).fill(1.0)
            };
            this.currentTheme = data.currentTheme || '#7700ff';
            this.isCollapsed = data.isCollapsed || false;
			this.additionalPromptText = data.additionalPromptText || '';
            
            if (this.isInitialized) {
                this.updateTabVisibility();
                this.updateLoRAStaticSlotsDisplay();
                this.updateLoRABlocksUI();
                this.switchLoRABlocksTab(this.activeLoRABlocksTab);
                this.applyTheme(this.currentTheme);
                this.syncAllUIToWidgetState();
                if (this.isCollapsed !== this.container.querySelector('.fluxaio-content-container').classList.contains('collapsed')) {
                    this.toggleCollapse();
                }
            }
        } catch (error) {
            console.error(`[FluxAIOUI] Error deserializing data for node ${this.node.id}:`, error);
        }
    }
}

app.registerExtension({
    name: "Comfy.FluxAIOUI.Enhanced",

    async setup() {
        console.log("[FluxAIOUI] Setting up API event interception for live preview.");

        let apiObject = api || window.app?.api || window.api;

        if (apiObject && apiObject.addEventListener) {
            const originalAddEventListener = apiObject.addEventListener;

            apiObject.addEventListener = function(type, callback) {
                if (type === "b_preview") {
                    const wrappedCallback = (event) => {
                        if (app.runningNodeId) {
                            const node = app.graph?.getNodeById(app.runningNodeId);
                            if (node && node.comfyClass === "FluxAIO_CRT" && window.fluxAIOPreviewNodes?.has(node.id)) {
                                const uiInstance = window.fluxAIOPreviewNodes.get(node.id);
                                if (uiInstance) {
                                    uiInstance.updatePreview(event.detail);
                                    return;
                                }
                            }
                        }
                        return callback(event);
                    };
                    return originalAddEventListener.call(this, type, wrappedCallback);
                }
                return originalAddEventListener.call(this, type, callback);
            };
            console.log("[FluxAIOUI] Live preview interception is active.");
        } else {
            console.error("[FluxAIOUI] Could not find API object to hook for previews.");
        }
    },

	async nodeCreated(node) {
		if (node.comfyClass === "FluxAIO_CRT") {
			console.log(`[FluxAIOUI Extension] Node created: ${node.id}`);

			node.bgcolor = "transparent";
			node.color = "transparent";
			node.onDrawBackground = function(ctx) { /* Do nothing */ };
			node.computeSize = function() { return [900, 50]; };

			setTimeout(() => {
				if (!node.fluxAIOUIInstance) {
					try {
						node.fluxAIOUIInstance = new FluxAIOUI(node);
					} catch (error) {
						console.error(`[FluxAIOUI Extension] Error creating UI instance for node ${node.id}:`, error);
					}
				}
			}, 10);

			const originalOnExecutionStart = node.onExecutionStart;
			node.onExecutionStart = function() {
				if (this.fluxAIOUIInstance) {
					this.fluxAIOUIInstance.forceSyncToBackend();
				}
				if (originalOnExecutionStart) {
					return originalOnExecutionStart.apply(this, arguments);
				}
			};

			const originalSerialize = node.serialize;
			node.serialize = function() {
				const data = originalSerialize ? originalSerialize.call(this) : {};
				if (this.fluxAIOUIInstance) data.fluxAIOUIData = this.fluxAIOUIInstance.serialize();
				return data;
			};

			const originalConfigure = node.configure;
			node.configure = function(data) {
				if (originalConfigure) originalConfigure.call(this, data);
				if (this.fluxAIOUIInstance && data.fluxAIOUIData) {
					this.fluxAIOUIInstance.deserialize(data.fluxAIOUIData);
				}
			};

			const originalOnRemoved = node.onRemoved;
			node.onRemoved = function() {
				if (this.fluxAIOUIInstance) {
					this.fluxAIOUIInstance.destroy();
					this.fluxAIOUIInstance = null;
				}
				if (originalOnRemoved) originalOnRemoved.call(this);
			};

			const originalOnExecuted = node.onExecuted;
			node.onExecuted = function(message) {
				if (this.fluxAIOUIInstance) {
                    // FIX: Handle both preview types
					if (message?.ui?.previews && message.ui.previews[0]?.source) {
						this.fluxAIOUIInstance.displayFinalImage(message.ui.previews[0]);
                        this.fluxAIOUIInstance.addFinalImageToGallery(message.ui.previews[0]);
					} else if (message?.ui?.images && message.ui.images.length > 0) {
                        this.fluxAIOUIInstance.displayFinalImage(message.ui.images[0]);
                        message.ui.images.forEach(img => this.fluxAIOUIInstance.addFinalImageToGallery(img));
                    }
				}

				if (originalOnExecuted) {
					return originalOnExecuted.apply(this, arguments);
				}
			};
		}
	}
});

console.log("FluxAIO_UI.js: Complete Professional UI loaded successfully");	
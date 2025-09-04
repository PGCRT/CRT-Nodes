import { app } from "/scripts/app.js";

const CSS_STYLES_ENHANCED_AUDIO = `
@font-face {
    font-family: 'Orbitron';
    src: url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&display=swap') format('woff2'),
         local('Orbitron');
}

:root {
    --audio-primary: #00ff88;
    --audio-primary-light: #40ffaa;
    --audio-secondary: #ff6b00;
    --audio-accent: #00d4ff;
    --audio-background: #000000;
    --audio-surface: #1a1a1a;
    --audio-border: #333333;
    --audio-text: #ffffff;
    --audio-text-dim: #888888;
    --audio-danger: #ff4444;
    --audio-warning: #ffaa00;
    --audio-grid: rgba(0, 255, 136, 0.2);
}

/* Make all text non-selectable */
.enhanced-audio-preview-container * {
    user-select: none !important;
    -webkit-user-select: none !important;
    -moz-user-select: none !important;
    -ms-user-select: none !important;
}

/* Main container */
.enhanced-audio-preview-node-widget {
    position: relative !important;
    box-sizing: border-box !important;
    width: 600px !important;
    min-height: 400px !important;
    padding: 0px !important;
    margin: 0 !important;
    overflow: visible !important;
    display: block !important;
    visibility: visible !important;
    z-index: 1 !important;
    top: -17px !important;
}

.enhanced-audio-preview-container {
    background: var(--audio-background);
    border: 2px solid var(--audio-primary);
    border-radius: 20px;
    padding: 20px;
    margin: 0;
    width: 100%;
    max-width: 600px;
    box-sizing: border-box;
    min-height: 380px;
    box-shadow: 
        0 12px 40px rgba(0, 255, 136, 0.4),
        inset 0 2px 0 rgba(255, 255, 255, 0.1),
        0 0 60px rgba(0, 255, 136, 0.2);
    display: flex;
    flex-direction: column;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(10px);
}

.enhanced-audio-preview-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(circle at 30% 20%, rgba(0, 255, 136, 0.1), transparent 50%),
                radial-gradient(circle at 70% 80%, rgba(0, 212, 255, 0.05), transparent 50%);
    pointer-events: none;
    z-index: 0;
}

.enhanced-audio-preview-container > * {
    position: relative;
    z-index: 1;
}

/* Title */
.enhanced-audio-title {
    color: var(--audio-primary);
    font-family: 'Orbitron', monospace;
    font-size: 20px;
    font-weight: 700;
    text-align: center;
    margin: 0 0 20px 0;
    padding: 12px 0;
    text-shadow: 
        1px 1px 2px rgba(0, 0, 0, 0.8), 
        0 0 30px var(--audio-primary), 
        0 0 8px var(--audio-primary-light);
    background: linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.15), transparent);
    border-radius: 12px;
    animation: audioTitleGlow 4s ease-in-out infinite;
    position: relative;
}

.enhanced-audio-title::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    width: 60%;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--audio-primary), transparent);
    transform: translateX(-50%);
    border-radius: 1px;
}

@keyframes audioTitleGlow {
    0%, 100% {
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8), 0 0 30px var(--audio-primary), 0 0 8px var(--audio-primary-light);
        transform: scale(1);
    }
    50% {
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8), 0 0 40px var(--audio-primary-light), 0 0 15px var(--audio-primary);
        transform: scale(1.02);
        color: var(--audio-primary-light);
    }
}

/* Main audio section */
.audio-main-section {
    display: flex;
    flex-direction: column;
    gap: 20px;
    margin-bottom: 20px;
}

/* Waveform display */
.audio-waveform-container {
    background: rgba(0, 0, 0, 0.4);
    border: 2px solid var(--audio-border);
    border-radius: 16px;
    padding: 16px;
    height: 120px;
    position: relative;
    overflow: hidden;
    box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.6);
    cursor: pointer;
    transition: all 0.3s ease;
}

.audio-waveform-container:hover {
    border-color: var(--audio-primary);
    box-shadow: 
        inset 0 2px 8px rgba(0, 0, 0, 0.6),
        0 0 15px rgba(0, 255, 136, 0.3);
}

.audio-waveform-canvas {
    width: 100%;
    height: 100%;
    background: transparent;
    border-radius: 8px;
    pointer-events: none;
}

.audio-waveform-progress {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    background: linear-gradient(90deg, 
        rgba(0, 255, 136, 0.5), 
        rgba(0, 255, 136, 0.2));
    border-radius: 16px;
    pointer-events: none;
    width: 0%;
}

/* Control panel */
.audio-control-panel {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid var(--audio-border);
    border-radius: 16px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

/* Transport controls */
.audio-transport-controls {
    display: flex;
    align-items: center;
    gap: 16px;
    justify-content: space-between;
    width: 100%;
}

.audio-transport-left {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-shrink: 0;
}

.audio-transport-right {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-shrink: 0;
    min-width: 200px;
}

.audio-play-button {
    background: linear-gradient(45deg, var(--audio-primary), var(--audio-primary-light));
    border: none;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    cursor: pointer;
    color: var(--audio-background);
    font-size: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.3s ease;
    box-shadow: 
        0 4px 15px rgba(0, 255, 136, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.2);
    position: relative;
    overflow: hidden;
}

.audio-play-button:hover {
    transform: scale(1.05);
    box-shadow: 
        0 6px 20px rgba(0, 255, 136, 0.6),
        inset 0 1px 0 rgba(255, 255, 255, 0.3);
}

.audio-play-button.playing {
    background: linear-gradient(45deg, var(--audio-danger), #ff6666);
    animation: playingPulse 2s ease-in-out infinite;
}

@keyframes playingPulse {
    0%, 100% { box-shadow: 0 4px 15px rgba(255, 68, 68, 0.4); }
    50% { box-shadow: 0 6px 25px rgba(255, 68, 68, 0.8); }
}

.audio-stop-button, .audio-loop-button {
    background: linear-gradient(45deg, var(--audio-text-dim), #aaa);
    border: none;
    border-radius: 8px;
    width: 40px;
    height: 40px;
    cursor: pointer;
    color: var(--audio-background);
    font-size: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.3s ease;
}

.audio-stop-button:hover {
    background: linear-gradient(45deg, var(--audio-warning), #ffcc44);
}

.audio-loop-button {
    font-size: 18px;
    background: var(--audio-surface);
    color: var(--audio-text-dim);
    border: 1px solid var(--audio-border);
}

.audio-loop-button.active, .audio-loop-button:hover {
    color: var(--audio-primary);
    border-color: var(--audio-primary);
    box-shadow: 0 0 8px rgba(0, 255, 136, 0.4);
}

/* Time display */
.audio-time-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    min-width: 160px;
    flex-shrink: 0;
}

.audio-time-display {
    font-family: 'Orbitron', monospace;
    font-size: 16px;
    font-weight: 600;
    color: var(--audio-accent);
    text-shadow: 0 0 8px var(--audio-accent);
    background: rgba(0, 0, 0, 0.4);
    padding: 8px 16px;
    border-radius: 8px;
    border: 1px solid var(--audio-border);
    text-align: center;
    font-variant-numeric: tabular-nums;
    letter-spacing: 1px;
    width: 140px;
    box-sizing: border-box;
}

.audio-progress-bar {
    width: 100%;
    height: 6px;
    background: var(--audio-border);
    border-radius: 3px;
    overflow: hidden;
    cursor: pointer;
    position: relative;
}

.audio-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--audio-accent), var(--audio-primary));
    border-radius: 3px;
    width: 0%;
    box-shadow: 0 0 8px rgba(0, 212, 255, 0.5);
}

/* VU Meters */
.audio-vu-section {
    display: flex;
    justify-content: center;
    gap: 30px;
    padding: 20px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 12px;
    border: 1px solid var(--audio-border);
}

.audio-vu-meter {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
}

.audio-vu-label {
    font-family: 'Orbitron', monospace;
    font-size: 12px;
    font-weight: 600;
    color: var(--audio-text-dim);
    text-transform: uppercase;
    letter-spacing: 1px;
}

.audio-vu-bar-container {
    width: 30px;
    height: 180px;
    background: var(--audio-background);
    border: 2px solid var(--audio-border);
    border-radius: 15px;
    position: relative;
    overflow: hidden;
    box-shadow: inset 0 3px 6px rgba(0, 0, 0, 0.6);
}

.audio-vu-bar {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(to top, 
        var(--audio-primary) 0%, 
        var(--audio-primary) 60%, 
        var(--audio-warning) 80%, 
        var(--audio-danger) 100%);
    border-radius: 13px;
    transition: height 0.05s ease-out;
    height: 0%;
    box-shadow: 0 0 12px rgba(0, 255, 136, 0.5);
}

.audio-vu-peak-indicator {
    position: absolute;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--audio-text);
    border-radius: 2px;
    transition: top 0.05s ease-out;
    top: 100%;
    box-shadow: 0 0 6px rgba(255, 255, 255, 0.8);
}

/* Metrics display */
.audio-metrics-section {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 8px;
}

.audio-metric-item {
    background: rgba(0, 0, 0, 0.4);
    border: 1px solid var(--audio-border);
    border-radius: 12px;
    padding: 12px;
    text-align: center;
    transition: all 0.3s ease;
}

.audio-metric-item:hover {
    border-color: var(--audio-primary);
    box-shadow: 0 4px 12px rgba(0, 255, 136, 0.3);
    transform: translateY(-2px);
}

.audio-metric-label {
    font-family: 'Orbitron', monospace;
    font-size: 10px;
    font-weight: 600;
    color: var(--audio-text-dim);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}

.audio-metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 14px;
    font-weight: 700;
    color: var(--audio-primary);
    text-shadow: 0 0 8px var(--audio-primary);
}

.audio-metric-value.danger {
    color: var(--audio-danger);
    text-shadow: 0 0 8px var(--audio-danger);
}

.audio-metric-value.warning {
    color: var(--audio-warning);
    text-shadow: 0 0 8px var(--audio-warning);
}

/* Volume control */
.audio-volume-section {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 12px;
    border: 1px solid var(--audio-border);
}

.audio-volume-label {
    font-family: 'Orbitron', monospace;
    font-size: 12px;
    font-weight: 600;
    color: var(--audio-text-dim);
    min-width: 60px;
}

.audio-volume-slider {
    flex: 1;
    height: 6px;
    background: var(--audio-border);
    border-radius: 3px;
    outline: none;
    cursor: pointer;
    -webkit-appearance: none;
    appearance: none;
}

.audio-volume-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 16px;
    height: 16px;
    background: var(--audio-primary);
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 0 8px rgba(0, 255, 136, 0.5);
    transition: all 0.2s ease;
}

.audio-volume-slider::-webkit-slider-thumb:hover {
    transform: scale(1.2);
    box-shadow: 0 0 12px rgba(0, 255, 136, 0.8);
}

.audio-volume-value {
    font-family: 'Orbitron', monospace;
    font-size: 12px;
    font-weight: 600;
    color: var(--audio-primary);
    min-width: 40px;
    text-align: right;
}

/* Status indicators */
.audio-status-indicators {
    display: flex;
    gap: 8px;
    position: absolute;
    top: 16px;
    right: 16px;
}

.audio-status-led {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--audio-text-dim);
    animation: pulse 2s ease-in-out infinite;
}

.audio-status-led.ready {
    background: var(--audio-primary);
    box-shadow: 0 0 8px var(--audio-primary);
}

.audio-status-led.playing {
    background: var(--audio-danger);
    box-shadow: 0 0 8px var(--audio-danger);
}

.audio-status-led.error {
    background: var(--audio-danger);
    box-shadow: 0 0 8px var(--audio-danger);
    animation: fastPulse 0.5s ease-in-out infinite;
}
`;

class EnhancedAudioPreviewUI {
    constructor(node) {
        this.node = node;
        this.isInitialized = false;
        this.isPlaying = false;
        this.isDraggingProgress = false;
        this.isLooping = false;
        this.audioContext = null;
        this.audioBuffer = null;
        this.sourceNode = null;
        this.gainNode = null;
        this.analyserL = null;
        this.analyserR = null;
        this.animationFrameId = null;
        this.currentTime = 0;
        this.startTime = 0;
        this.duration = 0;
        this.volume = 0.8;
        this.peakL = 0; this.peakR = 0;

        this.updateVisualization = this.updateVisualization.bind(this);
        this.handlePlayButton = this.handlePlayButton.bind(this);
        this.handleStopButton = this.handleStopButton.bind(this);
        this.handleVolumeChange = this.handleVolumeChange.bind(this);
        this.handleLoopButton = this.handleLoopButton.bind(this);
		
        this.node.title = "";
        this.node.bgcolor = "transparent";
        this.node.color = "transparent";

        this.initializeUI();
    }

    initializeUI() {
        if (this.isInitialized) return;
        this.injectStyles();
        this.hideOriginalWidgets();
        this.createCustomDOM();
        this.setupAudioContext();
        this.isInitialized = true;
    }

    injectStyles() {
        if (!document.getElementById('enhanced-audio-preview-styles')) {
            const style = document.createElement('style');
            style.id = 'enhanced-audio-preview-styles';
            style.textContent = CSS_STYLES_ENHANCED_AUDIO;
            document.head.appendChild(style);
        }
    }

    hideOriginalWidgets() {
        if (this.node.widgets) {
            this.node.widgets.forEach(w => {
                if(w.name !== 'audio') w.computeSize = () => [0, -4]
            });
        }
    }

    createCustomDOM() {
        const widgetWrapper = document.createElement('div');
        widgetWrapper.className = 'enhanced-audio-preview-node-widget';
        
        this.container = document.createElement('div');
        this.container.className = 'enhanced-audio-preview-container';
        
        this.container.innerHTML = `
            <div class="enhanced-audio-title">Audio Preview Pro</div>
            <div class="audio-status-indicators"><div class="audio-status-led"></div></div>
            <div class="audio-main-section">
                <div class="audio-waveform-container">
                    <div class="audio-waveform-progress"></div>
                    <canvas class="audio-waveform-canvas"></canvas>
                </div>
                <div class="audio-control-panel">
                    <div class="audio-transport-controls">
                        <div class="audio-transport-left">
                            <button class="audio-play-button">▶</button>
                            <button class="audio-stop-button">■</button>
                            <button class="audio-loop-button">⟲</button>
                            <div class="audio-autopreview-container">
                                <div class="audio-autopreview-toggle"><div class="audio-autopreview-knob"></div></div>
                                <div class="audio-autopreview-label">Auto Preview</div>
                            </div>
                        </div>
                        <div class="audio-transport-right">
                            <div class="audio-time-container">
                                <div class="audio-time-display">00:00 / 00:00</div>
                                <div class="audio-progress-bar"><div class="audio-progress-fill"></div></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="audio-vu-section">
                    <div class="audio-vu-meter"><div class="audio-vu-label">Left</div><div class="audio-vu-bar-container"><div class="audio-vu-bar"></div><div class="audio-vu-peak-indicator"></div></div></div>
                    <div class="audio-vu-meter"><div class="audio-vu-label">Right</div><div class="audio-vu-bar-container"><div class="audio-vu-bar"></div><div class="audio-vu-peak-indicator"></div></div></div>
                </div>
                <div class="audio-metrics-section">
                    <div class="audio-metric-item"><div class="audio-metric-label">Peak</div><div class="audio-metric-value" data-id="peak">--</div></div>
                    <div class="audio-metric-item"><div class="audio-metric-label">RMS</div><div class="audio-metric-value" data-id="rms">--</div></div>
                    <div class="audio-metric-item"><div class="audio-metric-label">LUFS</div><div class="audio-metric-value" data-id="lufs">--</div></div>
                </div>
                <div class="audio-volume-section">
                    <div class="audio-volume-label">Volume</div><input type="range" class="audio-volume-slider" min="0" max="1" step="0.01"><div class="audio-volume-value">80%</div>
                </div>
            </div>`;
        
        widgetWrapper.appendChild(this.container);
        this.node.addDOMWidget('enhanced_audio_preview_ui', 'div', widgetWrapper, { serialize: false });

        const q = (sel) => this.container.querySelector(sel);
        const qa = (sel) => this.container.querySelectorAll(sel);
        this.playButton = q('.audio-play-button');
        this.stopButton = q('.audio-stop-button');
        this.loopButton = q('.audio-loop-button');
        this.timeDisplay = q('.audio-time-display');
        this.progressBar = q('.audio-progress-bar');
        this.progressFill = q('.audio-progress-fill');
        this.waveformContainer = q('.audio-waveform-container');
        this.waveformCanvas = q('.audio-waveform-canvas');
        this.waveformProgress = q('.audio-waveform-progress');
        [this.vuBarL, this.vuBarR] = qa('.audio-vu-bar');
        [this.vuPeakL, this.vuPeakR] = qa('.audio-vu-peak-indicator');
        this.peakValueEl = q('[data-id="peak"]');
        this.rmsValueEl = q('[data-id="rms"]');
        this.lufsValueEl = q('[data-id="lufs"]');
        this.volumeSlider = q('.audio-volume-slider');
        this.volumeValue = q('.audio-volume-value');
        this.statusLed = q('.audio-status-led');
        this.autoPreviewToggle = q('.audio-autopreview-toggle');
        this.autoPreviewKnob = q('.audio-autopreview-knob');
        this.autoPreviewLabel = q('.audio-autopreview-label');
        
        this.waveformCtx = this.waveformCanvas.getContext('2d');
        
        this.playButton.addEventListener('click', this.handlePlayButton);
        this.stopButton.addEventListener('click', this.handleStopButton);
        this.loopButton.addEventListener('click', this.handleLoopButton);
        this.volumeSlider.addEventListener('input', this.handleVolumeChange);
        this.setupProgressBarInteraction();
        this.setupAutoPreviewToggle();
        
        this.volumeSlider.value = this.volume;
        this.setStatus('ready');
    }
    
    setupAutoPreviewToggle() {
        this.autoPreviewEnabled = true;
        this.updateAutoPreviewToggleVisuals();
        const toggleContainer = this.container.querySelector('.audio-autopreview-container');
        toggleContainer.addEventListener('click', () => {
            this.autoPreviewEnabled = !this.autoPreviewEnabled;
            this.updateAutoPreviewToggleVisuals();
        });
    }

    updateAutoPreviewToggleVisuals() {
        if (this.autoPreviewEnabled) {
            this.autoPreviewToggle.style.background = 'var(--audio-primary)';
            this.autoPreviewKnob.style.left = '22px';
            this.autoPreviewLabel.style.color = 'var(--audio-primary)';
        } else {
            this.autoPreviewToggle.style.background = 'var(--audio-border)';
            this.autoPreviewKnob.style.left = '2px';
            this.autoPreviewLabel.style.color = 'var(--audio-text-dim)';
        }
    }

    setupProgressBarInteraction() {
        const setupSeeking = (container) => {
            let wasPlayingBeforeSeek = false;
            
            const seek = (e) => {
                if (!this.audioBuffer) return;
                const rect = container.getBoundingClientRect();
                const progress = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                this.currentTime = this.duration * progress;
                this.updateTimeDisplay();
                this.updateProgress();
            };
    
            container.addEventListener('mousedown', (e) => {
                e.preventDefault();
                this.isDraggingProgress = true;
                wasPlayingBeforeSeek = this.isPlaying;
                if (wasPlayingBeforeSeek) {
                    this.pausePlayback();
                }
                seek(e);
    
                const onMouseMove = (moveEvent) => seek(moveEvent);
                const onMouseUp = () => {
                    document.removeEventListener('mousemove', onMouseMove);
                    document.removeEventListener('mouseup', onMouseUp);
                    this.isDraggingProgress = false;
                    if (wasPlayingBeforeSeek) {
                        this.startPlayback();
                    }
                };
    
                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', onMouseUp);
            });
        };
        
        setupSeeking(this.progressBar);
        setupSeeking(this.waveformContainer);
    }
    
    setupAudioContext() {
        if (!window.enhancedAudioContext) {
            window.enhancedAudioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        this.audioContext = window.enhancedAudioContext;
    }

    handlePlayButton() {
        if (!this.audioBuffer) return;
        if (this.isPlaying) this.pausePlayback();
        else this.startPlayback();
    }

    handleStopButton() { this.stopPlayback(); }

    handleVolumeChange() {
        this.volume = parseFloat(this.volumeSlider.value);
        this.volumeValue.textContent = Math.round(this.volume * 100) + '%';
        if (this.gainNode) this.gainNode.gain.setValueAtTime(this.volume, this.audioContext.currentTime);
    }

    handleLoopButton() {
        this.isLooping = !this.isLooping;
        this.loopButton.classList.toggle('active', this.isLooping);
        if (this.sourceNode) this.sourceNode.loop = this.isLooping;
    }

    startPlayback() {
        if (!this.audioBuffer || this.isPlaying) return;

        this.audioContext.resume();
        this.sourceNode = this.audioContext.createBufferSource();
        this.sourceNode.buffer = this.audioBuffer;
        
        this.sourceNode.loop = this.isLooping; // Apply loop state
        
        this.gainNode = this.audioContext.createGain();
        this.gainNode.gain.value = this.volume;
        
        this.analyserL = this.audioContext.createAnalyser();
        this.analyserL.fftSize = 1024;
        this.analyserL.smoothingTimeConstant = 0.3;
        
        if (this.audioBuffer.numberOfChannels > 1) {
            this.analyserR = this.audioContext.createAnalyser();
            this.analyserR.fftSize = 1024;
            this.analyserR.smoothingTimeConstant = 0.3;
            const splitter = this.audioContext.createChannelSplitter(2);
            this.sourceNode.connect(splitter);
            splitter.connect(this.analyserL, 0);
            splitter.connect(this.analyserR, 1);
        } else {
            this.sourceNode.connect(this.analyserL);
            this.analyserR = this.analyserL;
        }
        
        this.sourceNode.connect(this.gainNode);
        this.gainNode.connect(this.audioContext.destination);
        
        this.sourceNode.start(0, this.currentTime % this.duration);
        this.isPlaying = true;
        this.playButton.innerHTML = '⏸';
        this.playButton.classList.add('playing');
        this.setStatus('playing');
        
        this.startTime = this.audioContext.currentTime - this.currentTime;
        
        this.sourceNode.onended = () => { if (this.isPlaying && !this.isLooping) this.stopPlayback(); };
        
        this.updateVisualization();
    }

    pausePlayback() {
        if (!this.isPlaying) return;
        this.currentTime = this.audioContext.currentTime - this.startTime;
        this.stopAudioNodes();
        this.isPlaying = false;
        this.playButton.innerHTML = '▶';
        this.playButton.classList.remove('playing');
        this.setStatus('ready');
        if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
        this.animationFrameId = null;
    }

    stopPlayback() {
        this.stopAudioNodes();
        this.isPlaying = false;
        this.currentTime = 0;
        this.playButton.innerHTML = '▶';
        this.playButton.classList.remove('playing');
        this.setStatus('ready');
        if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
        this.animationFrameId = null;
        this.updateTimeDisplay();
        this.updateProgress();
        this.resetVUMeters();
    }

    stopAudioNodes() {
        if (this.sourceNode) {
            try { this.sourceNode.stop(); } catch (e) {}
            this.sourceNode.disconnect();
            this.sourceNode = null;
        }
        if(this.gainNode) this.gainNode.disconnect();
        this.analyserL = null; this.analyserR = null; this.gainNode = null;
    }

    updateVisualization() {
        if (!this.isPlaying) return;
        
        if (!this.isDraggingProgress) {
            let elapsed = this.audioContext.currentTime - this.startTime;
            if (this.isLooping && this.duration > 0) {
                this.currentTime = elapsed % this.duration;
            } else {
                this.currentTime = elapsed;
            }
            if (!this.isLooping && this.currentTime >= this.duration) {
                this.stopPlayback();
                return;
            }
            this.updateTimeDisplay();
            this.updateProgress();
        }
        
        this.updateVUMeters();
        
        this.animationFrameId = requestAnimationFrame(this.updateVisualization);
    }

    updateTimeDisplay() {
        this.timeDisplay.textContent = `${this.formatTime(this.currentTime)} / ${this.formatTime(this.duration)}`;
    }

    updateProgress() {
        const percentage = this.duration > 0 ? (this.currentTime / this.duration) * 100 : 0;
        this.progressFill.style.width = `${Math.max(0, Math.min(100, percentage))}%`;
        this.waveformProgress.style.width = `${Math.max(0, Math.min(100, percentage))}%`;
    }

    updateVUMeters() {
        if (!this.analyserL) return;
        
        const dataArrayL = new Float32Array(this.analyserL.fftSize);
        this.analyserL.getFloatTimeDomainData(dataArrayL);
        
        let sumSquaresL = 0, currentPeakL = 0;
        dataArrayL.forEach(v => {
            const absV = Math.abs(v);
            if (absV > currentPeakL) currentPeakL = absV;
            sumSquaresL += v * v;
        });
        if (currentPeakL > this.peakL) this.peakL = currentPeakL;
        this.peakL *= 0.999;
        
        const rmsL = Math.sqrt(sumSquaresL / dataArrayL.length);
        const dbL = rmsL > 0 ? 20 * Math.log10(rmsL) : -60;
        const peakDbL = this.peakL > 0 ? 20 * Math.log10(this.peakL) : -60;

        let rmsR = rmsL, peakDbR = peakDbL;
        if (this.analyserR) {
            const dataArrayR = new Float32Array(this.analyserR.fftSize);
            this.analyserR.getFloatTimeDomainData(dataArrayR);
            let sumSquaresR = 0, currentPeakR = 0;
            dataArrayR.forEach(v => {
                const absV = Math.abs(v);
                if (absV > currentPeakR) currentPeakR = absV;
                sumSquaresR += v * v;
            });
            if (currentPeakR > this.peakR) this.peakR = currentPeakR;
            this.peakR *= 0.999;
            rmsR = Math.sqrt(sumSquaresR / dataArrayR.length);
            peakDbR = this.peakR > 0 ? 20 * Math.log10(this.peakR) : -60;
        }

        const vuPercentL = Math.max(0, Math.min(100, (dbL + 60) / 60 * 100));
        const vuPercentR = Math.max(0, Math.min(100, (20 * Math.log10(rmsR) + 60) / 60 * 100));
        this.vuBarL.style.height = `${vuPercentL}%`;
        this.vuBarR.style.height = `${vuPercentR}%`;
        
        const peakPercentL = Math.max(0, Math.min(100, (peakDbL + 60) / 60 * 100));
        const peakPercentR = Math.max(0, Math.min(100, (peakDbR + 60) / 60 * 100));
        this.vuPeakL.style.top = `${100 - peakPercentL}%`;
        this.vuPeakR.style.top = `${100 - peakPercentR}%`;
        
        const overallPeakDB = Math.max(peakDbL, peakDbR);
        const overallRMSDB = 20 * Math.log10((rmsL + rmsR) / 2);
        this.peakValueEl.textContent = overallPeakDB > -Infinity ? `${overallPeakDB.toFixed(1)} dBFS` : "-∞";
        this.rmsValueEl.textContent = overallRMSDB > -Infinity ? `${overallRMSDB.toFixed(1)} dBFS` : "-∞";
        
        this.peakValueEl.className = 'audio-metric-value';
        if (overallPeakDB > -3) this.peakValueEl.classList.add('danger');
        else if (overallPeakDB > -6) this.peakValueEl.classList.add('warning');
    }

    updateWaveform() {
        if (!this.waveformCtx || !this.audioBuffer) return;
        const { width, height } = this.waveformCanvas;
        this.waveformCtx.clearRect(0, 0, width, height);
        const channelData = this.audioBuffer.getChannelData(0);
        const step = Math.ceil(channelData.length / width);
        const amp = height / 2;
        this.waveformCtx.strokeStyle = 'rgba(0, 255, 136, 0.6)';
        this.waveformCtx.lineWidth = 1;
        this.waveformCtx.beginPath();
        for (let i = 0; i < width; i++) {
            let min = 1, max = -1;
            for (let j = i * step; j < (i + 1) * step; j++) {
                if(channelData[j] < min) min = channelData[j];
                if(channelData[j] > max) max = channelData[j];
            }
            this.waveformCtx.moveTo(i, (1 + min) * amp);
            this.waveformCtx.lineTo(i, (1 + max) * amp);
        }
        this.waveformCtx.stroke();
        this.waveformCtx.strokeStyle = 'rgba(0, 212, 255, 0.3)';
        this.waveformCtx.beginPath();
        this.waveformCtx.moveTo(0, amp);
        this.waveformCtx.lineTo(width, amp);
        this.waveformCtx.stroke();
    }

    resetVUMeters() {
        this.vuBarL.style.height = '0%';
        this.vuBarR.style.height = '0%';
        this.vuPeakL.style.top = '100%';
        this.vuPeakR.style.top = '100%';
        this.peakL = 0; this.peakR = 0;
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }

    setStatus(status) {
        if (this.statusLed) this.statusLed.className = `audio-status-led ${status}`;
    }

    async loadAudio(audioInfo, metricsInfo, autoplay) {
        try {
            this.setStatus('loading');
            this.stopPlayback();
            this.audioBuffer = null;
            this.duration = 0;
            this.currentTime = 0;
            
            const params = new URLSearchParams({ filename: audioInfo.filename, subfolder: audioInfo.subfolder, type: audioInfo.type });
            const response = await fetch(`/view?${params.toString()}`);
            if (!response.ok) throw new Error('Failed to fetch audio');
            const arrayBuffer = await response.arrayBuffer();
            
            if (this.audioContext.state === 'suspended') await this.audioContext.resume();
            this.audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
            this.duration = this.audioBuffer.duration;
            
            this.updateTimeDisplay();
            this.updateProgress();
            this.updateWaveform();
            
            if (metricsInfo) {
                this.peakValueEl.textContent = `${metricsInfo.peak} dBFS`;
                this.rmsValueEl.textContent = `${metricsInfo.rms} dBFS`;
                this.lufsValueEl.textContent = `${metricsInfo.lufs} LUFS`;
            }
            
            this.setStatus('ready');
            if (autoplay && this.autoPreviewEnabled) {
                setTimeout(() => this.startPlayback(), 100);
            }
        } catch (error) {
            console.error('Error loading audio:', error);
            this.setStatus('error');
        }
    }

    destroy() {
        this.stopPlayback();
        if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
        if (this.container) this.container.remove();
    }
}

app.registerExtension({
    name: "Comfy.AudioPreviewer.UI.Enhanced",
    async nodeCreated(node) {
        if (node.comfyClass === "AudioPreviewer") {
            if (node.enhancedAudioPreviewUIInstance) node.enhancedAudioPreviewUIInstance.destroy();
            node.enhancedAudioPreviewUIInstance = new EnhancedAudioPreviewUI(node);

            const originalOnExecuted = node.onExecuted;
            node.onExecuted = async function (message) {
                if (originalOnExecuted) originalOnExecuted.apply(this, arguments);
                if (message?.audio && node.enhancedAudioPreviewUIInstance) {
                    await node.enhancedAudioPreviewUIInstance.loadAudio(
                        message.audio[0], 
                        message.metrics ? message.metrics[0] : null, 
                        message.autoplay ? message.autoplay[0] : false
                    );
                }
            };

            const originalOnRemove = node.onRemove;
            node.onRemove = function() {
                if (node.enhancedAudioPreviewUIInstance) node.enhancedAudioPreviewUIInstance.destroy();
                if (originalOnRemove) originalOnRemove.call(node);
            };
        }
    }
});
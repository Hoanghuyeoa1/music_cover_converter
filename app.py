import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import gradio as gr

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.downloader import ensure_ffmpeg, ensure_fluidsynth, ensure_soundfonts, download_audio_from_url
from src.audio_processor import (
    AudioProcessor,
    INSTRUMENT_CONFIGS,
    detect_musical_key,
    calculate_smart_transpose,
    KEY_CENTER_PAIRS,
    extract_beats_and_tempo,
    estimate_bpm,
    time_stretch_audio,
    stretch_midi_tempo,
    extract_chord_progression_from_midi,
    format_chord_progression_text,
    is_sheet_music_file
)
from src.omr_processor import SheetMusicOMR

TARGET_KEY_CHOICES = ["Giữ nguyên gốc (Không đổi)"] + [p["label"] for p in KEY_CENTER_PAIRS]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("App")

# Initialize AudioProcessor singleton
try:
    processor = AudioProcessor()
except Exception as init_err:
    logger.error(f"Failed to initialize AudioProcessor: {init_err}")
    processor = None

device_label = "⚡ CUDA GPU (Tăng tốc)" if torch.cuda.is_available() else "💻 CPU Multi-Threading"

ALL_INSTRUMENTS = [
    'piano',
    'guitar_steel',
    'guitar_nylon',
    'flute',
    'strings',
    'saxophone',
    'epiano',
    'harp',
    'musicbox'
]


def normalize_instrument_key(name_or_key: str) -> Optional[str]:
    """Robustly map user selection labels or legacy aliases to ALL_INSTRUMENTS keys."""
    if not name_or_key:
        return None
    if name_or_key in ALL_INSTRUMENTS:
        return name_or_key
    if name_or_key in ("bamboo_flute", "flute"):
        return "flute"
    if name_or_key in ("electric_piano", "epiano"):
        return "epiano"
    if name_or_key in ("music_box", "musicbox"):
        return "musicbox"
    # Exact match from INSTRUMENT_CONFIGS name
    for k in ALL_INSTRUMENTS:
        if k in INSTRUMENT_CONFIGS and INSTRUMENT_CONFIGS[k]["name"] == name_or_key:
            return k
    # Keyword/substring matching
    low = name_or_key.lower()
    if "sáo" in low or "flute" in low or "tiêu" in low or "shakuhachi" in low:
        return "flute"
    if "piano" in low and "electric" not in low:
        return "piano"
    if "steel" in low:
        return "guitar_steel"
    if "nylon" in low:
        return "guitar_nylon"
    if "guitar" in low:
        return "guitar_steel"
    if "dây" in low or "string" in low:
        return "strings"
    if "sax" in low:
        return "saxophone"
    if "rhodes" in low or "electric piano" in low:
        return "epiano"
    if "harp" in low or "hạc" in low:
        return "harp"
    if "box" in low or "hộp nhạc" in low:
        return "musicbox"
    return None


CUSTOM_CSS = """
/* Modern Dark Glassmorphism Theme */
:root {
    --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    --card-bg: rgba(30, 32, 48, 0.7);
    --card-border: rgba(255, 255, 255, 0.08);
    --accent-glow: 0 8px 32px 0 rgba(99, 102, 241, 0.25);
}

body, .gradio-container {
    background-color: #0b0f19 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

.hero-banner {
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.8) 0%, rgba(49, 46, 129, 0.6) 50%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.3);
}

.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #c7d2fe 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    font-size: 1.02rem;
    color: #94a3b8;
    margin: 0;
    font-weight: 400;
    line-height: 1.5;
}

.pipeline-badges {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 14px;
}

.badge {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #e2e8f0;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.badge-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10b981;
}

.card-box {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 14px !important;
    padding: 16px !important;
    box-shadow: var(--accent-glow) !important;
    backdrop-filter: blur(12px) !important;
}

.convert-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.15rem !important;
    padding: 14px 28px !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 10px 25px -5px rgba(124, 58, 237, 0.5) !important;
    cursor: pointer !important;
}

.convert-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 15px 30px -5px rgba(219, 39, 119, 0.6) !important;
}

.sheet-preview-box {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px dashed rgba(139, 92, 246, 0.4) !important;
    border-radius: 12px !important;
    padding: 12px !important;
    margin-top: 10px !important;
}

.sheet-btn {
    background: linear-gradient(135deg, #059669 0%, #10b981 50%, #06b6d4 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 12px 20px !important;
    border-radius: 10px !important;
    margin-top: 8px !important;
    box-shadow: 0 8px 20px -4px rgba(16, 185, 129, 0.4) !important;
    cursor: pointer !important;
}

.sheet-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 25px -4px rgba(6, 182, 212, 0.5) !important;
}

.result-card-title {
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
    color: #e2e8f0;
}

.feature-highlight {
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 12px;
}

.danger-btn {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(220, 38, 38, 0.35) !important;
}

.danger-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(220, 38, 38, 0.55) !important;
    background: linear-gradient(135deg, #f87171 0%, #b91c1c 100%) !important;
}

.danger-all-btn {
    background: linear-gradient(135deg, #b91c1c 0%, #7f1d1d 100%) !important;
    border: 1px solid rgba(239, 68, 68, 0.4) !important;
    color: #fee2e2 !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(185, 28, 28, 0.4) !important;
}

.danger-all-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(239, 68, 68, 0.6) !important;
}

.empty-notice-box {
    background: rgba(30, 32, 48, 0.5) !important;
    border: 2px dashed rgba(139, 92, 246, 0.4) !important;
    border-radius: 14px !important;
    padding: 32px 24px !important;
    text-align: center !important;
    color: #94a3b8 !important;
    margin-bottom: 16px !important;
}

.inst-group {
    margin-bottom: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    transition: all 0.3s ease !important;
}

/* Audio Speed Controller & Waveform Styling */
.speed-controller-bar {
    background: rgba(15, 23, 42, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    margin-top: 10px !important;
    margin-bottom: 6px !important;
    backdrop-filter: blur(8px) !important;
}

.speed-label-row {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    margin-bottom: 8px !important;
}

.speed-title {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #cbd5e1 !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}

.speed-badge {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    color: #c084fc !important;
    background: rgba(168, 85, 247, 0.15) !important;
    border: 1px solid rgba(168, 85, 247, 0.35) !important;
    padding: 2px 10px !important;
    border-radius: 9999px !important;
    transition: all 0.2s ease !important;
}

.speed-buttons-row {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
    margin-bottom: 8px !important;
}

.speed-btn {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #cbd5e1 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 5px 11px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.speed-btn:hover {
    background: rgba(99, 102, 241, 0.25) !important;
    border-color: rgba(99, 102, 241, 0.5) !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}

.speed-btn.active {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    border-color: rgba(167, 139, 250, 0.6) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.45) !important;
}

.speed-slider-row {
    display: flex !important;
    flex-direction: column !important;
    gap: 3px !important;
}

.speed-slider {
    width: 100% !important;
    accent-color: #a855f7 !important;
    cursor: pointer !important;
    height: 5px !important;
    border-radius: 3px !important;
    background: rgba(255, 255, 255, 0.12) !important;
}

.speed-slider-ticks {
    display: flex !important;
    justify-content: space-between !important;
    font-size: 0.68rem !important;
    color: #64748b !important;
    padding: 0 4px !important;
}

.shortcuts-legend {
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    gap: 10px !important;
    margin-top: 8px !important;
    font-size: 0.76rem !important;
    color: #94a3b8 !important;
    border-top: 1px dashed rgba(255, 255, 255, 0.08) !important;
    padding-top: 6px !important;
}

.shortcuts-legend kbd {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    font-size: 0.72rem !important;
    color: #f1f5f9 !important;
    font-family: monospace !important;
}

.master-toolbar {
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.5) 0%, rgba(15, 23, 42, 0.7) 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    margin-bottom: 12px !important;
}

.audio-hud-toast {
    position: fixed !important;
    bottom: 30px !important;
    right: 30px !important;
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.95), rgba(49, 46, 129, 0.95)) !important;
    border: 1px solid rgba(167, 139, 250, 0.6) !important;
    color: #ffffff !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    padding: 10px 22px !important;
    border-radius: 9999px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6), 0 0 25px rgba(139, 92, 246, 0.5) !important;
    z-index: 999999 !important;
    opacity: 0 !important;
    transform: translateY(12px) scale(0.94) !important;
    transition: opacity 0.2s cubic-bezier(0.16, 1, 0.3, 1), transform 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    pointer-events: none !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    backdrop-filter: blur(14px) !important;
}

.audio-hud-toast.visible {
    opacity: 1 !important;
    transform: translateY(0) scale(1) !important;
}

/* Real-time Chord Progression HUD */
.chord-hud-container {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    padding: 16px !important;
    margin-bottom: 14px !important;
}

.chord-hud-scroll {
    max-height: 300px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 10px 6px !important;
    margin: 8px 0 !important;
    border-radius: 12px !important;
    background: rgba(10, 15, 29, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    scrollbar-width: thin !important;
    scrollbar-color: rgba(168, 85, 247, 0.6) rgba(15, 23, 42, 0.7) !important;
}

.chord-hud-scroll::-webkit-scrollbar {
    width: 6px !important;
}

.chord-hud-scroll::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.7) !important;
    border-radius: 8px !important;
}

.chord-hud-scroll::-webkit-scrollbar-thumb {
    background: rgba(168, 85, 247, 0.6) !important;
    border-radius: 8px !important;
}

.chord-hud-scroll::-webkit-scrollbar-thumb:hover {
    background: rgba(217, 70, 239, 0.8) !important;
}

.chord-chips-grid {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 12px !important;
    padding: 4px 2px !important;
}

.chord-chip {
    display: inline-flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    min-width: 95px !important;
    padding: 10px 16px !important;
    background: rgba(30, 41, 59, 0.85) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 12px !important;
    transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    user-select: none !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25) !important;
}

.chord-chip:hover {
    border-color: rgba(168, 85, 247, 0.6) !important;
    background: rgba(49, 46, 129, 0.55) !important;
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 6px 14px rgba(168, 85, 247, 0.35) !important;
}

.chord-chip .chip-time {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #cbd5e1 !important;
    margin-bottom: 4px !important;
    letter-spacing: 0.3px !important;
}

.chord-chip .chip-chord {
    font-size: 1.45rem !important;
    font-weight: 800 !important;
    color: #f8fafc !important;
    letter-spacing: 0.5px !important;
    line-height: 1.15 !important;
}

.chord-chip .chip-bar {
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    margin-top: 4px !important;
}

/* Active glowing chord */
.chord-chip.active-chord {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.8) 0%, rgba(217, 70, 239, 0.88) 100%) !important;
    border-color: #f472b6 !important;
    box-shadow: 0 0 20px rgba(217, 70, 239, 0.9), 0 0 35px rgba(124, 58, 237, 0.55), inset 0 0 10px rgba(255, 255, 255, 0.35) !important;
    transform: scale(1.10) translateY(-2px) !important;
    z-index: 10 !important;
}

.chord-chip.active-chord .chip-chord {
    color: #ffffff !important;
    text-shadow: 0 0 12px #fbcfe8, 0 2px 4px rgba(0, 0, 0, 0.5) !important;
}

.chord-chip.active-chord .chip-time {
    color: #fdf2f8 !important;
    font-weight: 700 !important;
    text-shadow: 0 0 6px rgba(255, 255, 255, 0.4) !important;
}

.chord-chip.active-chord .chip-bar {
    color: #fce7f3 !important;
    font-weight: 700 !important;
}

.chord-empty-msg {
    text-align: center !important;
    color: #94a3b8 !important;
    padding: 28px !important;
    font-size: 0.95rem !important;
    font-style: italic !important;
}

/* Mini-Mixer Panel */
.mixer-panel {
    background: linear-gradient(135deg, rgba(20, 24, 45, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    box-shadow: 0 12px 35px -10px rgba(99, 102, 241, 0.25) !important;
}
"""

WAVEFORM_CONFIG = gr.WaveformOptions(
    waveform_color="#6366f1",
    waveform_progress_color="#ec4899",
    show_recording_waveform=False,
    skip_length=5
)


def render_speed_controller(container_id: str) -> str:
    return f"""
<div class="speed-controller-bar" id="speed-ctrl-{container_id}" data-container-id="{container_id}">
    <div class="speed-label-row">
        <span class="speed-title">⚡ Tốc Độ Phát</span>
        <span class="speed-badge">1.0x (Chuẩn)</span>
    </div>
    <div class="speed-buttons-row">
        <button type="button" class="speed-btn" onclick="window.changeAudioSpeed(0.5, '{container_id}')" data-speed="0.5">0.5x (Chậm/Tập đàn)</button>
        <button type="button" class="speed-btn" onclick="window.changeAudioSpeed(0.75, '{container_id}')" data-speed="0.75">0.75x</button>
        <button type="button" class="speed-btn active" onclick="window.changeAudioSpeed(1.0, '{container_id}')" data-speed="1.0">1.0x (Chuẩn)</button>
        <button type="button" class="speed-btn" onclick="window.changeAudioSpeed(1.25, '{container_id}')" data-speed="1.25">1.25x</button>
        <button type="button" class="speed-btn" onclick="window.changeAudioSpeed(1.5, '{container_id}')" data-speed="1.5">1.5x (Nhanh)</button>
    </div>
    <div class="speed-slider-row">
        <input type="range" class="speed-slider" min="0.5" max="1.5" step="0.25" value="1.0" 
               oninput="window.changeAudioSpeed(parseFloat(this.value), '{container_id}')">
        <div class="speed-slider-ticks">
            <span>0.5x</span>
            <span>0.75x</span>
            <span>1.0x</span>
            <span>1.25x</span>
            <span>1.5x</span>
        </div>
    </div>
    <div class="shortcuts-legend">
        <span>⌨️ Phím tắt:</span>
        <span class="legend-item"><kbd>Space</kbd> Play/Pause</span>
        <span class="legend-item"><kbd>←</kbd> / <kbd>→</kbd> Tua ±5s</span>
        <span class="legend-item"><kbd>[</kbd> / <kbd>]</kbd> Tốc độ ±0.25x</span>
    </div>
</div>
"""


def render_master_toolbar(tab_id: str) -> str:
    return f"""
<div class="master-toolbar speed-controller-bar" id="speed-ctrl-{tab_id}" data-container-id="{tab_id}">
    <div class="speed-label-row">
        <span class="speed-title">⚡ Điều Khiển Tốc Độ & Phím Tắt Chung</span>
        <span class="speed-badge">1.0x (Chuẩn)</span>
    </div>
    <div class="speed-buttons-row">
        <button type="button" class="speed-btn" onclick="window.changeAudioSpeed(0.5, '{tab_id}')" data-speed="0.5">0.5x (Chậm/Tập đàn)</button>
        <button type="button" class="speed-btn" onclick="window.changeAudioSpeed(0.75, '{tab_id}')" data-speed="0.75">0.75x</button>
        <button type="button" class="speed-btn active" onclick="window.changeAudioSpeed(1.0, '{tab_id}')" data-speed="1.0">1.0x (Chuẩn)</button>
        <button type="button" class="speed-btn" onclick="window.changeAudioSpeed(1.25, '{tab_id}')" data-speed="1.25">1.25x</button>
        <button type="button" class="speed-btn" onclick="window.changeAudioSpeed(1.5, '{tab_id}')" data-speed="1.5">1.5x (Nhanh)</button>
    </div>
    <div class="shortcuts-legend">
        <span>⌨️ <b>Phím tắt bàn phím:</b></span>
        <span class="legend-item"><kbd>Space</kbd> Phát / Tạm dừng</span>
        <span class="legend-item"><kbd>←</kbd> / <kbd>→</kbd> Tua ±5 giây</span>
        <span class="legend-item"><kbd>[</kbd> / <kbd>]</kbd> Giảm / Tăng tốc độ (-0.25x / +0.25x)</span>
    </div>
</div>
"""


PLAYER_ENHANCEMENT_JS = """
(function() {
    window.__currentAudioSpeed = 1.0;
    window.__lastActiveAudio = null;
    window.__hudTimer = null;

    function formatTime(seconds) {
        if (isNaN(seconds) || seconds < 0) return '00:00';
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s;
    }

    function showHudToast(text) {
        let hud = document.getElementById('audio-hud-toast');
        if (!hud) {
            hud = document.createElement('div');
            hud.id = 'audio-hud-toast';
            hud.className = 'audio-hud-toast';
            document.body.appendChild(hud);
        }
        hud.innerHTML = text;
        hud.classList.add('visible');
        if (window.__hudTimer) clearTimeout(window.__hudTimer);
        window.__hudTimer = setTimeout(() => {
            hud.classList.remove('visible');
        }, 1300);
    }

    function applyPlaybackRateToAudio(audio, speed) {
        if (!audio) return;
        try {
            audio.playbackRate = speed;
            audio.preservesPitch = true;
            if ('webkitPreservesPitch' in audio) audio.webkitPreservesPitch = true;
            if ('mozPreservesPitch' in audio) audio.mozPreservesPitch = true;
        } catch (e) {}
    }

    window.changeAudioSpeed = function(speed, containerId) {
        speed = parseFloat(speed);
        if (isNaN(speed)) speed = 1.0;
        speed = Math.max(0.5, Math.min(2.0, speed));
        window.__currentAudioSpeed = speed;

        let audios = [];
        if (containerId && containerId !== 'global' && containerId !== 'tab1' && containerId !== 'tab2') {
            const ctrl = document.getElementById('speed-ctrl-' + containerId);
            if (ctrl) {
                const group = ctrl.closest('.inst-group') || ctrl.parentElement;
                if (group) {
                    audios = Array.from(group.querySelectorAll('audio'));
                }
            }
        }
        if (audios.length === 0) {
            audios = Array.from(document.querySelectorAll('audio'));
        }

        audios.forEach(a => applyPlaybackRateToAudio(a, speed));

        const speedBars = (containerId && containerId !== 'global' && containerId !== 'tab1' && containerId !== 'tab2')
            ? document.querySelectorAll('#speed-ctrl-' + containerId)
            : document.querySelectorAll('.speed-controller-bar');

        speedBars.forEach(bar => {
            bar.querySelectorAll('.speed-btn').forEach(btn => {
                const btnSpeed = parseFloat(btn.getAttribute('data-speed'));
                if (Math.abs(btnSpeed - speed) < 0.01) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
            const slider = bar.querySelector('.speed-slider');
            if (slider) slider.value = speed;
            const badge = bar.querySelector('.speed-badge');
            if (badge) {
                let label = speed + 'x';
                if (speed === 0.5) label = '0.5x (Chậm/Tập đàn)';
                else if (speed === 1.0) label = '1.0x (Chuẩn)';
                else if (speed === 1.5) label = '1.5x (Nhanh)';
                badge.innerText = label;
            }
        });

        let speedLabel = speed + 'x';
        if (speed === 0.5) speedLabel = '0.5x (Chậm/Tập đàn)';
        else if (speed === 1.0) speedLabel = '1.0x (Chuẩn)';
        else if (speed === 1.5) speedLabel = '1.5x (Nhanh)';
        showHudToast('⚡ Tốc độ: ' + speedLabel);
    };

    window.setGlobalAudioSpeed = function(speed) {
        window.changeAudioSpeed(speed, 'global');
    };

    document.addEventListener('play', function(e) {
        if (e.target && e.target.tagName === 'AUDIO') {
            window.__lastActiveAudio = e.target;
            const speed = window.__currentAudioSpeed || 1.0;
            applyPlaybackRateToAudio(e.target, speed);
        }
    }, true);

    document.addEventListener('loadedmetadata', function(e) {
        if (e.target && e.target.tagName === 'AUDIO') {
            const speed = window.__currentAudioSpeed || 1.0;
            applyPlaybackRateToAudio(e.target, speed);
        }
    }, true);

    document.addEventListener('keydown', function(e) {
        const active = document.activeElement;
        if (active) {
            const tag = active.tagName.toLowerCase();
            if (tag === 'input' || tag === 'textarea' || tag === 'select' || active.isContentEditable) {
                return;
            }
        }

        let audio = window.__lastActiveAudio;
        if (!audio || !document.body.contains(audio)) {
            const allAudios = Array.from(document.querySelectorAll('audio'));
            audio = allAudios.find(a => !a.paused) || allAudios.find(a => a.src && a.src !== '');
        }

        if (!audio) return;

        if (e.code === 'Space') {
            e.preventDefault();
            if (audio.paused) {
                audio.play().then(() => {
                    showHudToast('▶️ Tiếp tục phát');
                }).catch(() => {});
            } else {
                audio.pause();
                showHudToast('⏸️ Tạm dừng');
            }
        } else if (e.code === 'ArrowLeft') {
            e.preventDefault();
            audio.currentTime = Math.max(0, audio.currentTime - 5);
            showHudToast('⏪ Tua lùi 5s (' + formatTime(audio.currentTime) + ')');
        } else if (e.code === 'ArrowRight') {
            e.preventDefault();
            const maxTime = audio.duration || (audio.currentTime + 5);
            audio.currentTime = Math.min(maxTime, audio.currentTime + 5);
            showHudToast('⏩ Tua tới 5s (' + formatTime(audio.currentTime) + ')');
        } else if (e.key === '[' || e.code === 'BracketLeft') {
            e.preventDefault();
            const currentSpeed = window.__currentAudioSpeed || 1.0;
            const newSpeed = Math.max(0.5, Math.round((currentSpeed - 0.25) * 100) / 100);
            window.setGlobalAudioSpeed(newSpeed);
        } else if (e.key === ']' || e.code === 'BracketRight') {
            e.preventDefault();
            const currentSpeed = window.__currentAudioSpeed || 1.0;
            const newSpeed = Math.min(1.5, Math.round((currentSpeed + 0.25) * 100) / 100);
            window.setGlobalAudioSpeed(newSpeed);
        }
    });

    // Real-time Chord Progression Active Glow Sync & Auto-Scroll
    function syncActiveChord(currentTime) {
        const grids = document.querySelectorAll('.chord-chips-grid');
        grids.forEach(grid => {
            const chips = grid.querySelectorAll('.chord-chip');
            chips.forEach(chip => {
                const start = parseFloat(chip.getAttribute('data-start'));
                const end = parseFloat(chip.getAttribute('data-end'));
                if (currentTime >= start && currentTime < end) {
                    if (!chip.classList.contains('active-chord')) {
                        chip.classList.add('active-chord');
                        try {
                            chip.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
                        } catch (err) {}
                    }
                } else {
                    chip.classList.remove('active-chord');
                }
            });
        });
    }

    document.addEventListener('timeupdate', function(e) {
        if (e.target && e.target.tagName === 'AUDIO') {
            syncActiveChord(e.target.currentTime);
        }
    }, true);

    document.addEventListener('click', function(e) {
        const chip = e.target.closest('.chord-chip');
        if (chip) {
            const start = parseFloat(chip.getAttribute('data-start'));
            if (!isNaN(start)) {
                let audio = window.__lastActiveAudio || document.querySelector('audio');
                if (audio) {
                    audio.currentTime = start;
                    if (audio.paused) {
                        audio.play().catch(() => {});
                    }
                }
            }
        }
    });
})();
"""



def render_chord_chips_html(chords: Optional[List[Dict[str, Any]]]) -> str:
    """Generate interactive glowing HTML chips for chord progression."""
    if not chords:
        return (
            '<div class="chord-hud-scroll">'
            '<div class="chord-empty-msg">Chưa có dữ liệu hợp âm. Chuyển đổi bài hát để xem vòng hợp âm đồng bộ thời gian thực!</div>'
            '</div>'
        )

    html_parts = ['<div class="chord-hud-scroll"><div class="chord-chips-grid" id="chord-chips-grid">']
    for c in chords:
        start_t = float(c.get("time", 0.0))
        end_t = float(c.get("end_time", start_t + 2.0))
        st_m, st_s = divmod(int(start_t), 60)
        time_str = f"{st_m:02d}:{st_s:02d}"
        chord_name = c.get("chord", "C")
        bar_num = c.get("bar", 1)
        html_parts.append(
            f'<div class="chord-chip" data-start="{start_t:.2f}" data-end="{end_t:.2f}" title="Nhấn để tua tới {time_str}">'
            f'<span class="chip-time">{time_str}</span>'
            f'<span class="chip-chord">{chord_name}</span>'
            f'<span class="chip-bar">Ô {bar_num}</span>'
            f'</div>'
        )
    html_parts.append('</div></div>')
    return "".join(html_parts)


def get_history_dropdown_choices():
    if not processor:
        return []
    entries = processor.list_history_entries()
    choices = []
    for e in entries:
        insts = e.get("selected_instruments", [])
        if not insts and "instruments" in e:
            insts = list(e["instruments"].keys())
        inst_icons = "".join([INSTRUMENT_CONFIGS[k]["name"][:2] for k in insts if k in INSTRUMENT_CONFIGS]) or "🎹🎸"
        title = f"📅 {e.get('created_at', '')} | {e.get('song_title', e['id'])} [{inst_icons} | {e.get('mode', 'Tiêu chuẩn')} - {e.get('transpose', '0')}]"
        choices.append((title, e["id"]))
    return choices


def handle_sheet_file_change(sheet_file_path: Optional[str]):
    """
    Handle sheet music file upload:
    - If image (.png, .jpg, .jpeg, .webp, .bmp), display directly.
    - If PDF, extract first page image.
    - If MusicXML / MIDI, display musical notation details.
    """
    if not sheet_file_path:
        return None, "📄 *Chưa tải lên sheet nhạc.*"

    p = Path(sheet_file_path)
    ext = p.suffix.lower()

    if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        file_sz_kb = round(p.stat().st_size / 1024, 1)
        return str(p), f"✅ **Hình ảnh Sheet nhạc:** `{p.name}` ({file_sz_kb} KB)\n*Sẵn sàng quét OMR nhận diện khuông nhạc, khóa nhạc và nốt.*"

    if ext == ".pdf":
        try:
            omr = SheetMusicOMR()
            page_img = omr._extract_first_page_from_pdf(p)
            return str(page_img), f"✅ **Tài liệu PDF Sheet nhạc:** `{p.name}` (Trang 1 đã trích xuất)\n*Sẵn sàng quét OMR nhận diện nốt nhạc.*"
        except Exception as e:
            return None, f"⚠️ **Tệp PDF:** `{p.name}` (Không thể trích xuất trang xem trước: {e})\n*Hệ thống vẫn sẽ thử xử lý OMR khi bấm Bắt đầu.*"

    if ext in [".musicxml", ".xml", ".mxl"]:
        file_sz_kb = round(p.stat().st_size / 1024, 1)
        return None, f"🎼 **Ký âm số MusicXML:** `{p.name}` ({file_sz_kb} KB)\n*Tệp ký âm số hóa chuẩn xác 100%. Sẽ được phân tích trực tiếp thành MIDI hoàn hảo!*"

    if ext in [".mid", ".midi"]:
        file_sz_kb = round(p.stat().st_size / 1024, 1)
        return None, f"🎹 **Tệp MIDI chuẩn:** `{p.name}` ({file_sz_kb} KB)\n*Sẵn sàng nạp vào dàn SoundFonts phòng thu & Mastering EBU R128!*"

    return None, f"📄 **Tệp đã tải lên:** `{p.name}`"


DEFAULT_SELECTED_KEYS = [ALL_INSTRUMENTS[0], ALL_INSTRUMENTS[1]]


def convert_song_to_covers(
    file_path,
    url_input="",
    sheet_file=None,
    cover_mode="Đệm Hát Tiêu Chuẩn (Chords Only)",
    transpose_val=0,
    tempo_multiplier=1.0,
    selected_inst_names=None,
    soundfont_choice="Studio Hi-Fi SoundFonts (Steinway Grand & Studio Guitars - Khuyên dùng)",
    acoustic_preset="Phòng Thu Acoustic (Studio Warmth) - Mặc định",
    preview_mode=False,
    preview_start=30.0,
    preview_duration=30.0,
    onset_thresh=0.5,
    frame_thresh=0.3,
    min_length=110.0,
    vocal_vol=0.85,
    inst_vol=1.25,
    include_bass=True,
    min_midi_dur=60.0,
    min_pitch=36,
    reverb_room=0.68,
    reverb_lvl=0.72,
    apply_enhancer=True,
    progress=gr.Progress(track_tqdm=True)
):
    if processor is None:
        raise gr.Error("AudioProcessor chưa được khởi tạo. Vui lòng kiểm tra FFmpeg và FluidSynth.")

    def progress_handler(fraction: float, message: str):
        progress(fraction, desc=message)

    # Resolve target input source
    target_input_path = None
    target_song_name = None

    if sheet_file:
        target_input_path = sheet_file
        target_song_name = Path(sheet_file).name
    elif url_input and url_input.strip():
        try:
            progress_handler(0.02, f"Đang tải âm thanh chất lượng cao từ URL bằng yt-dlp...")
            downloaded_file, extracted_title = download_audio_from_url(url_input.strip())
            target_input_path = str(downloaded_file)
            target_song_name = extracted_title
        except Exception as url_err:
            logger.error(f"Lỗi khi tải từ URL: {url_err}")
            raise gr.Error(f"Không thể tải âm thanh từ link: {str(url_err)}")
    elif file_path:
        target_input_path = file_path
        target_song_name = Path(file_path).name
    else:
        raise gr.Error("Vui lòng tải lên tệp âm thanh/video, bản sheet nhạc (.png/.jpg/.pdf/.musicxml) HOẶC nhập đường link URL!")

    # Validation: Map selection to canonical ALL_INSTRUMENTS keys
    if not selected_inst_names:
        raise gr.Error("Vui lòng tích chọn ít nhất 1 nhạc cụ ở danh sách bên trái để xuất bản cover!")

    selected_keys = []
    for n in (selected_inst_names or []):
        norm = normalize_instrument_key(n)
        if norm and norm in ALL_INSTRUMENTS and norm not in selected_keys:
            selected_keys.append(norm)

    if not selected_keys:
        raise gr.Error("Vui lòng tích chọn ít nhất 1 nhạc cụ ở danh sách bên trái để xuất bản cover!")

    # Safe defaults for dropdowns if None
    if not soundfont_choice:
        soundfont_choice = "Studio Hi-Fi SoundFonts"
    if not acoustic_preset:
        acoustic_preset = "Phòng Thu Acoustic (Studio Warmth) - Mặc định"

    try:
        sf_preset = "studio" if "Studio" in str(soundfont_choice) else "standard"
        mode_val = "solo" if "Solo" in str(cover_mode) or "Fingerstyle" in str(cover_mode) else "chords"
        
        # Only render instruments selected by user
        results = processor.process(
            input_path=target_input_path,
            selected_instruments=selected_keys,
            cover_mode=mode_val,
            transpose_semitones=int(transpose_val),
            tempo_multiplier=float(tempo_multiplier),
            soundfont_preset=sf_preset,
            acoustic_preset=acoustic_preset,
            onset_threshold=float(onset_thresh),
            frame_threshold=float(frame_thresh),
            min_note_length_ms=float(min_length),
            vocal_volume=float(vocal_vol),
            inst_volume=float(inst_vol),
            include_bass=bool(include_bass),
            min_pitch=int(min_pitch),
            min_midi_duration_ms=float(min_midi_dur),
            reverb_room_size=float(reverb_room),
            reverb_level=float(reverb_lvl),
            apply_reverb_enhancer=bool(apply_enhancer),
            preview_mode=bool(preview_mode),
            preview_start_sec=float(preview_start),
            preview_duration_sec=float(preview_duration),
            progress_callback=progress_handler
        )

        new_choices = get_history_dropdown_choices()
        updated_dropdown = gr.Dropdown(choices=new_choices, value=results["history_id"])

        inst_badges = ", ".join([INSTRUMENT_CONFIGS[k]["name"] for k in selected_keys if k in INSTRUMENT_CONFIGS])
        preset_name = results.get("acoustic_preset", "Phòng Thu Acoustic")
        display_title = results.get('song_title', target_song_name)
        tempo_str = f" | Tốc độ: {float(tempo_multiplier):.2f}x" if abs(float(tempo_multiplier) - 1.0) >= 0.01 else ""
        if is_sheet_music_file(target_input_path):
            success_msg = f"🎉 Quét Sheet nhạc & Chuyển đổi thành công: **{display_title}** ({preset_name} | Tone: {int(transpose_val):+d}{tempo_str})\n\n**Nhạc cụ:** {inst_badges}"
        else:
            success_msg = f"🎉 Chuyển đổi thành công: **{display_title}** ({cover_mode} | {preset_name} | Tone: {int(transpose_val):+d}{tempo_str})\n\n**Nhạc cụ:** {inst_badges}"

        # Clean 1-to-1 audio outputs following ALL_INSTRUMENTS strictly:
        # Each instrument produces exactly 2 outputs (Instrumental, With Vocals)
        audio_outputs = []
        for k in ALL_INSTRUMENTS:
            if k in selected_keys:
                k_res = results.get("instruments", {}).get(k, {})
                if not k_res and k == "flute":
                    k_res = results.get("instruments", {}).get("bamboo_flute", {})
                elif not k_res and k == "bamboo_flute":
                    k_res = results.get("instruments", {}).get("flute", {})
                audio_outputs.append(k_res.get("instrumental"))
                audio_outputs.append(k_res.get("with_vocals"))
            else:
                audio_outputs.append(None)
                audio_outputs.append(None)

        empty_notice_update = gr.Markdown(visible=False)
        group_updates = [gr.Group(visible=(k in selected_keys)) for k in ALL_INSTRUMENTS]

        chord_chips_html = render_chord_chips_html(results.get("chord_progression"))
        chord_file = results.get("chord_txt_file")

        return (
            empty_notice_update,
            *group_updates,
            *audio_outputs,
            results["midi_file"],
            chord_chips_html,
            chord_file,
            success_msg,
            updated_dropdown,
            results["history_id"]
        )

    except Exception as e:
        logger.error(f"Error during conversion: {e}", exc_info=True)
        raise gr.Error(f"Quá trình xử lý thất bại: {str(e)}")


def load_history_song(entry_id):
    empty_audio = [None] * (len(ALL_INSTRUMENTS) * 2)
    empty_groups = [gr.Group(visible=False) for _ in ALL_INSTRUMENTS]

    if not entry_id or not processor:
        return (
            gr.Markdown(visible=True),
            *empty_groups,
            *empty_audio,
            None,
            render_chord_chips_html(None),
            None,
            "💡 *Vui lòng chọn một bài hát từ danh sách lịch sử bên trên.*",
            ""
        )

    meta = processor.get_history_entry_files(entry_id)
    if not meta:
        return (
            gr.Markdown(visible=True),
            *empty_groups,
            *empty_audio,
            None,
            render_chord_chips_html(None),
            None,
            "⚠️ Không tìm thấy dữ liệu tệp của bài hát đã chọn.",
            ""
        )

    files = meta.get("files", {})
    inst_dict = meta.get("instruments", {})
    midi_path = files.get("midi_file")

    audio_res = []
    generated_names = []
    present_keys = set()

    for k in ALL_INSTRUMENTS:
        k_data = inst_dict.get(k, {})
        inst_track = k_data.get("instrumental") or files.get(f"{k}_instrumental")
        voc_track = k_data.get("with_vocals") or files.get(f"{k}_with_vocals") or files.get(f"{k}_vocal")
        if not inst_track and k == "flute":
            inst_track = files.get("bamboo_flute_instrumental") or files.get("flute_instrumental") or inst_dict.get("bamboo_flute", {}).get("instrumental") or inst_dict.get("flute", {}).get("instrumental")
        if not voc_track and k == "flute":
            voc_track = files.get("bamboo_flute_with_vocals") or files.get("flute_with_vocals") or files.get("bamboo_flute_vocal") or files.get("flute_vocal") or inst_dict.get("bamboo_flute", {}).get("with_vocals") or inst_dict.get("flute", {}).get("with_vocals")

        audio_res.append(inst_track)
        audio_res.append(voc_track)
        if inst_track or voc_track:
            present_keys.add(k)
            name_disp = "🎋 Sáo Trúc Hòa Tấu" if k == "flute" else INSTRUMENT_CONFIGS[k]["name"]
            generated_names.append(name_disp)

    inst_summary = ", ".join(generated_names) if generated_names else "Không rõ"
    tempo_disp = f" ({meta.get('estimated_tempo_bpm')} BPM)" if meta.get("estimated_tempo_bpm") else ""
    info_md = f"""
### 🎶 Chi Tiết Bản Thu: **{meta.get('song_title', entry_id)}**
- **🎼 Nhạc Cụ Đã Xuất:** {inst_summary}
- **🏛️ Không Gian Biểu Diễn:** `{meta.get('acoustic_preset', 'Phòng Thu Acoustic (Studio Warmth)')}`
- **🎹 Phong Cách:** `{meta.get('mode', 'Đệm Hát Tiêu Chuẩn')}`
- **🎵 Tone / Transpose:** `{meta.get('transpose', 'Tone gốc (0)')}`{tempo_disp}
- **📅 Thời Gian:** `{meta.get('created_at', 'N/A')}`
- **📦 Định Dạng:** MP3 Studio 320kbps (EBU R128 Loudnorm -14 LUFS)
- **📂 Vị Trí Lưu:** `outputs/history/{entry_id}`
"""
    notice_update = gr.Markdown(visible=(len(present_keys) == 0))
    group_updates = [gr.Group(visible=(k in present_keys)) for k in ALL_INSTRUMENTS]

    chord_prog = meta.get("chord_progression")
    chord_chips_html = render_chord_chips_html(chord_prog)
    chord_file = files.get("chord_txt_file")
    run_dir = processor.history_dir / entry_id
    if not chord_file or not Path(chord_file).exists():
        cand = run_dir / f"{meta.get('song_title', entry_id)}_chords.txt"
        if cand.exists():
            chord_file = str(cand.resolve())

    return (
        notice_update,
        *group_updates,
        *audio_res,
        midi_path,
        chord_chips_html,
        chord_file,
        info_md,
        entry_id
    )


def refresh_history():
    choices = get_history_dropdown_choices()
    default_val = choices[0][1] if choices else None
    empty_audio = [None] * (len(ALL_INSTRUMENTS) * 2)
    empty_groups = [gr.Group(visible=False) for _ in ALL_INSTRUMENTS]

    if default_val:
        notice_up, *rest = load_history_song(default_val)
        return (
            gr.Dropdown(choices=choices, value=default_val),
            notice_up,
            *rest
        )
    return (
        gr.Dropdown(choices=[], value=None),
        gr.Markdown(visible=True),
        *empty_groups,
        *empty_audio,
        None,
        render_chord_chips_html(None),
        None,
        "💡 *Danh sách lịch sử hiện đang trống.*",
        ""
    )


def delete_single_history(entry_id):
    empty_audio = [None] * (len(ALL_INSTRUMENTS) * 2)
    empty_groups = [gr.Group(visible=False) for _ in ALL_INSTRUMENTS]

    if not entry_id or not processor:
        return (
            gr.Dropdown(),
            gr.Markdown(visible=True),
            *empty_groups,
            *empty_audio,
            None,
            render_chord_chips_html(None),
            None,
            "⚠️ *Chưa chọn bài hát để xóa hoặc danh sách lịch sử hiện đang trống.*",
            ""
        )

    processor.delete_history_entry(entry_id)
    new_choices = get_history_dropdown_choices()
    new_default = new_choices[0][1] if new_choices else None

    if new_default:
        notice_up, *rest = load_history_song(new_default)
        rest_list = list(rest)
        rest_list[-2] = f"🗑️ **Đã xóa bài hát `{entry_id}` thành công!**\n\n---\n" + str(rest_list[-2])
        return (
            gr.Dropdown(choices=new_choices, value=new_default),
            notice_up,
            *rest_list
        )
    else:
        return (
            gr.Dropdown(choices=[], value=None),
            gr.Markdown(visible=True),
            *empty_groups,
            *empty_audio,
            None,
            render_chord_chips_html(None),
            None,
            f"🗑️ **Đã xóa bài hát `{entry_id}` thành công!**\n\n💡 *Danh sách lịch sử hiện đang trống.*",
            ""
        )


def clear_all_history_entries():
    empty_audio = [None] * (len(ALL_INSTRUMENTS) * 2)
    empty_groups = [gr.Group(visible=False) for _ in ALL_INSTRUMENTS]

    if not processor:
        return (
            gr.Dropdown(choices=[], value=None),
            gr.Markdown(visible=True),
            *empty_groups,
            *empty_audio,
            None,
            render_chord_chips_html(None),
            None,
            "⚠️ *AudioProcessor chưa được khởi tạo.*",
            ""
        )

    count = processor.clear_all_history()
    return (
        gr.Dropdown(choices=[], value=None),
        gr.Markdown(visible=True),
        *empty_groups,
        *empty_audio,
        None,
        render_chord_chips_html(None),
        None,
        f"⚠️ **Đã xóa toàn bộ {count} bản thu trong thư mục `outputs/history/` thành công!**\n\n💡 *Danh sách lịch sử hiện đang trống.*",
        ""
    )


def handle_auto_detect_key(file_path, url_input="", sheet_file=None):
    """Auto-detect musical key and BPM from uploaded file, URL input, or Sheet Music."""
    target_path = None
    if sheet_file:
        try:
            sp = Path(sheet_file)
            ext = sp.suffix.lower()
            if ext in [".musicxml", ".xml", ".mxl", ".mid", ".midi"]:
                import music21
                score = music21.converter.parse(str(sp))
                k = score.analyze('key')
                return f"🎵 **Tone từ Ký âm Sheet:** **{k.tonic.name} {k.mode.capitalize()}** | ⚡ **Nhịp:** 120.0 BPM", f"{k.tonic.name} {k.mode.capitalize()}"
            elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf"]:
                omr = SheetMusicOMR()
                res = omr.process(sp, generate_overlay=False)
                return f"🎵 **Tone từ Sheet OMR:** **{res['detected_key']}** (Độ tin cậy: {int(res['detected_key_confidence']*100)}%) | ⚡ **BPM:** {res['estimated_tempo_bpm']:.1f}", res['detected_key']
        except Exception as sheet_err:
            logger.warning(f"Key detection on sheet music failed: {sheet_err}")
            return f"⚠️ *Không thể phân tích Tone từ Sheet: {sheet_err}*", ""
    elif url_input and url_input.strip():
        try:
            downloaded_file, _ = download_audio_from_url(url_input.strip())
            target_path = Path(downloaded_file)
        except Exception as e:
            logger.warning(f"Could not download audio from URL for key/BPM detection: {e}")
            return "⚠️ *Không thể trích xuất âm thanh từ link URL để nhận diện.*", ""
    elif file_path:
        target_path = Path(file_path)

    if not target_path or not target_path.exists():
        return "⚠️ *Vui lòng tải lên file, sheet nhạc hoặc dán link URL trước khi nhấn Nhận diện!*", ""

    try:
        key_info = detect_musical_key(target_path, max_duration_sec=60.0)
        bpm_val, _ = extract_beats_and_tempo(target_path, max_duration_sec=60.0)
        display_md = f"🎵 **Tone gốc phát hiện:** **{key_info['label']}** (Độ tin cậy: {int(key_info['confidence']*100)}%) | ⚡ **BPM gốc:** **{bpm_val:.1f} BPM**"
        return display_md, key_info["label"]
    except Exception as e:
        logger.error(f"Error during key/BPM detection: {e}")
        return f"⚠️ *Không thể nhận diện Tone & BPM: {e}*", ""


def mix_custom_stems(
    current_entry_id: str,
    vocal_vol: float = 1.0,
    inst_vol: float = 1.0,
    bass_vol: float = 0.8,
    reverb_pct: float = 25.0,
    transpose_val: int = 0
):
    """Mix vocal, instrument, and acoustic bass stems with live remastering."""
    if not processor:
        raise gr.Error("AudioProcessor chưa được khởi tạo.")
    if not current_entry_id:
        raise gr.Error("Vui lòng chuyển đổi một bài hát hoặc chọn bài từ lịch sử trước khi phối bè!")

    meta = processor.get_history_entry_files(current_entry_id)
    if not meta:
        raise gr.Error(f"Không tìm thấy dữ liệu tệp của bài hát {current_entry_id}!")

    files = meta.get("files", {})
    run_dir = processor.history_dir / current_entry_id
    clean_title = meta.get("song_title", current_entry_id)

    vocals_p = files.get("vocals_wav")
    if not vocals_p or not Path(vocals_p).exists():
        cand = run_dir / f"{clean_title}_Vocals.wav"
        if cand.exists():
            vocals_p = str(cand)

    bass_p = files.get("acoustic_bass_wav")
    if not bass_p or not Path(bass_p).exists():
        cand = run_dir / f"{clean_title}_Acoustic_Bass.wav"
        if cand.exists():
            bass_p = str(cand)

    inst_p = files.get("primary_instrument_wav")
    if not inst_p or not Path(inst_p).exists():
        cand = run_dir / f"{clean_title}_Primary_Instrument.wav"
        if cand.exists():
            inst_p = str(cand)

    if not inst_p or not Path(inst_p).exists():
        for k in ALL_INSTRUMENTS:
            cand = files.get(f"{k}_instrumental")
            if cand and Path(cand).exists():
                inst_p = cand
                break

    if not inst_p or not Path(inst_p).exists():
        raise gr.Error("Không tìm thấy tệp nhạc cụ đệm nền để thực hiện mix!")

    output_master_mp3 = run_dir / f"{clean_title}_Custom_Mix_Master.mp3"

    try:
        processor.mix_stems_master(
            vocals_wav=Path(vocals_p) if (vocals_p and Path(vocals_p).exists()) else None,
            instrument_wav=Path(inst_p),
            bass_wav=Path(bass_p) if (bass_p and Path(bass_p).exists()) else None,
            output_mp3=output_master_mp3,
            vocal_volume=float(vocal_vol),
            inst_volume=float(inst_vol),
            bass_volume=float(bass_vol),
            reverb_send_pct=float(reverb_pct),
            transpose_semitones=int(transpose_val)
        )
        return str(output_master_mp3.resolve()), f"🎉 **Đã xuất bản mix đa kênh thành công!** (`{output_master_mp3.name}` - {round(output_master_mp3.stat().st_size / 1024, 1)} KB)"
    except Exception as mix_err:
        logger.error(f"Lỗi khi mix stems: {mix_err}", exc_info=True)
        raise gr.Error(f"Quá trình mix thất bại: {str(mix_err)}")


def handle_target_key_change(target_choice, detected_key_label):
    """Calculate semitone transpose delta when user selects a target key center."""
    if not detected_key_label or not target_choice or "giữ nguyên" in str(target_choice).lower():
        return 0
    delta = calculate_smart_transpose(detected_key_label, target_choice)
    return int(delta)


def handle_clear_stem_cache():
    """Clear Demucs stems cache and return feedback message."""
    if not processor:
        return "⚠️ *AudioProcessor chưa được khởi tạo.*"
    count = processor.clear_stem_cache()
    return f"🧹 **Đã dọn dẹp sạch sẽ {count} mục cache Demucs stems tại `temp/cache_stems/` thành công!**"


with gr.Blocks(title="Song to Multi-Instrument Cover Converter") as demo:
    gr.HTML(f"""
        <div class="hero-banner">
            <h1 class="hero-title">🎹 AI Multi-Instrument Cover Converter 🎸🎻🎷🎋</h1>
            <p class="hero-subtitle">
                Chuyển đổi bài hát thành bản Cover đa nhạc cụ: Piano, Acoustic Guitars, Sáo Trúc / Tiêu, Dàn Dây Strings, Alto Sax, Rhodes, Harp & Music Box.<br/>
                Hỗ trợ Solo / Fingerstyle (Giai điệu + Hợp âm), 🎋 Sáo Trúc Hòa Tấu (Lead + Chords Backing), Smart Sustain (CC 64), SoundFonts Phòng Thu Hi-Fi, Mastering chuẩn Spotify/Apple.
            </p>
            <div class="pipeline-badges">
                <span class="badge"><span class="badge-dot"></span> Demucs 4-Stem (--shifts=1)</span>
                <span class="badge"><span class="badge-dot"></span> 9 Nhạc Cụ Phòng Thu</span>
                <span class="badge"><span class="badge-dot"></span> 🎋 Sáo Trúc Hòa Tấu (Lead + Chords)</span>
                <span class="badge"><span class="badge-dot"></span> Solo / Fingerstyle Engine</span>
                <span class="badge"><span class="badge-dot"></span> Guitar Strumming Delay (15-25ms)</span>
                <span class="badge"><span class="badge-dot"></span> Smart Sustain & Humanization (CC 64)</span>
                <span class="badge"><span class="badge-dot"></span> Vocal Clean (HPF 85Hz + De-Essing)</span>
                <span class="badge"><span class="badge-dot"></span> Sidechain Ducking Compressor</span>
                <span class="badge"><span class="badge-dot"></span> Steinway Grand (74MB) & Guitars (150MB)</span>
                <span class="badge"><span class="badge-dot" style="background: #38bdf8;"></span> 🎼 OMR Sheet Music (Ảnh / MusicXML)</span>
                <span class="badge"><span class="badge-dot"></span> 3 Acoustic Presets</span>
                <span class="badge"><span class="badge-dot"></span> MP3 320kbps EBU R128 (-14 LUFS)</span>
                <span class="badge"><span class="badge-dot"></span> {device_label}</span>
            </div>
        </div>
    """)

    with gr.Tabs():
        # TAB 1: NEW CONVERSION
        with gr.TabItem("✨ 1. Chuyển Đổi Mới"):
            with gr.Row():
                # Left column: Upload & Configuration
                with gr.Column(scale=5):
                    gr.Markdown("### 📥 Nguồn Đầu Vào (Âm Thanh, Link URL HOẶC Sheet Nhạc)")
                    with gr.Tabs():
                        with gr.TabItem("📁 Tải Lên Tệp Audio/Video"):
                            input_audio = gr.File(
                                label="Chọn hoặc kéo thả tệp (.mp3, .mp4, .wav, .m4a, .flac, .ogg)",
                                file_types=["audio", "video", ".mp3", ".wav", ".mp4", ".m4a", ".flac", ".ogg"],
                                type="filepath",
                                elem_classes=["card-box"]
                            )
                        with gr.TabItem("🔗 Dán Link URL (YouTube / TikTok / SoundCloud)"):
                            url_input = gr.Textbox(
                                label="Dán đường dẫn video hoặc nhạc trực tiếp",
                                placeholder="https://www.youtube.com/watch?v=... hoặc link TikTok, SoundCloud",
                                info="⚡ Hệ thống tự động trích xuất âm thanh tốt nhất bằng yt-dlp.",
                                lines=2,
                                elem_classes=["card-box"]
                            )
                        with gr.TabItem("🎼 Tải Lên Sheet Nhạc (Ảnh / MusicXML)"):
                            input_sheet = gr.File(
                                label="Tải ảnh bản nhạc (.png, .jpg, .pdf) hoặc file MusicXML (.musicxml, .mxl, .mid)",
                                file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf", ".musicxml", ".xml", ".mxl", ".mid", ".midi"],
                                type="filepath",
                                elem_classes=["card-box"]
                            )
                            with gr.Group(elem_classes=["sheet-preview-box"]):
                                sheet_preview = gr.Image(
                                    label="👁️ Xem Trước Bản Nhạc (Sheet Music Preview)",
                                    type="filepath",
                                    height=240,
                                    interactive=False
                                )
                                sheet_info_label = gr.Markdown("📄 *Chưa tải lên sheet nhạc. Hỗ trợ ảnh JPG/PNG, tài liệu PDF hoặc tệp MusicXML.*")
                            sheet_convert_btn = gr.Button("🎹 Quét Sheet Nhạc & Tạo Bản Cover", variant="primary", elem_classes=["convert-btn", "sheet-btn"])

                    with gr.Group(elem_classes=["feature-highlight"]):
                        gr.Markdown("#### 🌟 1. Phong Cách, Tone & Nhạc Cụ Cover")
                        cover_mode = gr.Radio(
                            label="Phong Cách Chuyển Đổi",
                            choices=[
                                "Đệm Hát Tiêu Chuẩn (Chords Only)",
                                "Độc Tấu / Fingerstyle (Solo Melody + Chords)"
                            ],
                            value="Đệm Hát Tiêu Chuẩn (Chords Only)",
                            info="Chế độ 'Độc Tấu / Fingerstyle' bóc cả câu hát lẫn hợp âm đệm vào chung một đàn, cho phép nghe rõ giai điệu bài hát mà không cần lời."
                        )

                        # --- Auto Key Detection & Smart Transpose (AI Tone Engine) ---
                        with gr.Row():
                            detect_key_btn = gr.Button("🔍 Nhận Diện Tone & Nhịp BPM (AI)", variant="secondary", scale=1)
                            key_display = gr.Markdown("🎵 **Tone & Nhịp:** *Chưa nhận diện (Tải file hoặc dán link rồi nhấn nút)*", scale=2)
                        detected_key_state = gr.State(value="")

                        target_key_dropdown = gr.Dropdown(
                            label="🎯 Chọn Tone Đích Mong Muốn (Tự Động Tính Quãng Dịch Giọng)",
                            choices=TARGET_KEY_CHOICES,
                            value="Giữ nguyên gốc (Không đổi)",
                            info="Hệ thống tự động tính khoảng cách số nửa cung ngắn nhất [-6..+6] từ Tone gốc và tự động đồng bộ vào thanh trượt bên dưới."
                        )

                        transpose_slider = gr.Slider(
                            minimum=-6, maximum=6, value=0, step=1,
                            label="Đổi Tone (Transpose - Nửa Cung)",
                            info="-2 là hạ 1 cung, +1 là nâng nửa cung. Tự động đồng bộ từ Tone đích, người dùng vẫn có thể kéo chỉnh thủ công."
                        )

                        tempo_multiplier_slider = gr.Slider(
                            minimum=0.6, maximum=1.5, value=1.0, step=0.05,
                            label="⏱️ Điều Chỉnh Tốc Độ / Tempo Multiplier (Time-Stretching)",
                            info="0.6x (chậm) đến 1.5x (nhanh). Co giãn nốt MIDI và kéo giãn Vocal bằng FFmpeg atempo bảo toàn cao độ tuyệt đối (Preserves Pitch)."
                        )
                        instrument_selector = gr.CheckboxGroup(
                            label="🎼 Chọn Các Nhạc Cụ Muốn Xuất Bản Cover",
                            choices=[INSTRUMENT_CONFIGS[k]["name"] for k in ALL_INSTRUMENTS],
                            value=[],
                            info="Tích chọn 1 hoặc nhiều nhạc cụ tùy thích. Hệ thống sẽ bóc tách và render riêng cho từng nhạc cụ."
                        )
                        soundfont_dropdown = gr.Dropdown(
                            label="Bộ SoundFont (Màu Sắc Âm Thanh)",
                            choices=[
                                "Studio Hi-Fi SoundFonts (Steinway Grand & Studio Guitars - Khuyên dùng)",
                                "Standard General MIDI (GeneralUser GS - Tốc độ nhanh)"
                            ],
                            value="Studio Hi-Fi SoundFonts (Steinway Grand & Studio Guitars - Khuyên dùng)",
                            info="Studio Hi-Fi sử dụng bộ sample Nice Steinway Lite Concert Grand (74MB) và Guitars Universal (150MB) có độ ngân và dải âm sống động."
                        )
                        acoustic_preset_dropdown = gr.Dropdown(
                            label="🏛️ Không Gian Biểu Diễn (Acoustic Presets)",
                            choices=[
                                "Phòng Thu Acoustic (Studio Warmth) - Mặc định",
                                "Nhà Hát / Khán Phòng Lớn (Concert Hall)",
                                "Quán Cà Phê / Lo-Fi Chill"
                            ],
                            value="Phòng Thu Acoustic (Studio Warmth) - Mặc định",
                            info="Tự động áp dụng Reverb không gian, Stereo Enhancer, EQ cắt tần số bùn (275Hz) và tăng sáng (11kHz)."
                        )

                    with gr.Group(elem_classes=["feature-highlight"]):
                        gr.Markdown("#### ✂️ 2. Nghe Thử Đoạn Ngắn (Audio Range Trimmer / Preview Mode)")
                        preview_mode = gr.Checkbox(
                            label="⚡ Bật chế độ nghe thử đoạn ngắn (Preview Mode - Tốc độ cực nhanh)",
                            value=False,
                            info="Chỉ bóc tách và chuyển đổi một đoạn ngắn (10s - 60s) để nghe thử âm sắc các nhạc cụ ngay lập tức, tiết kiệm 80% thời gian chờ."
                        )
                        with gr.Row(visible=False) as preview_row:
                            preview_start = gr.Slider(
                                minimum=0, maximum=600, value=30, step=1,
                                label="Thời điểm bắt đầu (Start Time - Giây)",
                                info="Ví dụ: 30s hoặc 60s (vào thẳng đoạn điệp khúc)."
                            )
                            preview_duration = gr.Slider(
                                minimum=10, maximum=60, value=30, step=5,
                                label="Độ dài đoạn cắt (Duration - Giây)",
                                info="Mặc định 30 giây (tối đa 60s để tối ưu tốc độ)."
                            )

                    with gr.Accordion("⚙️ 3. Cấu Hình Nâng Cao (Cách Ly Trống, Lọc MIDI, Reverb & Tỉ Lệ Mix)", open=False):
                        with gr.Row():
                            include_bass = gr.Checkbox(
                                label="Kết hợp dải cao của Bass (Lấy nốt gốc hợp âm, không lẫn trống)",
                                value=True,
                                info="Nếu bỏ chọn, chỉ sử dụng track 'Other' thuần túy."
                            )
                            apply_enhancer = gr.Checkbox(
                                label="Kích hoạt âm vang FFmpeg Reverb (aecho) cho Master MP3",
                                value=True,
                                info="Bổ sung độ ngân không gian 35ms giúp tiếng đàn ấm và sâu hơn."
                            )
                        with gr.Row():
                            min_midi_dur = gr.Slider(
                                minimum=30.0, maximum=150.0, value=60.0, step=5.0,
                                label="Lọc bỏ nốt MIDI quá ngắn (< ms)",
                                info="Mặc định 60ms để loại bỏ hoàn toàn các nốt rè, transient click tạp âm."
                            )
                            min_pitch = gr.Slider(
                                minimum=24, maximum=48, value=36, step=1,
                                label="Lọc bỏ nốt quá trầm (< C2 / 36)",
                                info="Mặc định 36 (C2) để tránh nốt ồn sub-bass không phù hợp với nhạc cụ mộc."
                            )
                        with gr.Row():
                            reverb_room = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.68, step=0.02,
                                label="Độ vang phòng (Reverb Room Size)",
                                info="Tạo không gian vang hòa tấu tự nhiên cho các nhạc cụ, tránh âm thanh khô."
                            )
                            reverb_lvl = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.72, step=0.02,
                                label="Mức âm lượng Reverb (Wet Level)",
                                info="Tỉ lệ tín hiệu vang dội trong bản phối."
                            )
                        with gr.Row():
                            onset_thresh = gr.Slider(
                                minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                                label="Basic Pitch Onset Threshold",
                                info="Ngưỡng nhạy bắt đầu nốt."
                            )
                            frame_thresh = gr.Slider(
                                minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                                label="Basic Pitch Frame Threshold",
                                info="Ngưỡng duy trì trường độ nốt."
                            )
                        with gr.Row():
                            min_length = gr.Slider(
                                minimum=50.0, maximum=300.0, value=110.0, step=10.0,
                                label="Độ dài nốt nhận diện tối thiểu (ms)",
                                info="Bộ lọc sơ cấp trong quá trình inference của Basic Pitch."
                            )
                        with gr.Row():
                            vocal_vol = gr.Slider(
                                minimum=0.2, maximum=2.0, value=0.85, step=0.05,
                                label="Tỉ lệ Vocal gốc (85% / 0.85x)",
                                info="Tự động lọc sub-bass 85Hz + de-essing và kết hợp Sidechain Compressor ducking nhường chỗ cho giọng hát."
                            )
                            inst_vol = gr.Slider(
                                minimum=0.2, maximum=2.0, value=1.25, step=0.05,
                                label="Tỉ lệ Nhạc Cụ đệm (125% / 1.25x)",
                                info="Tiếng đàn giữ mức 1.25x kết hợp EQ Carving 2kHz và Sidechain Ducking tự động nở to khi ngắt câu hát."
                            )

                    convert_btn = gr.Button("🚀 BẮT ĐẦU CHUYỂN ĐỔI", variant="primary", elem_classes=["convert-btn"])
                    status_text = gr.Markdown("💡 *Sẵn sàng xử lý. Vui lòng tải file lên, chọn nhạc cụ và nhấn Bắt đầu.*")

                # Right column: Output Master Players (Dynamic Groups)
                with gr.Column(scale=7):
                    gr.Markdown("### 🎧 Kết Quả Master Cover MP3 320kbps (Nghe Thử & Tải Về)")
                    gr.HTML(render_master_toolbar("tab1"))
                    
                    empty_notice = gr.Markdown(
                        "### 🎧 Chưa chọn nhạc cụ nào\n*Vui lòng tích chọn ít nhất 1 nhạc cụ ở danh sách bên trái để xuất bản cover.*",
                        visible=True,
                        elem_classes=["empty-notice-box"]
                    )

                    inst_containers = {}
                    inst_audio_players = {}  # {k: (inst_player, voc_player)}
                    for k in ALL_INSTRUMENTS:
                        cfg = INSTRUMENT_CONFIGS[k]
                        with gr.Group(visible=False, elem_classes=["card-box", "inst-group"]) as grp:
                            title_disp = "🎋 Sáo Trúc Hòa Tấu" if k == "flute" else cfg["name"]
                            gr.HTML(f'<div class="result-card-title">{title_disp}</div>')
                            with gr.Row():
                                inst_p = gr.Audio(
                                    label="🎋 Sáo Trúc Hòa Tấu (Không Lời)" if k == "flute" else (f"{cfg['name']} Độc Tấu (Không Lời)" if cfg.get("is_monophonic") else f"{cfg['label']} Không Lời (Master Reverb)"),
                                    type="filepath",
                                    waveform_options=WAVEFORM_CONFIG
                                )
                                voc_p = gr.Audio(
                                    label="🎤🎋 Sáo Trúc Cover + Vocal Gốc" if k == "flute" else (f"🎤🎋 {cfg['name']} Cover + Vocal Gốc" if cfg.get("is_monophonic") else f"{cfg['label']} + Vocal Gốc"),
                                    type="filepath",
                                    waveform_options=WAVEFORM_CONFIG
                                )
                            gr.HTML(render_speed_controller(f"tab1_{k}"))
                            inst_containers[k] = grp
                            inst_audio_players[k] = (inst_p, voc_p)

                    tab1_audio_outputs = []
                    for k in ALL_INSTRUMENTS:
                        p_inst, p_voc = inst_audio_players[k]
                        tab1_audio_outputs.append(p_inst)
                        tab1_audio_outputs.append(p_voc)

                    with gr.Group(elem_classes=["card-box"]):
                        gr.HTML('<div class="result-card-title midi-title">🎼 File MIDI Đã Lọc Sạch Nốt Rác</div>')
                        gr.Markdown("File MIDI đã được lọc bỏ nốt <60ms và chuẩn hóa Program GM. Sẵn sàng import vào FL Studio, Ableton, Logic Pro:")
                        midi_file_output = gr.File(label="Tải file MIDI (.mid)")

                    current_tab1_entry_id = gr.State("")

                    with gr.Group(elem_classes=["card-box", "chord-hud-container"]):
                        gr.HTML('<div class="result-card-title chord-hud-title">🎸 Vòng Hợp Âm & Tiến Trình Nốt (Real-Time Audio Sync)</div>')
                        gr.Markdown("Hệ thống tự động gom cụm nốt từ MIDI để nhận diện hợp âm. Khi bạn bấm Play bất kỳ bài hát nào ở trên, hợp âm đang phát sẽ phát sáng rực rỡ và bám sát tiến độ bài hát!")
                        tab1_chord_hud = gr.HTML(render_chord_chips_html(None))
                        tab1_chord_file = gr.File(label="📄 Tải Bảng Hợp Âm Rút Gọn (.txt)")

                    with gr.Group(elem_classes=["card-box", "mixer-panel"]):
                        gr.HTML("""
                            <div class="result-card-title mixer-title">
                                🎛️ Trình Trộn Âm Tương Tác Trực Quan (Interactive Stems Mini-Mixer)
                            </div>
                            <p style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 12px;">
                                Tự do cân chỉnh âm lượng giữa <b>Giọng Hát (Vocal)</b>, <b>Nhạc Cụ Cover</b> và <b>Acoustic Upright Bass (GM 32 mộc ấm)</b>, bổ sung Master Reverb Send để tạo ra bản phối riêng biệt chuẩn EBU R128 (-14 LUFS).
                            </p>
                        """)
                        with gr.Row():
                            with gr.Column(scale=3):
                                tab1_mix_vocal_vol = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="🎤 Vocal Volume", info="Âm lượng giọng hát gốc")
                            with gr.Column(scale=3):
                                tab1_mix_inst_vol = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="🎹 Cover Instrument Volume", info="Âm lượng nhạc cụ cover")
                            with gr.Column(scale=3):
                                tab1_mix_bass_vol = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="🎻 Acoustic Bass Volume", info="Âm lượng Acoustic Upright Bass (GM 32)")
                            with gr.Column(scale=3):
                                tab1_mix_reverb_send = gr.Slider(0, 100, value=25, step=5, label="🏛️ Master Reverb Send (%)", info="Độ vang không gian phòng thu")

                        tab1_mix_btn = gr.Button("🎧 Mix & Xuất Bản Phối Này", variant="primary", size="lg")
                        tab1_mixer_status = gr.Markdown("💡 *Chuyển đổi bài hát ở trên để kích hoạt nguồn Stems cho Mini-Mixer.*")
                        tab1_mixed_audio = gr.Audio(
                            label="🎧 Bản Phối Riêng Của Bạn (Master Re-Mix MP3 320kbps)",
                            type="filepath",
                            waveform_options=WAVEFORM_CONFIG
                        )

        # TAB 2: HISTORY
        with gr.TabItem("📂 2. Lịch Sử Chuyển Đổi"):
            with gr.Row():
                with gr.Column(scale=6):
                    initial_choices = get_history_dropdown_choices()
                    default_choice = initial_choices[0][1] if initial_choices else None
                    history_dropdown = gr.Dropdown(
                        label="Chọn bài hát đã chuyển đổi trong lịch sử",
                        choices=initial_choices,
                        value=default_choice,
                        interactive=True
                    )
                with gr.Column(scale=6):
                    gr.Markdown("#### ⚡ Thao Tác Lịch Sử")
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Làm Mới", variant="secondary")
                        clear_cache_btn = gr.Button("🧹 Xóa Cache Demucs", variant="secondary")
                        delete_btn = gr.Button("🗑️ Xóa Bài Này", variant="stop", elem_classes=["danger-btn"])
                        clear_all_btn = gr.Button("⚠️ Xóa Toàn Bộ Lịch Sử", variant="stop", elem_classes=["danger-all-btn"])

            history_info_box = gr.Markdown("💡 *Chọn một bài hát từ danh sách trên để nạp ngay kết quả.*")
            gr.HTML(render_master_toolbar("tab2"))

            h_empty_notice = gr.Markdown(
                "### 📂 Chưa chọn bài hát lịch sử\n*Vui lòng chọn một bài hát từ danh sách trên để xem kết quả.*",
                visible=(not bool(default_choice)),
                elem_classes=["empty-notice-box"]
            )

            hist_containers = {}
            hist_audio_players = {}
            for k in ALL_INSTRUMENTS:
                cfg = INSTRUMENT_CONFIGS[k]
                with gr.Group(visible=False, elem_classes=["card-box", "inst-group"]) as h_grp:
                    title_disp = "🎋 Sáo Trúc Hòa Tấu" if k == "flute" else cfg["name"]
                    gr.HTML(f'<div class="result-card-title">{title_disp}</div>')
                    with gr.Row():
                        h_inst = gr.Audio(
                            label="🎋 Sáo Trúc Hòa Tấu (Không Lời)" if k == "flute" else (f"{cfg['name']} Độc Tấu (Không Lời)" if cfg.get("is_monophonic") else f"{cfg['label']} Không Lời"),
                            type="filepath",
                            waveform_options=WAVEFORM_CONFIG
                        )
                        h_voc = gr.Audio(
                            label="🎤🎋 Sáo Trúc Cover + Vocal Gốc" if k == "flute" else (f"🎤🎋 {cfg['name']} With Vocals" if cfg.get("is_monophonic") else f"{cfg['label']} With Vocals"),
                            type="filepath",
                            waveform_options=WAVEFORM_CONFIG
                        )
                    gr.HTML(render_speed_controller(f"tab2_{k}"))
                    hist_containers[k] = h_grp
                    hist_audio_players[k] = (h_inst, h_voc)

            tab2_audio_outputs = []
            for k in ALL_INSTRUMENTS:
                h_inst, h_voc = hist_audio_players[k]
                tab2_audio_outputs.append(h_inst)
                tab2_audio_outputs.append(h_voc)

            with gr.Group(elem_classes=["card-box"]):
                gr.HTML('<div class="result-card-title midi-title">🎼 File MIDI Bóc Tách</div>')
                h_midi_file = gr.File(label="Tải file MIDI (.mid)")

            current_tab2_entry_id = gr.State("")

            with gr.Group(elem_classes=["card-box", "chord-hud-container"]):
                gr.HTML('<div class="result-card-title chord-hud-title">🎸 Vòng Hợp Âm & Tiến Trình Nốt (Real-Time Audio Sync)</div>')
                h_chord_hud = gr.HTML(render_chord_chips_html(None))
                h_chord_file = gr.File(label="📄 Tải Bảng Hợp Âm Rút Gọn (.txt)")

            with gr.Group(elem_classes=["card-box", "mixer-panel"]):
                gr.HTML("""
                    <div class="result-card-title mixer-title">
                        🎛️ Phối Lại Bản Thu Lịch Sử (Interactive Stems Mini-Mixer)
                    </div>
                    <p style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 12px;">
                        Tùy chỉnh lại tỷ lệ âm lượng Vocal, Nhạc Cụ Cover, Acoustic Upright Bass và Master Reverb cho bài hát đang chọn trong lịch sử.
                    </p>
                """)
                with gr.Row():
                    with gr.Column(scale=3):
                        tab2_mix_vocal_vol = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="🎤 Vocal Volume")
                    with gr.Column(scale=3):
                        tab2_mix_inst_vol = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="🎹 Cover Instrument Volume")
                    with gr.Column(scale=3):
                        tab2_mix_bass_vol = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="🎻 Acoustic Bass Volume")
                    with gr.Column(scale=3):
                        tab2_mix_reverb_send = gr.Slider(0, 100, value=25, step=5, label="🏛️ Master Reverb Send (%)")
                tab2_mix_btn = gr.Button("🎧 Mix & Phối Lại Bài Lịch Sử Này", variant="primary")
                tab2_mixer_status = gr.Markdown("")
                tab2_mixed_audio = gr.Audio(label="🎧 Bản Phối Lịch Sử Xuất Mới", type="filepath", waveform_options=WAVEFORM_CONFIG)

    # Dynamic sync: when user toggles checkboxes on the left, immediately show/hide right-side groups
    def on_instrument_change(selected_names):
        selected_keys = []
        for n in (selected_names or []):
            norm = normalize_instrument_key(n)
            if norm and norm in ALL_INSTRUMENTS and norm not in selected_keys:
                selected_keys.append(norm)
        has_selection = len(selected_keys) > 0
        notice_update = gr.Markdown(visible=not has_selection)
        group_updates = [gr.Group(visible=(k in selected_keys)) for k in ALL_INSTRUMENTS]
        return (notice_update, *group_updates)

    instrument_selector.change(
        fn=on_instrument_change,
        inputs=[instrument_selector],
        outputs=[empty_notice, *[inst_containers[k] for k in ALL_INSTRUMENTS]]
    )

    preview_mode.change(
        fn=lambda active: gr.Row(visible=active),
        inputs=[preview_mode],
        outputs=[preview_row]
    )

    # Sheet music file change event to update preview
    input_sheet.change(
        fn=handle_sheet_file_change,
        inputs=[input_sheet],
        outputs=[sheet_preview, sheet_info_label]
    )

    conversion_inputs = [
        input_audio,               # 1. file_path
        url_input,                 # 2. url_input
        input_sheet,               # 3. sheet_file
        cover_mode,                # 4. cover_mode
        transpose_slider,          # 5. transpose_val
        tempo_multiplier_slider,   # 6. tempo_multiplier
        instrument_selector,       # 7. selected_inst_names
        soundfont_dropdown,        # 8. soundfont_choice
        acoustic_preset_dropdown,  # 9. acoustic_preset
        preview_mode,              # 10. preview_mode
        preview_start,             # 11. preview_start
        preview_duration,          # 12. preview_duration
        onset_thresh,              # 13. onset_thresh
        frame_thresh,              # 14. frame_thresh
        min_length,                # 15. min_length
        vocal_vol,                 # 16. vocal_vol
        inst_vol,                  # 17. inst_vol
        include_bass,              # 18. include_bass
        min_midi_dur,              # 19. min_midi_dur
        min_pitch,                 # 20. min_pitch
        reverb_room,               # 21. reverb_room
        reverb_lvl,                # 22. reverb_lvl
        apply_enhancer             # 23. apply_enhancer
    ]

    conversion_outputs = [
        empty_notice,
        *[inst_containers[k] for k in ALL_INSTRUMENTS],
        *tab1_audio_outputs,
        midi_file_output,
        tab1_chord_hud,
        tab1_chord_file,
        status_text,
        history_dropdown,
        current_tab1_entry_id
    ]

    # Event binding for Tab 1 (Conversion)
    convert_btn.click(
        fn=convert_song_to_covers,
        inputs=conversion_inputs,
        outputs=conversion_outputs
    )

    sheet_convert_btn.click(
        fn=convert_song_to_covers,
        inputs=conversion_inputs,
        outputs=conversion_outputs
    )

    tab1_mix_btn.click(
        fn=mix_custom_stems,
        inputs=[
            current_tab1_entry_id,
            tab1_mix_vocal_vol,
            tab1_mix_inst_vol,
            tab1_mix_bass_vol,
            tab1_mix_reverb_send
        ],
        outputs=[
            tab1_mixed_audio,
            tab1_mixer_status
        ]
    )

    tab2_outputs = [
        h_empty_notice,
        *[hist_containers[k] for k in ALL_INSTRUMENTS],
        *tab2_audio_outputs,
        h_midi_file,
        h_chord_hud,
        h_chord_file,
        history_info_box,
        current_tab2_entry_id
    ]

    # Event binding for Tab 2 (History)
    history_dropdown.change(
        fn=load_history_song,
        inputs=[history_dropdown],
        outputs=tab2_outputs
    )

    tab2_mix_btn.click(
        fn=mix_custom_stems,
        inputs=[
            current_tab2_entry_id,
            tab2_mix_vocal_vol,
            tab2_mix_inst_vol,
            tab2_mix_bass_vol,
            tab2_mix_reverb_send
        ],
        outputs=[
            tab2_mixed_audio,
            tab2_mixer_status
        ]
    )

    refresh_btn.click(
        fn=refresh_history,
        inputs=[],
        outputs=[
            history_dropdown,
            *tab2_outputs
        ]
    )

    delete_btn.click(
        fn=delete_single_history,
        inputs=[history_dropdown],
        outputs=[
            history_dropdown,
            *tab2_outputs
        ]
    )

    clear_all_btn.click(
        fn=clear_all_history_entries,
        inputs=[],
        outputs=[
            history_dropdown,
            *tab2_outputs
        ]
    )

    clear_cache_btn.click(
        fn=handle_clear_stem_cache,
        inputs=[],
        outputs=[history_info_box]
    )

    detect_key_btn.click(
        fn=handle_auto_detect_key,
        inputs=[input_audio, url_input, input_sheet],
        outputs=[key_display, detected_key_state]
    )

    target_key_dropdown.change(
        fn=handle_target_key_change,
        inputs=[target_key_dropdown, detected_key_state],
        outputs=[transpose_slider]
    )

    # Automatically load the first history item on page mount if available
    if default_choice:
        demo.load(
            fn=load_history_song,
            inputs=[history_dropdown],
            outputs=tab2_outputs
        )


if __name__ == "__main__":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("\n" + "="*60)
    print(" 🎹🎸 SONG TO MULTI-INSTRUMENT COVER CONVERTER IS STARTING... ")
    print("="*60)
    
    server_name = os.environ.get("SERVER_NAME", "127.0.0.1")
    port = int(os.environ.get("PORT", "7860"))
    print(f" Server URL: http://{server_name}:{port}")
    print("="*60 + "\n")
    
    import time
    for attempt in range(10):
        try:
            demo.launch(
                server_name=server_name,
                server_port=port,
                show_error=True,
                theme=gr.themes.Soft(),
                css=CUSTOM_CSS,
                head=f"<script>{PLAYER_ENHANCEMENT_JS}</script>"
            )
            break
        except OSError:
            print(f"Port {port} busy or in TimeWait, retrying in 3s (attempt {attempt+1}/10)...")
            time.sleep(3)

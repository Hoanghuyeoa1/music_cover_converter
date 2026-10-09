import os
import sys
import json
import shutil
import tempfile
import logging
import subprocess
import re
import random
import hashlib
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Dict, List, Tuple, Any

import numpy as np
import librosa
import pretty_midi

from src.downloader import ensure_ffmpeg, ensure_fluidsynth, ensure_soundfonts, PROJECT_ROOT
from src.omr_processor import SheetMusicOMR, is_sheet_music_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AudioProcessor")

# Safe encoding configuration on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Prevent UnicodeEncodeError from Basic Pitch printing emojis on Windows console
try:
    import basic_pitch.commandline_printing as bp_print
    bp_print.file_saved_confirmation = lambda output_type, save_path: logger.info(f"Saved {output_type} to {save_path}")
    bp_print.failed_to_save = lambda output_type, save_path: logger.warning(f"Failed to save {output_type} to {save_path}")
    bp_print.generating_file_message = lambda output_type: logger.info(f"Generating {output_type}...")
except Exception as patch_e:
    logger.debug(f"Could not patch basic_pitch printing: {patch_e}")


def sanitize_filename(name: str) -> str:
    """Sanitize string to create safe folder/file names."""
    clean = re.sub(r'[\\/*?:"<>|]', "", name)
    clean = clean.strip().replace(" ", "_")
    return clean[:40] if clean else "Cover_Song"


INSTRUMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "piano": {
        "name": "🎹 Piano Acoustic (Grand Piano)",
        "label": "Piano",
        "program": 0,
        "min_pitch": 36,   # C2
        "max_pitch": 108,  # C8
        "gain_boost": 1.4,
        "reverb_level": 0.72,
        "soundfont_type": "piano",
    },
    "guitar_steel": {
        "name": "🎸 Acoustic Steel Guitar (Dây Sắt)",
        "label": "Guitar_Steel",
        "program": 25,
        "min_pitch": 40,   # E2
        "max_pitch": 96,   # C7
        "gain_boost": 1.4,
        "reverb_level": 0.72,
        "soundfont_type": "guitar",
    },
    "guitar_nylon": {
        "name": "🎸 Acoustic Nylon Guitar (Dây Nilon)",
        "label": "Guitar_Nylon",
        "program": 24,
        "min_pitch": 40,   # E2
        "max_pitch": 96,   # C7
        "gain_boost": 1.4,
        "reverb_level": 0.74,
        "soundfont_type": "guitar",
    },
    "strings": {
        "name": "🎻 Dàn Dây / Violin Ensemble (Ballad / Điện ảnh)",
        "label": "Strings",
        "program": 48,     # String Ensemble 1
        "min_pitch": 48,   # C3
        "max_pitch": 100,  # E7
        "gain_boost": 1.45,
        "reverb_level": 0.85,
        "soundfont_type": "general",
    },
    "saxophone": {
        "name": "🎷 Saxophone Alto (Jazz / R&B / Chill)",
        "label": "Saxophone",
        "program": 65,     # Alto Sax
        "min_pitch": 49,   # Db3
        "max_pitch": 90,   # F#6
        "gain_boost": 1.35,
        "reverb_level": 0.70,
        "soundfont_type": "general",
    },
    "epiano": {
        "name": "🎹 Electric Piano / Rhodes (Lo-Fi / Soul / Pop)",
        "label": "Electric_Piano",
        "program": 4,      # Electric Piano 1 (Rhodes)
        "min_pitch": 36,   # C2
        "max_pitch": 100,
        "gain_boost": 1.35,
        "reverb_level": 0.75,
        "soundfont_type": "general",
    },
    "harp": {
        "name": "🪕 Đàn Hạc / Harp (Trong trẻo / Cổ tích)",
        "label": "Harp",
        "program": 46,     # Orchestral Harp
        "min_pitch": 36,   # C2
        "max_pitch": 103,  # G7
        "gain_boost": 1.45,
        "reverb_level": 0.82,
        "soundfont_type": "general",
    },
    "musicbox": {
        "name": "🧸 Hộp Nhạc / Music Box (Dễ thương / Viral TikTok)",
        "label": "Music_Box",
        "program": 10,     # Music Box
        "min_pitch": 60,   # C4
        "max_pitch": 108,  # C8
        "gain_boost": 1.45,
        "reverb_level": 0.65,
        "soundfont_type": "general",
    },
    "flute": {
        "name": "🎋 Sáo Trúc Hòa Tấu (Bamboo Flute)",
        "label": "Bamboo_Flute",
        "program": 77,     # Shakuhachi / Tiêu - Sáo Trúc Đông Á
        "min_pitch": 55,   # G3 / A3 (Cho phép dải nốt trầm ấm của Sáo Trúc & Tiêu)
        "max_pitch": 84,   # C6 (Khống chế trần cao độ chống rít the thé)
        "gain_boost": 1.45,
        "reverb_level": 0.80,
        "reverb_room_size": 0.75,
        "reverb_damp": 0.25,
        "soundfont_type": "general",
        "is_monophonic": True,
        "has_backing": True,
    },
}

# Aliases for backwards compatibility
INSTRUMENT_CONFIGS["electric_piano"] = INSTRUMENT_CONFIGS["epiano"]
INSTRUMENT_CONFIGS["music_box"] = INSTRUMENT_CONFIGS["musicbox"]
INSTRUMENT_CONFIGS["bamboo_flute"] = INSTRUMENT_CONFIGS["flute"]


ACOUSTIC_PRESETS: Dict[str, Dict[str, Any]] = {
    "studio_warmth": {
        "name": "Phòng Thu Acoustic (Studio Warmth)",
        "label": "Studio_Warmth",
        "description": "Âm thanh dày dặn, mộc mạc, rõ từng phím gảy với độ ngân phòng ấm áp.",
        "fs_room_size": 0.62,
        "fs_damp": 0.35,
        "fs_level": 0.65,
        "fs_width": 0.90,
        "fs_chorus": False,
        "eq_filter": "equalizer=f=275:width_type=q:w=1.5:g=-2.5,equalizer=f=11000:width_type=q:w=1.2:g=2.2",
        "stereo_width": 1.22,
        "echo_filter": "aecho=0.8:0.85:32:0.20",
        "vocal_blend_reverb": "aecho=0.8:0.8:28:0.18"
    },
    "concert_hall": {
        "name": "Nhà Hát / Khán Phòng Lớn (Concert Hall)",
        "label": "Concert_Hall",
        "description": "Không gian rộng lớn, độ ngân vang hoành tráng như đang ngồi giữa khán phòng thính phòng.",
        "fs_room_size": 0.85,
        "fs_damp": 0.20,
        "fs_level": 0.84,
        "fs_width": 1.0,
        "fs_chorus": True,
        "eq_filter": "equalizer=f=275:width_type=q:w=1.2:g=-3.0,equalizer=f=12000:width_type=q:w=1.0:g=3.2",
        "stereo_width": 1.35,
        "echo_filter": "aecho=0.8:0.88:65:0.35",
        "vocal_blend_reverb": "aecho=0.8:0.85:50:0.28"
    },
    "lofi_chill": {
        "name": "Quán Cà Phê / Lo-Fi Chill",
        "label": "Lo-Fi_Chill",
        "description": "Tông trầm ấm cổ điển, treble êm dịu thư giãn, cảm giác gần gũi như một buổi acoustic coffee shop.",
        "fs_room_size": 0.54,
        "fs_damp": 0.55,
        "fs_level": 0.55,
        "fs_width": 0.80,
        "fs_chorus": False,
        "eq_filter": "bass=g=3.2:f=120,equalizer=f=3500:width_type=q:w=1.0:g=1.2,treble=g=-3.5:f=8000",
        "stereo_width": 1.10,
        "echo_filter": "aecho=0.8:0.78:38:0.18",
        "vocal_blend_reverb": "aecho=0.8:0.75:32:0.16"
    }
}


def compute_file_sha256(filepath: Path, chunk_size: int = 65536) -> str:
    """Calculate SHA-256 hash of a file for stem caching."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles (standard tonal hierarchies for 12 pitch classes)
KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# 12 Tonal Center Pairs (Relative Major & Minor Pairs)
KEY_CENTER_PAIRS: List[Dict[str, Any]] = [
    {"label": "C Major / A Minor", "major_root": 0, "minor_root": 9, "display": "C / Am"},
    {"label": "C# Major / A# Minor (Db / Bbm)", "major_root": 1, "minor_root": 10, "display": "C# / A#m"},
    {"label": "D Major / B Minor", "major_root": 2, "minor_root": 11, "display": "D / Bm"},
    {"label": "D# Major / C Minor (Eb / Cm)", "major_root": 3, "minor_root": 0, "display": "D# / Cm"},
    {"label": "E Major / C# Minor", "major_root": 4, "minor_root": 1, "display": "E / C#m"},
    {"label": "F Major / D Minor", "major_root": 5, "minor_root": 2, "display": "F / Dm"},
    {"label": "F# Major / D# Minor (Gb / Ebm)", "major_root": 6, "minor_root": 3, "display": "F# / D#m"},
    {"label": "G Major / E Minor", "major_root": 7, "minor_root": 4, "display": "G / Em"},
    {"label": "G# Major / F Minor (Ab / Fm)", "major_root": 8, "minor_root": 5, "display": "G# / Fm"},
    {"label": "A Major / F# Minor", "major_root": 9, "minor_root": 6, "display": "A / F#m"},
    {"label": "A# Major / G Minor (Bb / Gm)", "major_root": 10, "minor_root": 7, "display": "A# / Gm"},
    {"label": "B Major / G# Minor", "major_root": 11, "minor_root": 8, "display": "B / G#m"},
]


def detect_musical_key(audio_path: Path, max_duration_sec: float = 60.0) -> Dict[str, Any]:
    """
    Detect musical key of an audio file using the Krumhansl-Schmuckler Key-Finding Algorithm:
    - Extracts 12-dimensional chroma feature matrix via librosa.feature.chroma_cqt.
    - Computes Pearson correlation coefficient against 24 standard major and minor profiles.
    - Returns best matching key, scale mode, confidence score, and short label.
    """
    audio_file = Path(audio_path)
    if not audio_file.exists():
        return {
            "key": "C Major",
            "root": "C",
            "scale": "major",
            "confidence": 0.50,
            "label": "C Major (C)",
            "correlation": 0.50
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(str(audio_file), sr=22050, mono=True, duration=max_duration_sec)

        if len(y) < 1024:
            return {
                "key": "C Major",
                "root": "C",
                "scale": "major",
                "confidence": 0.50,
                "label": "C Major (C)",
                "correlation": 0.50
            }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Extract constant-Q chromagram (12 semitones)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        if chroma.size == 0 or chroma.shape[0] != 12:
            return {
                "key": "C Major",
                "root": "C",
                "scale": "major",
                "confidence": 0.50,
                "label": "C Major (C)",
                "correlation": 0.50
            }

        chroma_mean = np.mean(chroma, axis=1)
        # Avoid division by zero in correlation if chroma is silent
        if np.std(chroma_mean) < 1e-6:
            return {
                "key": "C Major",
                "root": "C",
                "scale": "major",
                "confidence": 0.50,
                "label": "C Major (C)",
                "correlation": 0.50
            }

        scores = []
        for i in range(12):
            maj_prof = np.roll(KRUMHANSL_MAJOR, i)
            min_prof = np.roll(KRUMHANSL_MINOR, i)
            r_maj = np.corrcoef(chroma_mean, maj_prof)[0, 1]
            r_min = np.corrcoef(chroma_mean, min_prof)[0, 1]
            scores.append((r_maj, f"{PITCH_NAMES[i]} Major", PITCH_NAMES[i], "major"))
            scores.append((r_min, f"{PITCH_NAMES[i]} Minor", PITCH_NAMES[i], "minor"))

        scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_name, root, mode = scores[0]
        second_score = scores[1][0] if len(scores) > 1 else 0.0

        # Calculate confidence based on difference between 1st and 2nd candidate
        raw_conf = (best_score - second_score) / (1.0 - second_score + 1e-6) * 1.5
        confidence = float(np.clip(raw_conf, 0.45, 0.98))

        disp_suffix = "m" if mode == "minor" else ""
        short_label = f"{best_name} ({root}{disp_suffix})"

        logger.info(f"Auto Key Detection [{audio_file.name}]: {best_name} (Confidence: {int(confidence*100)}%, Correlation: {best_score:.3f})")

        return {
            "key": best_name,
            "root": root,
            "scale": mode,
            "confidence": round(confidence, 2),
            "label": short_label,
            "correlation": round(float(best_score), 3)
        }
    except Exception as e:
        logger.warning(f"Auto Key Detection warning on {audio_file.name}: {e}. Falling back to C Major.")
        return {
            "key": "C Major",
            "root": "C",
            "scale": "major",
            "confidence": 0.50,
            "label": "C Major (C)",
            "correlation": 0.50
        }


def calculate_smart_transpose(detected_key_name: str, target_center_label: str) -> int:
    """
    Calculate optimal semitone transposition delta [-6..+6] from detected key to target tonal center:
    - Normalizes both source and target keys to their relative major tonal center.
    - Computes shortest circular chromatic distance mod 12 wrapped into [-6, +6].
    """
    if not detected_key_name or not target_center_label or "giữ nguyên" in target_center_label.lower():
        return 0

    parts = detected_key_name.strip().split()
    if not parts:
        return 0
    root_str = parts[0]
    is_minor = "minor" in detected_key_name.lower() or "m" in root_str[1:]

    # Clean root name
    clean_root = root_str.replace("m", "").replace("M", "")
    if clean_root not in PITCH_NAMES:
        return 0
    src_idx = PITCH_NAMES.index(clean_root)
    # Convert relative minor to major tonal center (e.g. A minor -> C major: (9 + 3) % 12 = 0)
    src_major_root = (src_idx + 3) % 12 if is_minor else src_idx

    tgt_pair = next((p for p in KEY_CENTER_PAIRS if p["label"] == target_center_label), None)
    if not tgt_pair:
        return 0
    tgt_major_root = tgt_pair["major_root"]

    delta = (tgt_major_root - src_major_root) % 12
    if delta > 6:
        delta -= 12

    return int(delta)


def extract_beats_and_tempo(audio_path: Path, max_duration_sec: Optional[float] = None) -> Tuple[float, np.ndarray]:
    """
    Extract tempo (BPM) and beat timestamps using librosa.beat.beat_track:
    - Provides beat timeline for groove and dynamic accentuation.
    - If fewer than 4 beats are detected (e.g. ambient/synth/isolated chords), synthesizes a reliable tempo-grid.
    """
    audio_file = Path(audio_path)
    default_tempo = 120.0

    if not audio_file.exists():
        interval = 60.0 / default_tempo
        return default_tempo, np.arange(0.0, 30.0, interval)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(str(audio_file), sr=22050, mono=True, duration=max_duration_sec)

        duration = len(y) / sr
        if duration < 0.2:
            interval = 60.0 / default_tempo
            return default_tempo, np.arange(0.0, max(1.0, duration), interval)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tempo_raw, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        bpm_val = float(tempo_raw[0] if isinstance(tempo_raw, (np.ndarray, list)) else tempo_raw)
        final_bpm = bpm_val if (45.0 <= bpm_val <= 230.0) else default_tempo

        # Fallback to tempo-grid if beat tracking detected insufficient beats
        if len(beat_times) < 4:
            interval = 60.0 / final_bpm
            beat_times = np.arange(0.0, duration + interval, interval)
            logger.debug(f"Beat tracking fallback to tempo-grid: {len(beat_times)} beats ({final_bpm:.1f} BPM)")
        else:
            logger.info(f"Beat tracking successful: {len(beat_times)} beats detected ({final_bpm:.1f} BPM)")

        return final_bpm, beat_times
    except Exception as e:
        logger.warning(f"Beat tracking warning on {audio_file.name}: {e}. Using 120 BPM grid.")
        interval = 60.0 / default_tempo
        return default_tempo, np.arange(0.0, 30.0, interval)


def apply_groove_and_beat_dynamics(
    notes: List[pretty_midi.Note],
    beat_times: Optional[np.ndarray] = None,
    tempo: float = 120.0,
    downbeat_tolerance_sec: float = 0.050,
    enable_micro_timing: bool = True
) -> List[pretty_midi.Note]:
    """
    Apply beat-aware groove dynamic accentuation & micro-timing:
    - Downbeat (phách 1 / phách mạnh nhất trong ô nhịp 4/4): +8 đến +14 velocity boost
    - Phách 3 (phách mạnh vừa): +4 đến +8 velocity boost
    - Phách 2 & 4 (phách nhẹ): -4 đến 0
    - Offbeats / Upbeats / Passing notes (nằm giữa 2 phách): -8 đến -4 (tạo độ nảy swing/groove)
    - Micro-timing: độ xê dịch thời gian ngẫu nhiên +/- 5ms chống cơ học robotic.
    """
    if not notes:
        return []

    if beat_times is None or len(beat_times) < 4:
        valid_tempo = tempo if 40.0 <= tempo <= 240.0 else 120.0
        interval = 60.0 / valid_tempo
        max_time = max(n.end for n in notes) if notes else 60.0
        beat_times = np.arange(0.0, max_time + interval, interval)

    for note in notes:
        t = note.start
        # Find nearest beat in beat_times
        idx = int(np.argmin(np.abs(beat_times - t)))
        nearest_beat_time = beat_times[idx]
        diff = abs(t - nearest_beat_time)

        if diff <= downbeat_tolerance_sec:
            # Note is on the beat
            beat_in_bar = idx % 4
            if beat_in_bar == 0:
                # Downbeat / Phách 1: mạnh nhất
                boost = random.randint(8, 14)
            elif beat_in_bar == 2:
                # Phách 3: mạnh vừa
                boost = random.randint(4, 8)
            else:
                # Phách 2 & 4: nhẹ
                boost = random.randint(-4, 0)
        else:
            # Offbeat / Passing note
            boost = random.randint(-8, -4)

        note.velocity = max(35, min(127, note.velocity + boost))

        if enable_micro_timing:
            jitter = random.uniform(-0.005, 0.005)
            new_start = max(0.0, note.start + jitter)
            new_end = max(new_start + 0.04, note.end + jitter)
            note.start = round(new_start, 4)
            note.end = round(new_end, 4)

    return notes


def fix_overlapping_notes(
    notes: List[pretty_midi.Note],
    min_duration_sec: float = 0.04
) -> List[pretty_midi.Note]:
    """
    Trim tails of overlapping notes with the same pitch to prevent harsh double-triggering,
    unwanted resonance, or phase cancellation in FluidSynth.
    """
    by_pitch: Dict[int, List[pretty_midi.Note]] = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)

    cleaned: List[pretty_midi.Note] = []
    for pitch, p_notes in by_pitch.items():
        p_notes.sort(key=lambda n: n.start)
        for i in range(len(p_notes)):
            curr = p_notes[i]
            if i + 1 < len(p_notes):
                nxt = p_notes[i + 1]
                if curr.end > nxt.start:
                    curr.end = max(curr.start + min_duration_sec, nxt.start - 0.015)
            if curr.end - curr.start >= min_duration_sec:
                cleaned.append(curr)

    cleaned.sort(key=lambda n: n.start)
    return cleaned


def apply_natural_dynamics(
    notes: List[pretty_midi.Note],
    is_lead_melody: bool = False,
    humanize_variation: int = 2
) -> List[pretty_midi.Note]:
    """
    Apply natural velocity dynamic curves & humanization:
    - Bass notes (pitch < 52): moderate velocity ~75-85
    - Mid accompaniment (52 <= pitch < 67): ~82-95
    - High treble / melody notes (pitch >= 67): ~95-115
    - Lead vocal melody (Solo mode): ~102-120
    """
    for note in notes:
        p = note.pitch
        if is_lead_melody:
            base = 104 + int(10 * min(1.0, max(0.0, (p - 60) / 24)))
            min_v, max_v = 98, 122
        elif p < 52:
            base = 76 + int(8 * min(1.0, max(0.0, (p - 36) / 16)))
            min_v, max_v = 72, 86
        elif p < 67:
            base = 85 + int(8 * min(1.0, max(0.0, (p - 52) / 15)))
            min_v, max_v = 80, 96
        else:
            base = 96 + int(16 * min(1.0, max(0.0, (p - 67) / 20)))
            min_v, max_v = 92, 116

        rel_factor = (note.velocity - 64) / 64.0
        adjusted = base + int(rel_factor * 6)
        if humanize_variation > 0:
            adjusted += random.randint(-humanize_variation, humanize_variation)
        note.velocity = max(min_v, min(max_v, adjusted))

    return notes


def humanize_guitar_strumming(
    notes: List[pretty_midi.Note],
    is_guitar: bool = False,
    chord_tolerance_sec: float = 0.030,   # <= 30ms tolerance
    min_stagger_sec: float = 0.015,       # 15ms
    max_stagger_sec: float = 0.025,       # 25ms
    velocity_variation_pct: float = 0.05  # +/- 5%
) -> List[pretty_midi.Note]:
    """
    Simulate authentic finger picking and plectrum strumming across guitar strings:
    - Identifies chords/clusters starting within chord_tolerance_sec (<= 30ms).
    - If is_guitar is True:
      * Sorts notes within each cluster by pitch ascending (low strings to high strings).
      * Staggers start time of successive notes by 15ms - 25ms to mimic plectrum/finger sweeping.
      * Introduces subtle +/- 5% velocity variation per note to eliminate robotic feel.
    - If is_guitar is False (e.g. Piano): keeps note timings intact for solid, crisp key action.
    """
    if not is_guitar or not notes:
        return notes

    sorted_notes = sorted(notes, key=lambda n: n.start)
    clusters: List[List[pretty_midi.Note]] = []
    current_cluster: List[pretty_midi.Note] = []
    cluster_anchor_time = -1.0

    for note in sorted_notes:
        if not current_cluster:
            current_cluster.append(note)
            cluster_anchor_time = note.start
        else:
            if abs(note.start - cluster_anchor_time) <= chord_tolerance_sec:
                current_cluster.append(note)
            else:
                clusters.append(current_cluster)
                current_cluster = [note]
                cluster_anchor_time = note.start

    if current_cluster:
        clusters.append(current_cluster)

    humanized_notes: List[pretty_midi.Note] = []
    for cluster in clusters:
        if len(cluster) >= 2:
            # Sort cluster from lowest pitch to highest pitch (low E -> high e string)
            cluster.sort(key=lambda n: n.pitch)
            accumulated_delay = 0.0
            for i, n in enumerate(cluster):
                if i > 0:
                    accumulated_delay += random.uniform(min_stagger_sec, max_stagger_sec)

                new_start = n.start + accumulated_delay
                new_end = max(new_start + 0.04, n.end + accumulated_delay)

                vel_factor = random.uniform(1.0 - velocity_variation_pct, 1.0 + velocity_variation_pct)
                new_vel = int(round(n.velocity * vel_factor))
                new_vel = max(1, min(127, new_vel))

                humanized_notes.append(
                    pretty_midi.Note(
                        velocity=new_vel,
                        pitch=n.pitch,
                        start=round(new_start, 4),
                        end=round(new_end, 4)
                    )
                )
        else:
            n = cluster[0]
            vel_factor = random.uniform(1.0 - (velocity_variation_pct * 0.5), 1.0 + (velocity_variation_pct * 0.5))
            new_vel = int(round(n.velocity * vel_factor))
            new_vel = max(1, min(127, new_vel))
            humanized_notes.append(
                pretty_midi.Note(
                    velocity=new_vel,
                    pitch=n.pitch,
                    start=n.start,
                    end=n.end
                )
            )

    humanized_notes.sort(key=lambda n: n.start)
    return humanized_notes


def extract_monophonic_lead_notes(
    notes: List[pretty_midi.Note],
    min_pitch: int = 58,
    max_pitch: int = 84,
    min_duration_sec: float = 0.100,
    chord_tolerance_sec: float = 0.035,
    max_gap_for_legato: float = 0.120,
    overlap_legato_sec: float = 0.025
) -> List[pretty_midi.Note]:
    """
    Extract a silky-smooth, strictly monophonic lead melody tailored for Bamboo Flute (Sáo Trúc):
    1. Anti-Piercing Pitch Clamp:
       - Transpose notes down 1 Octave (-12 semitones) if pitch > 84 (C6) to eliminate harsh screeching.
       - Discard notes below min_pitch (58 / A#3).
    2. De-Staccato & Noise Filter:
       - Discard short choppy notes < 100ms (breaths, artifacts, fragmented AI vibrato).
    3. Top-Note Filter:
       - Clusters notes starting within chord_tolerance_sec (<= 35ms) and preserves ONLY the note with maximum pitch.
    4. Timeline Reconciliation:
       - Resolves overlaps so higher pitch takes lead without conflicting double triggers.
    5. Legato Smoothing & Anti-Staccato (Gap Filling):
       - If gap between consecutive notes is < 120ms, extends the note to overlap by 25ms (Overlapping Legato),
         simulating the unbroken breath of a master flute performer.
       - If notes already overlap, smooths overlap to ~25ms.
    6. Mellow Wind Velocity:
       - Scales velocities gently into the warm, mellow 68 - 80 range (soft, breathy, non-fatiguing tone).
    """
    if not notes:
        return []

    # Step 1: Pitch clamping (anti-piercing) & de-staccato filter
    clamped_notes = []
    for n in notes:
        p = n.pitch
        while p > max_pitch:
            p -= 12
        while p < min_pitch:
            p += 12
        if min_pitch <= p <= max_pitch and (n.end - n.start) >= min_duration_sec:
            clamped_notes.append(
                pretty_midi.Note(
                    velocity=n.velocity,
                    pitch=p,
                    start=n.start,
                    end=n.end
                )
            )

    if not clamped_notes:
        return []

    # Step 2: Top-Note Filter for chords starting at nearly the same time (<= 35ms)
    sorted_notes = sorted(clamped_notes, key=lambda n: n.start)
    clusters: List[List[pretty_midi.Note]] = []
    current_cluster: List[pretty_midi.Note] = []
    cluster_anchor = -1.0

    for n in sorted_notes:
        if not current_cluster:
            current_cluster.append(n)
            cluster_anchor = n.start
        else:
            if abs(n.start - cluster_anchor) <= chord_tolerance_sec:
                current_cluster.append(n)
            else:
                clusters.append(current_cluster)
                current_cluster = [n]
                cluster_anchor = n.start
    if current_cluster:
        clusters.append(current_cluster)

    top_candidates = []
    for c in clusters:
        top_note = max(c, key=lambda n: (n.pitch, n.velocity))
        top_candidates.append(
            pretty_midi.Note(
                velocity=top_note.velocity,
                pitch=top_note.pitch,
                start=top_note.start,
                end=top_note.end
            )
        )

    # Step 3: Reconcile any remaining overlaps across the timeline to enforce strict monophony
    top_candidates.sort(key=lambda n: n.start)
    monophonic: List[pretty_midi.Note] = []

    for note in top_candidates:
        if not monophonic:
            monophonic.append(pretty_midi.Note(note.velocity, note.pitch, note.start, note.end))
            continue

        prev = monophonic[-1]
        if note.start < prev.end:
            if note.pitch >= prev.pitch:
                prev.end = max(prev.start + min_duration_sec, note.start)
                if prev.end <= prev.start + 0.02:
                    monophonic.pop()
                monophonic.append(pretty_midi.Note(note.velocity, note.pitch, note.start, note.end))
            else:
                if note.end > prev.end + min_duration_sec:
                    monophonic.append(pretty_midi.Note(note.velocity, note.pitch, prev.end, note.end))
        else:
            monophonic.append(pretty_midi.Note(note.velocity, note.pitch, note.start, note.end))

    # Step 4: Gap-Filling & Overlapping Legato + Mellow Velocity (68-80)
    for i in range(len(monophonic)):
        curr = monophonic[i]
        if i + 1 < len(monophonic):
            nxt = monophonic[i + 1]
            gap = nxt.start - curr.end
            if 0.0 <= gap < max_gap_for_legato:
                # Gap < 120ms: extend previous note to overlap next note by 25ms (continuous breath)
                curr.end = round(nxt.start + overlap_legato_sec, 4)
            elif gap < 0.0:
                # Already overlapping: cap overlap to ~25ms
                curr.end = round(max(curr.start + min_duration_sec, min(curr.end, nxt.start + overlap_legato_sec)), 4)
            else:
                # Musical rest > 120ms: subtle breath taper
                curr.end = round(min(curr.end + 0.035, nxt.start - 0.010), 4)
        else:
            # Final note of phrase/song
            curr.end = round(curr.end + 0.050, 4)

        # Scale velocity smoothly into vibrant, expressive 88 - 105 lead range
        rel = (curr.pitch - min_pitch) / max(1, max_pitch - min_pitch)
        base_vel = 92 + int(rel * 10)
        curr.velocity = max(86, min(108, base_vel + random.randint(-2, 3)))

    return monophonic


def generate_smart_sustain_events(
    notes: List[pretty_midi.Note],
    tempo: float = 120.0
) -> List[pretty_midi.ControlChange]:
    """
    Analyze note groupings and harmonic boundaries to generate authentic Sustain Pedal (CC 64) events:
    - Automatically pedal down (CC 64 = 127) at phrase/chord boundaries.
    - Release pedal (CC 64 = 0) ~40ms before chord changes to prevent dissonant mud.
    - Release pedal upon musical rests (> 0.45s of silence).
    """
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    events: List[pretty_midi.ControlChange] = []
    pedal_segments: List[Tuple[float, float]] = []

    current_segment_start = sorted_notes[0].start
    last_bass_pitch = next((n.pitch for n in sorted_notes if n.pitch < 55), None)
    last_note_end = sorted_notes[0].end

    for n in sorted_notes[1:]:
        t = n.start
        time_since_seg_start = t - current_segment_start
        is_silence_gap = (t - last_note_end) > 0.45
        is_bass_change = False
        if n.pitch < 55:
            if last_bass_pitch is None or n.pitch != last_bass_pitch:
                is_bass_change = True
                last_bass_pitch = n.pitch

        if is_silence_gap:
            rel_time = min(t - 0.05, last_note_end + 0.1)
            pedal_segments.append((current_segment_start, rel_time))
            current_segment_start = t
        elif (is_bass_change and time_since_seg_start >= 0.5) or (time_since_seg_start >= 1.25):
            rel_time = max(current_segment_start + 0.1, t - 0.04)
            pedal_segments.append((current_segment_start, rel_time))
            current_segment_start = t

        if n.end > last_note_end:
            last_note_end = n.end

    pedal_segments.append((current_segment_start, last_note_end))

    for press_t, release_t in pedal_segments:
        if release_t > press_t:
            events.append(pretty_midi.ControlChange(number=64, value=127, time=round(max(0.0, press_t), 3)))
            events.append(pretty_midi.ControlChange(number=64, value=0, time=round(max(0.0, release_t), 3)))

    events.sort(key=lambda cc: (cc.time, cc.value))
    return events


def estimate_bpm(audio_path: Path, max_duration_sec: Optional[float] = None) -> float:
    """Convenience function returning estimated song tempo (BPM)."""
    tempo, _ = extract_beats_and_tempo(audio_path, max_duration_sec=max_duration_sec)
    return round(float(tempo), 1)


def time_stretch_audio(
    input_wav: Path,
    output_wav: Path,
    tempo_multiplier: float = 1.0,
    ffmpeg_bin: Optional[str] = None
) -> Path:
    """
    Time-stretch audio file while strictly preserving pitch using FFmpeg's high-quality atempo filter.
    Chains multiple atempo filters if tempo_multiplier is outside the single-filter range [0.5, 2.0].
    """
    input_wav = Path(input_wav)
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    if not ffmpeg_bin:
        ffmpeg_bin = ensure_ffmpeg()

    if abs(tempo_multiplier - 1.0) < 0.005:
        if input_wav.resolve() != output_wav.resolve():
            shutil.copy2(input_wav, output_wav)
        return output_wav

    # Chain atempo filters (each atempo filter accepts 0.5 to 2.0)
    filters = []
    rem = float(tempo_multiplier)
    while rem > 2.0:
        filters.append("atempo=2.0")
        rem /= 2.0
    while rem < 0.5:
        filters.append("atempo=0.5")
        rem /= 0.5
    filters.append(f"atempo={rem:.6f}")
    filter_chain = ",".join(filters)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(input_wav),
        "-filter:a", filter_chain,
        "-ar", "44100",
        "-ac", "2",
        str(output_wav)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        logger.error(f"FFmpeg time-stretch error: {res.stderr}")
        raise RuntimeError(f"FFmpeg time-stretch failed: {res.stderr[-500:]}")

    logger.info(f"Time-stretched audio {input_wav.name} -> {output_wav.name} (Multiplier: {tempo_multiplier:.2f}x, Filter: {filter_chain})")
    return output_wav


def stretch_midi_tempo(
    input_midi_path: Path,
    output_midi_path: Path,
    tempo_multiplier: float = 1.0
) -> Path:
    """
    Scale MIDI timeline by tempo_multiplier:
    - Target tempo faster (tempo_multiplier > 1.0): notes are compressed in time (note.start /= tempo_multiplier).
    - Target tempo slower (tempo_multiplier < 1.0): notes are expanded in time (note.start /= tempo_multiplier).
    - Control changes, pitch bends, and tempo events are scaled synchronously without changing note pitches.
    """
    input_midi_path = Path(input_midi_path)
    output_midi_path = Path(output_midi_path)
    output_midi_path.parent.mkdir(parents=True, exist_ok=True)

    if abs(tempo_multiplier - 1.0) < 0.005:
        if input_midi_path.resolve() != output_midi_path.resolve():
            shutil.copy2(input_midi_path, output_midi_path)
        return output_midi_path

    midi_data = pretty_midi.PrettyMIDI(str(input_midi_path))
    time_factor = 1.0 / tempo_multiplier

    for inst in midi_data.instruments:
        for note in inst.notes:
            note.start = round(note.start * time_factor, 4)
            note.end = round(max(note.start + 0.03, note.end * time_factor), 4)
        for cc in inst.control_changes:
            cc.time = round(cc.time * time_factor, 4)
        for pb in inst.pitch_bends:
            pb.time = round(pb.time * time_factor, 4)

    # Adjust initial tempo if present
    try:
        if hasattr(midi_data, "estimate_tempo"):
            base_t = midi_data.estimate_tempo()
            new_t = base_t * tempo_multiplier
            midi_data.initial_tempo = float(np.clip(new_t, 30.0, 300.0))
    except Exception:
        pass

    midi_data.write(str(output_midi_path))
    logger.info(f"Scaled MIDI tempo: {input_midi_path.name} -> {output_midi_path.name} (Multiplier: {tempo_multiplier:.2f}x)")
    return output_midi_path


CHORD_TEMPLATES: Dict[str, Tuple[int, ...]] = {
    "": (0, 4, 7),           # Major triad
    "m": (0, 3, 7),          # Minor triad
    "7": (0, 4, 7, 10),      # Dominant 7th
    "maj7": (0, 4, 7, 11),   # Major 7th
    "m7": (0, 3, 7, 10),     # Minor 7th
    "sus4": (0, 5, 7),       # Suspended 4th
    "sus2": (0, 2, 7),       # Suspended 2nd
    "dim": (0, 3, 6),        # Diminished
    "aug": (0, 4, 8),        # Augmented
    "5": (0, 7),             # Power Chord
}


def identify_chord_from_pitches(pitches: List[int]) -> str:
    """
    Identify standard chord name from a collection of active MIDI pitches:
    - Computes pitch classes (0..11) mod 12.
    - Evaluates candidate templates across 12 chromatic roots using F1-score.
    - Prioritizes the lowest sounding pitch (bass root) with a strong match bonus.
    """
    if not pitches:
        return "N.C."
    if len(pitches) == 1:
        return PITCH_NAMES[pitches[0] % 12]

    active_pcs = set(p % 12 for p in pitches)
    lowest_pitch = min(pitches)
    bass_pc = lowest_pitch % 12

    best_chord = "N.C."
    best_score = -1.0

    for root_idx in range(12):
        for chord_type, template in CHORD_TEMPLATES.items():
            template_pcs = set((root_idx + offset) % 12 for offset in template)
            intersection = active_pcs.intersection(template_pcs)
            if not intersection:
                continue

            precision = len(intersection) / len(active_pcs)
            recall = len(intersection) / len(template_pcs)
            f1 = 2.0 * precision * recall / (precision + recall + 1e-6)

            # Bonus if root matches the lowest sounding note (Bass Root)
            bass_bonus = 0.35 if root_idx == bass_pc else 0.0

            # Slight penalty for complex chords if a simpler triad fits well
            complexity_penalty = 0.05 if len(template) > 3 else 0.0

            total_score = f1 + bass_bonus - complexity_penalty
            if total_score > best_score:
                best_score = total_score
                best_chord = f"{PITCH_NAMES[root_idx]}{chord_type}"

    if best_score < 0.40 and len(active_pcs) > 0:
        return PITCH_NAMES[bass_pc]

    return best_chord


def extract_chord_progression_from_midi(
    midi_path: Path,
    window_sec: float = 2.0,
    min_active_notes: int = 1
) -> List[Dict[str, Any]]:
    """
    Extract time-synchronized chord progression from MIDI:
    - Slices timeline into windows of window_sec.
    - Groups sounding notes in each window and identifies the chord.
    - Consolidates consecutive identical chords into contiguous time spans.
    - Returns structured progression: [{"time": 0.0, "end_time": 4.0, "chord": "C", "bar": 1}, ...]
    """
    midi_file = Path(midi_path)
    if not midi_file.exists():
        return []

    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_file))
    except Exception as e:
        logger.warning(f"Could not load MIDI for chord extraction: {e}")
        return []

    all_notes: List[pretty_midi.Note] = []
    for inst in midi_data.instruments:
        if not inst.is_drum:
            all_notes.extend(inst.notes)

    if not all_notes:
        return []

    total_duration = max(n.end for n in all_notes)
    if total_duration <= 0.1:
        return []

    # Dynamic window based on estimated tempo (approx 2 beats)
    try:
        tempo = midi_data.estimate_tempo()
        if 40.0 <= tempo <= 240.0:
            window_sec = max(1.0, min(3.0, (60.0 / tempo) * 2.0))
    except Exception:
        pass

    raw_progression = []
    t = 0.0
    while t < total_duration:
        win_start = t
        win_end = t + window_sec
        active_pitches = [
            n.pitch for n in all_notes
            if n.start < win_end and n.end > win_start
        ]

        if len(active_pitches) >= min_active_notes:
            chord = identify_chord_from_pitches(active_pitches)
        else:
            chord = raw_progression[-1]["chord"] if raw_progression else "C"

        raw_progression.append({
            "time": round(win_start, 2),
            "end_time": round(min(total_duration, win_end), 2),
            "chord": chord
        })
        t += window_sec

    # Merge consecutive identical chords
    merged: List[Dict[str, Any]] = []
    bar_index = 1
    for step in raw_progression:
        if not merged:
            step["bar"] = bar_index
            merged.append(step)
        else:
            last = merged[-1]
            if last["chord"] == step["chord"]:
                last["end_time"] = step["end_time"]
            else:
                bar_index += 1
                step["bar"] = bar_index
                merged.append(step)

    logger.info(f"Extracted {len(merged)} chord progressions from {midi_file.name}")
    return merged


def format_chord_progression_text(chords: List[Dict[str, Any]], title: str = "") -> str:
    """Format chord progression into clean, printable text representation."""
    lines = []
    if title:
        lines.append(f"=== BẢNG HỢP ÂM BÀI HÁT: {title.upper()} ===")
    else:
        lines.append("=== BẢNG HỢP ÂM BÀI HÁT ===")
    lines.append(f"Tổng số hợp âm: {len(chords)}")
    lines.append("-" * 45)
    lines.append(f"{'MỐC THỜI GIAN':<18} | {'Ô NHỊP':<8} | {'HỢP ÂM'}")
    lines.append("-" * 45)

    for c in chords:
        st_m, st_s = divmod(int(c["time"]), 60)
        et_m, et_s = divmod(int(c["end_time"]), 60)
        time_str = f"{st_m:02d}:{st_s:02d} - {et_m:02d}:{et_s:02d}"
        bar_str = f"Ô {c.get('bar', 1)}"
        lines.append(f"{time_str:<18} | {bar_str:<8} | {c['chord']}")

    lines.append("-" * 45)
    lines.append("Xuất bản từ AI Multi-Instrument Cover Converter Studio")
    return "\n".join(lines)


class AudioProcessor:
    """
    State-of-the-Art Music Cover Transformation Pipeline:
      1. Normalize input audio/video to WAV 44.1kHz stereo.
      2. 4-Stem separation via Demucs (htdemucs --shifts=1, CUDA priority).
      3. Complete drum isolation: zero drum transients fed to Basic Pitch.
      4. Polyphonic transcription via Spotify Basic Pitch.
      5. Solo / Fingerstyle mode: bóc nốt cả vocal và accompaniment, ghép thành bản độc tấu.
      6. Transpose (đổi tone tăng/giảm nửa cung) cho cả MIDI và vocal gốc.
      7. Dedicated Studio SoundFonts: Nice Steinway Lite Grand (74MB) & Guitars Universal (157MB).
      8. Smart Sustain Pedal (CC 64) & Natural Dynamics Curve for expressive playing.
      9. FluidSynth room reverberation + chorus synthesis.
      10. FFmpeg Studio Mastering Chain (EQ Mud Cut, Air Boost, Stereo Enhancer & Spotify EBU R128).
      11. 3 Acoustic Presets: Studio Warmth, Concert Hall, Lo-Fi Chill.
      12. Automatic History Management: lưu trữ outputs/history/<timestamp>_<song>/
    """

    def __init__(
        self,
        soundfont_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        self.ffmpeg_bin = ensure_ffmpeg()
        self.fluidsynth_bin = ensure_fluidsynth()
        
        # Resolve soundfonts
        self.soundfonts = ensure_soundfonts()
        self.general_sf = Path(self.soundfonts.get("general_gs", PROJECT_ROOT / "soundfonts" / "GeneralUser-GS.sf2"))
        
        # Priority for studio piano: Nice-Steinway-Lite (74MB) -> Abbey-Steinway-D -> GeneralUser
        self.studio_piano_sf = Path(self.soundfonts.get("studio_piano", PROJECT_ROOT / "soundfonts" / "Nice-Steinway-Lite-v3.0.sf2"))
        if not self.studio_piano_sf.exists():
            fallback_piano = PROJECT_ROOT / "soundfonts" / "Abbey-Steinway-D-v1.9.sf2"
            self.studio_piano_sf = fallback_piano if fallback_piano.exists() else self.general_sf

        # Priority for studio guitar: Guitars-Universal (157MB) -> Best-of-Guitars -> GeneralUser
        self.studio_guitar_sf = Path(self.soundfonts.get("studio_guitar", PROJECT_ROOT / "soundfonts" / "Guitars-Universal-V1.5.sf2"))
        if not self.studio_guitar_sf.exists():
            fallback_guitar = PROJECT_ROOT / "soundfonts" / "Best-of-Guitars-4U-v1.0.sf2"
            self.studio_guitar_sf = fallback_guitar if fallback_guitar.exists() else self.general_sf
            
        self.output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.history_dir = self.output_dir / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)

        # Smart Cache directory for Demucs stems
        self.cache_dir = PROJECT_ROOT / "temp" / "cache_stems"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # OMR Sheet Music & Notation Engine
        self.omr_engine = SheetMusicOMR(temp_dir=PROJECT_ROOT / "temp" / "omr")

        logger.info(f"AudioProcessor initialized.")
        logger.info(f" - FFmpeg: {self.ffmpeg_bin}")
        logger.info(f" - FluidSynth: {self.fluidsynth_bin}")
        logger.info(f" - SoundFont GM: {self.general_sf}")
        logger.info(f" - SoundFont Studio Piano: {self.studio_piano_sf}")
        logger.info(f" - SoundFont Studio Guitar: {self.studio_guitar_sf}")
        logger.info(f" - OMR Sheet Music Engine: Initialized")
        logger.info(f" - History Directory: {self.history_dir}")
        logger.info(f" - Demucs Stem Cache Directory: {self.cache_dir}")

    def clear_stem_cache(self) -> int:
        """Clear all cached Demucs stem directories to free disk space."""
        if not self.cache_dir.exists():
            return 0
        removed_count = 0
        for item in self.cache_dir.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                    removed_count += 1
                elif item.is_file():
                    item.unlink(missing_ok=True)
                    removed_count += 1
            except Exception as e:
                logger.warning(f"Could not remove cache entry {item}: {e}")
        logger.info(f"Cleared {removed_count} Demucs stem cache entries from {self.cache_dir}")
        return removed_count

    def detect_audio_key(self, audio_file: Any) -> Dict[str, Any]:
        """Helper method to detect musical key and tonal center of given audio input."""
        return detect_musical_key(Path(audio_file))

    def normalize_to_wav(
        self,
        input_file: Path,
        output_wav: Path,
        start_sec: Optional[float] = None,
        duration_sec: Optional[float] = None
    ) -> Path:
        """
        Convert input audio/video into clean WAV 44.1kHz 16-bit Stereo.
        Supports optional trimming (start_sec, duration_sec) for fast preview mode.
        """
        slice_info = f" (start={start_sec}s, duration={duration_sec}s)" if (start_sec or duration_sec) else ""
        logger.info(f"[Step 1] Normalizing audio from {input_file.name} to WAV 44.1kHz{slice_info}...")
        cmd = [self.ffmpeg_bin, "-y"]
        if start_sec is not None and float(start_sec) > 0:
            cmd.extend(["-ss", str(float(start_sec))])
        cmd.extend(["-i", str(input_file)])
        if duration_sec is not None and float(duration_sec) > 0:
            cmd.extend(["-t", str(float(duration_sec))])
        cmd.extend([
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            str(output_wav)
        ])
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            logger.error(f"FFmpeg normalization error: {res.stderr}")
            raise RuntimeError(f"FFmpeg failed to normalize audio: {res.stderr[-500:]}")
        
        return output_wav

    def adapt_midi_instrument(self, input_midi_path: Path, output_midi_path: Path, program_number: int, **kwargs) -> Path:
        """Backwards compatibility alias for clean_and_prepare_midi."""
        return self.clean_and_prepare_midi(input_midi_path, output_midi_path, program_number=program_number, **kwargs)

    def mix_vocals_and_instrument(self, vocals_wav: Path, instrument_wav: Path, output_path: Path, **kwargs) -> Path:
        """Backwards compatibility alias for mix_vocals_and_instrument_to_mp3."""
        return self.mix_vocals_and_instrument_to_mp3(vocals_wav, instrument_wav, output_mp3=output_path, **kwargs)

    def separate_stems(self, input_wav: Path, work_dir: Path) -> Dict[str, Path]:
        """
        Separate audio into 4 stems: vocals, drums, bass, other using Demucs.
        Prioritizes CUDA GPU if available. Uses --shifts=1 for faster processing.
        Includes SHA-256 smart caching in temp/cache_stems/ to bypass repeat processing.
        """
        required_stems = ["vocals", "drums", "bass", "other"]
        file_hash = compute_file_sha256(input_wav)
        cached_stem_dir = self.cache_dir / file_hash

        demucs_out = work_dir / "demucs_out"
        demucs_out.mkdir(parents=True, exist_ok=True)
        stem_name = input_wav.stem
        target_stem_dir = demucs_out / "htdemucs" / stem_name
        target_stem_dir.mkdir(parents=True, exist_ok=True)

        # Check Cache Hit
        if cached_stem_dir.exists():
            cached_files_exist = all(
                (cached_stem_dir / f"{s}.wav").exists() and (cached_stem_dir / f"{s}.wav").stat().st_size > 0
                for s in required_stems
            )
            if cached_files_exist:
                logger.info(f"[CACHE HIT] Found cached Demucs stems for hash {file_hash[:12]}... Bypassing stem separation!")
                stems: Dict[str, Path] = {}
                for s in required_stems:
                    src_p = cached_stem_dir / f"{s}.wav"
                    dst_p = target_stem_dir / f"{s}.wav"
                    if not dst_p.exists() or dst_p.stat().st_size == 0:
                        shutil.copy2(src_p, dst_p)
                    stems[s] = dst_p
                logger.info(
                    f"Demucs 4-stem restored from cache successfully: "
                    f"vocals ({stems['vocals'].stat().st_size}B), drums ({stems['drums'].stat().st_size}B), "
                    f"bass ({stems['bass'].stat().st_size}B), other ({stems['other'].stat().st_size}B)"
                )
                return stems

        logger.info(f"[CACHE MISS] No cached stems found for hash {file_hash[:12]}. Running Demucs separation...")
        logger.info(f"[Step 2] Separating audio into 4 stems (vocals, drums, bass, other) with Demucs...")

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Demucs processing device: {device.upper()}")

        cmd = [
            sys.executable,
            "-m", "demucs.separate",
            "-n", "htdemucs",
            "--shifts=1",
            "-o", str(demucs_out),
            "-d", device,
        ]
        
        if device == "cpu":
            try:
                cpu_count = max(1, (os.cpu_count() or 4) - 1)
                cmd.extend(["-j", str(min(4, cpu_count))])
            except Exception:
                pass
            
        cmd.append(str(input_wav))
        
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            logger.error(f"Demucs separation error: {res.stderr}")
            raise RuntimeError(f"Demucs failed: {res.stderr[-500:]}")

        stems: Dict[str, Path] = {}
        for s in required_stems:
            p = target_stem_dir / f"{s}.wav"
            if not p.exists():
                matches = list(demucs_out.glob(f"**/{s}.wav"))
                if matches:
                    p = matches[0]
                else:
                    raise FileNotFoundError(
                        f"Demucs stem '{s}.wav' not found in expected directory: {target_stem_dir}"
                    )
            stems[s] = p

        # Cache newly separated stems
        try:
            cached_stem_dir.mkdir(parents=True, exist_ok=True)
            for s in required_stems:
                shutil.copy2(stems[s], cached_stem_dir / f"{s}.wav")
            logger.info(f"[CACHE SAVE] Saved Demucs stems to cache ({cached_stem_dir.name}) for hash {file_hash[:12]}.")
        except Exception as cache_err:
            logger.warning(f"Could not save stems to cache: {cache_err}")

        logger.info(
            f"Demucs 4-stem separation successful: "
            f"vocals ({stems['vocals'].stat().st_size}B), drums ({stems['drums'].stat().st_size}B), "
            f"bass ({stems['bass'].stat().st_size}B), other ({stems['other'].stat().st_size}B)"
        )
        return stems

    def prepare_harmonic_accompaniment(
        self,
        other_wav: Path,
        bass_wav: Optional[Path],
        output_wav: Path,
        include_bass: bool = True,
        bass_highpass_freq: int = 75,
        bass_volume: float = 0.5
    ) -> Path:
        """
        Create drum-free harmonic accompaniment track for Basic Pitch transcription.
        Excludes drums entirely to prevent kick/snare transients from generating false notes.
        Optionally blends 'other' with high-pass filtered 'bass' for chord root clarity.
        """
        logger.info("Preparing drum-free harmonic accompaniment for MIDI transcription...")
        if not include_bass or not bass_wav or not bass_wav.exists():
            shutil.copy2(other_wav, output_wav)
            logger.info("Using pure 'other' track (100% drum & bass free).")
            return output_wav

        # High-pass filter bass to remove muddy sub-rumble, mix with 'other'
        filter_complex = (
            f"[1:a]highpass=f={bass_highpass_freq},volume={bass_volume}[b];"
            f"[0:a][b]amix=inputs=2:duration=longest:normalize=0"
        )
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i", str(other_wav),
            "-i", str(bass_wav),
            "-filter_complex", filter_complex,
            "-ar", "44100",
            str(output_wav)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            logger.warning(f"Error mixing other + bass: {res.stderr}. Falling back to 'other' only.")
            shutil.copy2(other_wav, output_wav)
        else:
            logger.info("Successfully blended 'other' + high-pass filtered 'bass'.")

        return output_wav

    def audio_to_midi(
        self,
        audio_path: Path,
        output_dir: Path,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        min_note_length_ms: float = 110.0
    ) -> Path:
        """
        Transcribe audio track to MIDI using Spotify Basic Pitch.
        """
        logger.info(f"Transcribing {audio_path.name} to MIDI with Basic Pitch...")
        from basic_pitch.inference import predict_and_save
        from basic_pitch import ICASSP_2022_MODEL_PATH

        output_dir.mkdir(parents=True, exist_ok=True)
        target_mid = output_dir / f"{audio_path.stem}_basic_pitch.mid"
        if target_mid.exists():
            target_mid.unlink(missing_ok=True)

        predict_and_save(
            audio_path_list=[str(audio_path)],
            output_directory=str(output_dir),
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=min_note_length_ms,
        )

        expected_mid = output_dir / f"{audio_path.stem}_basic_pitch.mid"
        if not expected_mid.exists():
            mid_files = list(output_dir.glob("*.mid"))
            if mid_files:
                expected_mid = mid_files[0]
            else:
                raise FileNotFoundError(f"Basic Pitch did not output expected MIDI file in {output_dir}")

        logger.info(f"Basic Pitch MIDI generated: {expected_mid}")
        return expected_mid

    def clean_and_prepare_midi(
        self,
        input_midi_path: Path,
        output_midi_path: Path,
        program_number: int,
        transpose_semitones: int = 0,
        min_pitch: int = 36,             # C2 = MIDI 36
        max_pitch: int = 127,
        min_duration_sec: float = 0.060, # 60ms
        enable_sustain: bool = True,
        is_guitar: bool = False,
        beat_times: Optional[np.ndarray] = None
    ) -> Path:
        """
        Pre-process standard accompaniment MIDI transcription:
          - Filter out notes < min_duration_sec (< 60ms).
          - Filter out notes outside instrument pitch range [min_pitch, max_pitch].
          - Shift pitch by transpose_semitones.
          - Trim overlapping notes of same pitch (fix_overlapping_notes).
          - Apply natural velocity dynamics curve (apply_natural_dynamics).
          - Apply beat-aware groove dynamic accentuation (downbeat boost, passing note taper, micro-timing).
          - Strumming delay humanization for guitars (staggered 15-25ms delay, +/-5% vel).
          - Generate Smart Sustain Pedal (CC 64) for piano, electric piano, harp.
          - Assign target GM Instrument Program.
        """
        midi_data = pretty_midi.PrettyMIDI(str(input_midi_path))
        tempo = 120.0
        try:
            est_tempo = midi_data.estimate_tempo()
            if 40.0 <= est_tempo <= 240.0:
                tempo = est_tempo
        except Exception:
            pass

        new_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=program_number)

        raw_notes = []
        for inst in midi_data.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                duration = note.end - note.start
                if duration < min_duration_sec:
                    continue
                if note.pitch < min_pitch or note.pitch > max_pitch:
                    continue

                shifted_pitch = max(0, min(127, note.pitch + transpose_semitones))
                raw_notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=shifted_pitch,
                        start=note.start,
                        end=note.end
                    )
                )

        # 1. Overlapping note fix (prevents harsh double-triggers in FluidSynth)
        cleaned_notes = fix_overlapping_notes(raw_notes, min_duration_sec=0.04)

        # 2. Natural velocity dynamics smoothing & beat-aware groove accentuation
        shaped_notes = apply_natural_dynamics(cleaned_notes, is_lead_melody=False)
        shaped_notes = apply_groove_and_beat_dynamics(shaped_notes, beat_times=beat_times, tempo=tempo)

        # 3. Guitar Strumming Delay Humanization (staggered 15-25ms delay across strings)
        if is_guitar:
            shaped_notes = humanize_guitar_strumming(shaped_notes, is_guitar=True)
            shaped_notes = fix_overlapping_notes(shaped_notes, min_duration_sec=0.04)

        instrument.notes = shaped_notes

        # 4. Smart Sustain Pedal CC 64 (Piano, Electric Piano, Harp)
        if enable_sustain and program_number in [0, 1, 2, 3, 4, 5, 46]:
            instrument.control_changes = generate_smart_sustain_events(shaped_notes, tempo=tempo)

        new_midi.instruments.append(instrument)
        new_midi.write(str(output_midi_path))
        logger.info(
            f"Standard MIDI prepared ({output_midi_path.name}): "
            f"Kept {len(shaped_notes)} notes | Program: {program_number} | Transpose: {transpose_semitones:+d} | CC64 Events: {len(instrument.control_changes)}"
        )
        return output_midi_path

    def combine_solo_fingerstyle_midi(
        self,
        chords_midi_path: Path,
        melody_midi_path: Path,
        output_midi_path: Path,
        program_number: int,
        transpose_semitones: int = 0,
        min_pitch: int = 36,
        max_pitch: int = 127,
        min_duration_sec: float = 0.060,
        enable_sustain: bool = True,
        is_guitar: bool = False,
        beat_times: Optional[np.ndarray] = None
    ) -> Path:
        """
        Merge chord accompaniment (from other.wav) and lead vocal melody (from vocals.wav)
        into a single unified instrument track for Solo / Fingerstyle performance:
        - Applies natural dynamic separation: lead vocal melody notes are prominent (~102-120),
          while chord accompaniment notes stay gentle and supportive (~75-88).
        - Applies beat-aware groove dynamic accentuation to accompaniment chords.
        - Strumming delay humanization for guitar chords (15-25ms staggered delay).
        - Fixes overlapping notes across tracks to prevent double-striking identical pitches.
        - Injects Smart Sustain Pedal (CC 64) for piano, harp, and rhodes.
        """
        chords_data = pretty_midi.PrettyMIDI(str(chords_midi_path))
        melody_data = pretty_midi.PrettyMIDI(str(melody_midi_path))

        tempo = 120.0
        try:
            est = chords_data.estimate_tempo()
            if 40.0 <= est <= 240.0:
                tempo = est
        except Exception:
            pass

        solo_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=program_number)

        # 1. Collect accompaniment notes
        chord_notes = []
        for inst in chords_data.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                dur = note.end - note.start
                if dur < min_duration_sec or note.pitch < min_pitch or note.pitch > max_pitch:
                    continue
                shifted_pitch = max(0, min(127, note.pitch + transpose_semitones))
                chord_notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=shifted_pitch,
                        start=note.start,
                        end=note.end
                    )
                )

        # 2. Collect lead melody notes from vocals
        lead_notes = []
        for inst in melody_data.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                dur = note.end - note.start
                if dur < min_duration_sec or note.pitch < max(36, min_pitch - 12) or note.pitch > max_pitch:
                    continue
                shifted_pitch = max(0, min(127, note.pitch + transpose_semitones))
                lead_notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=shifted_pitch,
                        start=note.start,
                        end=note.end
                    )
                )

        # 3. Dynamic separation & beat-aware groove accentuation on chords
        chord_shaped = apply_natural_dynamics(chord_notes, is_lead_melody=False)
        chord_shaped = apply_groove_and_beat_dynamics(chord_shaped, beat_times=beat_times, tempo=tempo)
        lead_shaped = apply_natural_dynamics(lead_notes, is_lead_melody=True)

        # 4. Guitar Strumming Delay Humanization for accompaniment chords
        if is_guitar:
            chord_shaped = humanize_guitar_strumming(chord_shaped, is_guitar=True)

        # 5. Merge and fix overlapping notes
        all_notes = chord_shaped + lead_shaped
        all_notes.sort(key=lambda n: n.start)
        cleaned_notes = fix_overlapping_notes(all_notes, min_duration_sec=0.04)
        instrument.notes = cleaned_notes

        # 6. Smart Sustain Pedal
        if enable_sustain and program_number in [0, 1, 2, 3, 4, 5, 46]:
            instrument.control_changes = generate_smart_sustain_events(cleaned_notes, tempo=tempo)

        solo_midi.instruments.append(instrument)
        solo_midi.write(str(output_midi_path))
        logger.info(
            f"Solo/Fingerstyle MIDI created ({output_midi_path.name}): "
            f"{len(cleaned_notes)} total notes | Program: {program_number} | Transpose: {transpose_semitones:+d} | CC64 Events: {len(instrument.control_changes)}"
        )
        return output_midi_path

    def render_midi_to_wav(
        self,
        midi_path: Path,
        output_wav: Path,
        soundfont_file: Path,
        gain: float = 1.4,
        reverb_room_size: float = 0.62,
        reverb_damp: float = 0.35,
        reverb_level: float = 0.65,
        reverb_width: float = 0.9,
        enable_chorus: bool = False
    ) -> Path:
        """
        Synthesize MIDI into WAV using FluidSynth with configurable room reverberation & chorus.
        """
        soundfont_file = Path(soundfont_file)
        output_wav = Path(output_wav)
        midi_path = Path(midi_path)
        logger.info(f"Rendering MIDI to WAV with FluidSynth [{soundfont_file.name}] ({output_wav.name})...")
        cmd = [
            self.fluidsynth_bin,
            "-ni",
            "-g", str(gain),
            "-o", f"synth.gain={gain}",
            "-r", "44100",
            "-o", "synth.reverb.active=1",
            "-o", f"synth.reverb.room-size={reverb_room_size}",
            "-o", f"synth.reverb.damp={reverb_damp}",
            "-o", f"synth.reverb.level={reverb_level}",
            "-o", f"synth.reverb.width={reverb_width}",
        ]
        if enable_chorus:
            cmd.extend([
                "-o", "synth.chorus.active=1",
                "-o", "synth.chorus.level=0.35"
            ])
        else:
            cmd.extend(["-o", "synth.chorus.active=0"])

        cmd.extend([
            "-F", str(output_wav),
            str(soundfont_file),
            str(midi_path)
        ])

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            logger.error(f"FluidSynth error: {res.stderr}")
            raise RuntimeError(f"FluidSynth failed: {res.stderr[-500:]}")

        return output_wav

    def postprocess_and_encode_mp3(
        self,
        input_wav: Path,
        output_mp3: Path,
        acoustic_preset: str = "studio_warmth",
        apply_reverb_enhancer: bool = True,
        extra_eq: Optional[str] = None
    ) -> Path:
        """
        Studio Mastering Chain for Instrumental Cover:
        1. EQ: Cut mud frequencies (equalizer=f=280:t=q:w=1:g=-1), boost airy brilliance (treble=g=1.2:f=10000).
        2. Stereo Enhancer: Expand stereo width for an immersive soundstage.
        3. Acoustic Room Reverb (aecho) reflection matching chosen preset.
        4. Release Normalization: EBU R128 (loudnorm=I=-14:LRA=10:TP=-1.0) Spotify/Apple standard.
        5. Encode to pristine MP3 320kbps.
        """
        logger.info(f"Mastering instrumental {input_wav.name} -> MP3 320kbps [{acoustic_preset}] ({output_mp3.name})...")
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i", str(input_wav)
        ]

        preset = ACOUSTIC_PRESETS.get(acoustic_preset, ACOUSTIC_PRESETS["studio_warmth"])
        filter_parts = []

        # 1. EQ Color / Shaping (Mud cut & Air boost or Lo-Fi warmth)
        if preset.get("eq_filter"):
            filter_parts.append(preset["eq_filter"])

        # Subtle mastering EQ refinement:
        # Mud cut (280Hz) + Air brilliance boost (10kHz)
        filter_parts.append("equalizer=f=280:t=q:w=1:g=-1")
        filter_parts.append("treble=g=1.2:f=10000")

        # Instrument-specific EQ (e.g. Flute breath tone boost at 4kHz)
        if extra_eq:
            filter_parts.append(extra_eq)

        # 2. Stereo Width Enhancer
        stereo_w = preset.get("stereo_width", 1.22)
        filter_parts.append(f"extrastereo=m={stereo_w:.2f}")

        # 3. Acoustic Space Echo / Reverb
        if apply_reverb_enhancer and preset.get("echo_filter"):
            filter_parts.append(preset["echo_filter"])

        # 4. Industry Standard Loudness Normalization for Instrumental Master:
        # loudnorm=I=-14:LRA=10:TP=-1.0 (Spotify/YouTube/Apple standard)
        filter_parts.append("loudnorm=I=-14:LRA=10:TP=-1.0")

        cmd.extend([
            "-af", ",".join(filter_parts),
            "-c:a", "libmp3lame",
            "-b:a", "320k",
            str(output_mp3)
        ])

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            logger.error(f"FFmpeg MP3 encoding error: {res.stderr}")
            raise RuntimeError(f"FFmpeg MP3 encoding failed: {res.stderr[-500:]}")

        return output_mp3

    def mix_vocals_and_instrument_to_mp3(
        self,
        vocals_wav: Path,
        instrument_wav: Path,
        output_mp3: Path,
        vocal_volume: float = 0.85,
        inst_volume: float = 1.25,
        transpose_semitones: int = 0,
        acoustic_preset: str = "studio_warmth",
        apply_reverb_enhancer: bool = True,
        extra_eq: Optional[str] = None
    ) -> Path:
        """
        Studio Mastering Chain for Vocal + Instrument Mix:
        1. Vocal Cleaning Chain:
           - Highpass filter at 85Hz (highpass=f=85) to eliminate sub-bass mud.
           - De-essing EQ at 6000Hz (equalizer=f=6000:t=q:w=1.5:g=-1.5) to soften harsh sibilance.
           - Pitch transposition if requested (asetrate/atempo/aresample).
           - Split into sidechain control signal and mix stream via asplit=2.
        2. Instrument EQ Carving:
           - Carve 1kHz - 3kHz pocket (equalizer=f=2000:width_type=q:w=1.0:g=-2.8) to eliminate frequency masking.
           - Extra instrument EQ (e.g. breath tone for flute).
        3. Dynamic Sidechain Ducking Compressor:
           - Automatic ducking via sidechaincompress=threshold=0.08:ratio=2.5:attack=15:release=250:
             * When singer sings: instrument ducks gently by ~1.5dB - 2.5dB.
             * When singer pauses or takes breath: instrument dynamically blooms back up to fill the acoustic space.
        4. amix blend with balanced ratios + shared acoustic space reverb.
        5. Industry release normalization: EBU R128 (loudnorm=I=-14:LRA=9:TP=-1.0) -> pristine MP3 320kbps.
        """
        logger.info(f"Mixing vocals and {instrument_wav.name} -> MP3 [{acoustic_preset}] (Transpose: {transpose_semitones:+d}) with Sidechain Ducking...")

        preset = ACOUSTIC_PRESETS.get(acoustic_preset, ACOUSTIC_PRESETS["studio_warmth"])

        # 1. Vocal Cleaning Chain & Transpose
        vocal_filters = []
        if transpose_semitones != 0:
            pitch_ratio = 2.0 ** (transpose_semitones / 12.0)
            vocal_filters.extend([
                f"asetrate=44100*{pitch_ratio:.6f}",
                f"atempo={1.0/pitch_ratio:.6f}",
                "aresample=44100"
            ])
        vocal_filters.extend([
            "highpass=f=85",
            "equalizer=f=6000:t=q:w=1.5:g=-1.5",
            f"volume={vocal_volume:.2f}"
        ])
        vocal_chain_str = ",".join(vocal_filters)

        # 2. Instrument EQ Carving (Carve pocket for vocal clarity)
        inst_filters = [f"volume={inst_volume:.2f}", "equalizer=f=2000:width_type=q:w=1.0:g=-2.8"]
        if extra_eq:
            inst_filters.append(extra_eq)
        inst_chain_str = ",".join(inst_filters)

        # 3. Shared space blending & Loudness Normalization
        blend_parts = []
        if apply_reverb_enhancer and preset.get("vocal_blend_reverb"):
            blend_parts.append(preset["vocal_blend_reverb"])
        blend_parts.append("loudnorm=I=-14:LRA=9:TP=-1.0")
        blend_chain_str = ",".join(blend_parts)

        # Complete Filtergraph:
        # [0:a] Vocal -> Cleaned -> asplit into [voc_sc] (sidechain trigger) and [voc_mix] (mix audio)
        # [1:a] Instrument -> EQ Carving -> [inst_eq]
        # [inst_eq][voc_sc] Sidechain Compress -> [ducked_inst]
        # [ducked_inst][voc_mix] amix -> Reverb -> loudnorm
        filter_str = (
            f"[0:a]{vocal_chain_str},asplit=2[voc_sc][voc_mix];"
            f"[1:a]{inst_chain_str}[inst_eq];"
            f"[inst_eq][voc_sc]sidechaincompress=threshold=0.08:ratio=2.5:attack=15:release=250[ducked_inst];"
            f"[ducked_inst][voc_mix]amix=inputs=2:duration=longest:dropout_transition=2:normalize=0,{blend_chain_str}"
        )

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i", str(vocals_wav),
            "-i", str(instrument_wav),
            "-filter_complex", filter_str,
            "-c:a", "libmp3lame",
            "-b:a", "320k",
            str(output_mp3)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            logger.error(f"FFmpeg mixing error: {res.stderr}")
            raise RuntimeError(f"FFmpeg mix failed: {res.stderr[-500:]}")

        return output_mp3

    def render_acoustic_bass(
        self,
        bass_source: Path,
        output_wav: Path,
        temp_dir: Path,
        soundfont_file: Optional[Path] = None,
        gain: float = 1.35,
        tempo_multiplier: float = 1.0,
        transpose_semitones: int = 0
    ) -> Path:
        """
        Extract bass notes and render an authentic Acoustic Upright Bass track (GM Program 32):
        - Transcribes Demucs bass stem (or low register) to MIDI with pitch clamped to [28..60] (E1 - C4).
        - Synthesizes via FluidSynth using GeneralUser-GS Acoustic Upright Bass (Program 32).
        """
        bass_source = Path(bass_source)
        output_wav = Path(output_wav)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf = Path(soundfont_file) if soundfont_file else self.general_sf

        bass_midi = temp_dir / "acoustic_upright_bass.mid"
        raw_bass_midi = None

        if bass_source.suffix.lower() in [".wav", ".mp3", ".flac"] and bass_source.exists():
            try:
                logger.info(f"Transcribing bass stem {bass_source.name} to MIDI for Acoustic Upright Bass...")
                b_dir = temp_dir / "bass_transcribe"
                b_dir.mkdir(parents=True, exist_ok=True)
                raw_bass_midi = self.audio_to_midi(
                    bass_source,
                    b_dir,
                    onset_threshold=0.45,
                    frame_threshold=0.30,
                    min_note_length_ms=100.0
                )
            except Exception as e:
                logger.warning(f"Could not transcribe bass audio directly: {e}")
                raw_bass_midi = None

        if not raw_bass_midi or not raw_bass_midi.exists():
            # If no audio or transcription failed, check if bass_source was already a MIDI file
            if bass_source.suffix.lower() == ".mid" and bass_source.exists():
                raw_bass_midi = bass_source
            else:
                logger.warning("No valid bass source MIDI found. Creating minimal bass track.")
                pm_empty = pretty_midi.PrettyMIDI(initial_tempo=120.0)
                inst = pretty_midi.Instrument(program=32)
                pm_empty.instruments.append(inst)
                pm_empty.write(str(bass_midi))
                raw_bass_midi = bass_midi

        # Clean bass MIDI: Program 32 (Acoustic Upright Bass), pitch range 28 to 60 (E1 - C4)
        self.clean_and_prepare_midi(
            input_midi_path=raw_bass_midi,
            output_midi_path=bass_midi,
            program_number=32,  # Acoustic Bass (Upright)
            transpose_semitones=transpose_semitones,
            min_pitch=28,       # E1
            max_pitch=60,       # C4
            min_duration_sec=0.08,
            enable_sustain=False,
            is_guitar=False
        )

        # Scale MIDI tempo if time-stretching is requested
        if abs(tempo_multiplier - 1.0) >= 0.005:
            stretch_midi_tempo(bass_midi, bass_midi, tempo_multiplier)

        # Synthesize via FluidSynth
        self.render_midi_to_wav(
            midi_path=bass_midi,
            output_wav=output_wav,
            soundfont_file=sf,
            gain=gain,
            reverb_room_size=0.55,
            reverb_damp=0.40,
            reverb_level=0.50,
            reverb_width=0.80,
            enable_chorus=False
        )
        logger.info(f"Rendered Acoustic Upright Bass: {output_wav.name} ({output_wav.stat().st_size} bytes)")
        return output_wav

    def mix_stems_master(
        self,
        vocals_wav: Optional[Path],
        instrument_wav: Path,
        bass_wav: Optional[Path],
        output_mp3: Path,
        vocal_volume: float = 1.0,
        inst_volume: float = 1.0,
        bass_volume: float = 0.8,
        reverb_send_pct: float = 25.0,
        transpose_semitones: int = 0
    ) -> Path:
        """
        Live Re-Mastering Engine for Interactive Stems Mini-Mixer:
        - Receives 3 stems (Vocal, Cover Instrument, Acoustic Upright Bass).
        - Applies Vocal Cleaning (HPF 85Hz, de-essing 6kHz, pitch transpose).
        - Applies Instrument EQ Carving (dip at 2kHz) and dynamic Sidechain Ducking against Vocal.
        - Applies Acoustic Bass shaping (lowpass 2.5kHz, low-end warmth boost at 80Hz).
        - Blends with Master Reverb Send (aecho proportional to reverb_send_pct).
        - Final EBU R128 mastering normalization (-14 LUFS, TP -1.0) into MP3 320kbps.
        """
        output_mp3 = Path(output_mp3)
        output_mp3.parent.mkdir(parents=True, exist_ok=True)
        instrument_wav = Path(instrument_wav)

        inputs = []
        filter_parts = []
        current_input_idx = 0

        has_vocal = bool(vocals_wav and Path(vocals_wav).exists() and vocal_volume > 0.01)
        has_bass = bool(bass_wav and Path(bass_wav).exists() and bass_volume > 0.01)

        vocal_idx = -1
        inst_idx = -1
        bass_idx = -1

        if has_vocal:
            inputs.extend(["-i", str(vocals_wav)])
            vocal_idx = current_input_idx
            current_input_idx += 1

        inputs.extend(["-i", str(instrument_wav)])
        inst_idx = current_input_idx
        current_input_idx += 1

        if has_bass:
            inputs.extend(["-i", str(bass_wav)])
            bass_idx = current_input_idx
            current_input_idx += 1

        # 1. Vocal Processing
        if has_vocal:
            vocal_filters = []
            if transpose_semitones != 0:
                pitch_ratio = 2.0 ** (transpose_semitones / 12.0)
                vocal_filters.extend([
                    f"asetrate=44100*{pitch_ratio:.6f}",
                    f"atempo={1.0/pitch_ratio:.6f}",
                    "aresample=44100"
                ])
            vocal_filters.extend([
                "highpass=f=85",
                "equalizer=f=6000:t=q:w=1.5:g=-1.5",
                f"volume={vocal_volume:.2f}"
            ])
            vocal_chain = ",".join(vocal_filters)
            filter_parts.append(f"[{vocal_idx}:a]{vocal_chain},asplit=2[voc_sc][voc_mix]")

        # 2. Instrument Processing
        inst_filters = [f"volume={inst_volume:.2f}", "equalizer=f=2000:width_type=q:w=1.0:g=-2.5"]
        filter_parts.append(f"[{inst_idx}:a]{','.join(inst_filters)}[inst_proc]")

        if has_vocal:
            filter_parts.append("[inst_proc][voc_sc]sidechaincompress=threshold=0.08:ratio=2.5:attack=15:release=250[ducked_inst]")
            active_inst = "[ducked_inst]"
        else:
            active_inst = "[inst_proc]"

        # 3. Bass Processing
        if has_bass:
            bass_filters = [
                f"volume={bass_volume:.2f}",
                "lowpass=f=2500",
                "equalizer=f=80:width_type=q:w=1.2:g=2.0"
            ]
            filter_parts.append(f"[{bass_idx}:a]{','.join(bass_filters)}[bass_proc]")

        # 4. Mix Elements
        mix_inputs = []
        if has_vocal:
            mix_inputs.append("[voc_mix]")
        mix_inputs.append(active_inst)
        if has_bass:
            mix_inputs.append("[bass_proc]")

        mix_label = "".join(mix_inputs)
        filter_parts.append(f"{mix_label}amix=inputs={len(mix_inputs)}:duration=longest:dropout_transition=2:normalize=0[mixed_raw]")

        # 5. Master Reverb Send
        reverb_gain = max(0.0, min(0.60, (reverb_send_pct / 100.0) * 0.45))
        if reverb_gain > 0.02:
            reverb_filter = f"aecho=0.8:0.85:40:{reverb_gain:.3f}"
            filter_parts.append(f"[mixed_raw]{reverb_filter}[with_reverb]")
            final_src = "[with_reverb]"
        else:
            final_src = "[mixed_raw]"

        # 6. EBU R128 Loudness Normalization
        filter_parts.append(f"{final_src}loudnorm=I=-14:LRA=9:TP=-1.0[master_out]")

        filter_str = ";".join(filter_parts)

        cmd = [
            self.ffmpeg_bin,
            "-y",
            *inputs,
            "-filter_complex", filter_str,
            "-map", "[master_out]",
            "-c:a", "libmp3lame",
            "-b:a", "320k",
            str(output_mp3)
        ]

        logger.info(f"Executing Mini-Mixer live remastering -> {output_mp3.name}...")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            logger.error(f"Mini-Mixer mastering failed: {res.stderr}")
            raise RuntimeError(f"FFmpeg Mini-Mixer mastering failed: {res.stderr[-500:]}")

        logger.info(f"Mini-Mixer master complete: {output_mp3.name} ({output_mp3.stat().st_size} bytes)")
        return output_mp3

    def prepare_monophonic_lead_midi(
        self,
        melody_midi_path: Optional[Path],
        chords_midi_path: Path,
        output_midi_path: Path,
        program_number: int = 77,
        transpose_semitones: int = 0,
        min_pitch: int = 55,
        max_pitch: int = 84,
        min_duration_sec: float = 0.050
    ) -> Path:
        """
        Produce a pure monophonic lead melody for wind instruments like Bamboo Flute (Sáo Trúc):
        - Priority 1: Extract melody directly from vocal transcription (melody_midi_path),
          giving authentic pitch contours, ornaments, and emotional vocal phrasing.
        - Priority 2: If vocal melody is not available or empty, use the Top-Note Filter on accompaniment
          chords (chords_midi_path) to isolate the highest-pitch lead melody line and eliminate all lower chords.
        - Applies Legato Smoothing (Anti-Staccato & Gap-filling <= 120ms with 25ms overlap).
        - Clamps pitch to <= 84 (C6) and velocity to 68 - 80.
        """
        source_midi_path = None
        if melody_midi_path and melody_midi_path.exists():
            try:
                m_check = pretty_midi.PrettyMIDI(str(melody_midi_path))
                total_m_notes = sum(len(i.notes) for i in m_check.instruments if not i.is_drum)
                if total_m_notes > 0:
                    source_midi_path = melody_midi_path
                    logger.info(f"Monophonic Flute using vocal lead melody source ({melody_midi_path.name}) with {total_m_notes} notes.")
            except Exception as e:
                logger.warning(f"Could not read vocal melody for flute: {e}")

        if not source_midi_path:
            source_midi_path = chords_midi_path
            logger.info(f"Monophonic Flute extracting top-note lead from accompaniment ({chords_midi_path.name}).")

        midi_data = pretty_midi.PrettyMIDI(str(source_midi_path))
        tempo = 120.0
        try:
            est_tempo = midi_data.estimate_tempo()
            if 40.0 <= est_tempo <= 240.0:
                tempo = est_tempo
        except Exception:
            pass

        raw_notes = []
        for inst in midi_data.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                shifted = max(0, min(127, note.pitch + transpose_semitones))
                raw_notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=shifted,
                        start=note.start,
                        end=note.end
                    )
                )

        # Monophonic Top-Note extraction + Legato breath extension & anti-piercing pitch clamp
        flute_notes = extract_monophonic_lead_notes(
            raw_notes,
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            min_duration_sec=min_duration_sec,
            chord_tolerance_sec=0.035,
            max_gap_for_legato=0.120,
            overlap_legato_sec=0.025
        )

        out_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=program_number)
        instrument.notes = flute_notes
        out_midi.instruments.append(instrument)
        out_midi.write(str(output_midi_path))

        logger.info(
            f"Monophonic Lead Flute MIDI created ({output_midi_path.name}): "
            f"{len(flute_notes)} notes | Program: {program_number} | Pitch: [{min_pitch}..{max_pitch}] | Transpose: {transpose_semitones:+d}"
        )
        return output_midi_path

    def render_instrument(
        self,
        base_chords_midi: Path,
        base_melody_midi: Optional[Path],
        instrument_key: str,
        output_inst_mp3: Path,
        output_voc_mp3: Optional[Path],
        vocals_wav: Optional[Path],
        temp_dir: Path,
        cover_mode: str = "chords",
        transpose_semitones: int = 0,
        soundfont_preset: str = "studio",
        acoustic_preset: str = "studio_warmth",
        vocal_volume: float = 0.85,
        inst_volume: float = 1.25,
        reverb_room_size: Optional[float] = None,
        min_midi_duration_ms: float = 60.0,
        apply_reverb_enhancer: bool = True,
        beat_times: Optional[np.ndarray] = None
    ) -> Dict[str, str]:
        """
        Synthesize and master a specific instrument cover:
        1. Adapt MIDI with GM program, humanized dynamics, Smart Sustain (CC 64) and pitch bounds.
        2. Render high-res WAV via FluidSynth with gain_boost and acoustic preset reverb.
        3. Master Instrumental MP3 320k with mud cut EQ, stereo enhancer, and EBU R128 loudness.
        4. Mix With Vocals MP3 320k with EQ carving and shared room reverb.
        """
        cfg = INSTRUMENT_CONFIGS.get(instrument_key, INSTRUMENT_CONFIGS["piano"])
        preset = ACOUSTIC_PRESETS.get(acoustic_preset, ACOUSTIC_PRESETS["studio_warmth"])
        min_duration_sec = min_midi_duration_ms / 1000.0

        # Determine SoundFont
        if soundfont_preset == "studio":
            if cfg["soundfont_type"] == "piano":
                sf = self.studio_piano_sf
            elif cfg["soundfont_type"] == "guitar":
                sf = self.studio_guitar_sf
            else:
                sf = self.general_sf
        else:
            sf = self.general_sf

        inst_midi = temp_dir / f"{instrument_key}_clean.mid"
        is_solo = "solo" in cover_mode.lower() or "fingerstyle" in cover_mode.lower()
        is_guitar = cfg.get("soundfont_type") == "guitar" or cfg["program"] in [24, 25]
        is_monophonic = cfg.get("is_monophonic", False)

        if is_monophonic:
            # 1. Monophonic Lead Melody for Bamboo Flute (Sáo Trúc)
            flute_lead_midi = temp_dir / f"{instrument_key}_lead.mid"
            self.prepare_monophonic_lead_midi(
                melody_midi_path=base_melody_midi,
                chords_midi_path=base_chords_midi,
                output_midi_path=flute_lead_midi,
                program_number=cfg["program"],
                transpose_semitones=transpose_semitones,
                min_pitch=cfg["min_pitch"],
                max_pitch=cfg["max_pitch"],
                min_duration_sec=0.050
            )

            # 2. Harmonic Backing Chords (Piano Acoustic rải nhẹ ở dải trầm-trung)
            flute_backing_midi = temp_dir / f"{instrument_key}_backing.mid"
            self.clean_and_prepare_midi(
                input_midi_path=base_chords_midi,
                output_midi_path=flute_backing_midi,
                program_number=0,  # Acoustic Grand Piano
                transpose_semitones=transpose_semitones,
                min_pitch=36,      # C2
                max_pitch=64,      # E4 (keeps piano in low-mid register to cushion the flute without masking lead)
                min_duration_sec=0.060,
                enable_sustain=True,
                is_guitar=False,
                beat_times=beat_times
            )

            # Combine tracks into inst_midi for FL Studio / DAW multi-track export
            try:
                lead_pm = pretty_midi.PrettyMIDI(str(flute_lead_midi))
                backing_pm = pretty_midi.PrettyMIDI(str(flute_backing_midi))
                est_tempo = lead_pm.estimate_tempo() or 120.0
                comb_pm = pretty_midi.PrettyMIDI(initial_tempo=est_tempo)

                track_lead = pretty_midi.Instrument(program=cfg["program"], name="Bamboo Flute Lead")
                for inst in lead_pm.instruments:
                    track_lead.notes.extend(inst.notes)
                comb_pm.instruments.append(track_lead)

                track_backing = pretty_midi.Instrument(program=0, name="Piano Chords Backing")
                for inst in backing_pm.instruments:
                    track_backing.notes.extend(inst.notes)
                    track_backing.control_changes.extend(inst.control_changes)
                comb_pm.instruments.append(track_backing)

                comb_pm.write(str(inst_midi))
            except Exception as e:
                logger.warning(f"Could not write multi-track flute MIDI: {e}")
                shutil.copy2(flute_lead_midi, inst_midi)

            # 3. Render Flute Lead Raw Audio via FluidSynth
            # Ensure soundfont for Flute Lead is ALWAYS GeneralUser-GS (never piano/guitar soundfont!)
            flute_sf = self.general_sf
            if not flute_sf.exists():
                fallback_sf = PROJECT_ROOT / "soundfonts" / "GeneralUser-GS.sf2"
                flute_sf = fallback_sf if fallback_sf.exists() else sf

            flute_lead_raw_wav = temp_dir / f"{instrument_key}_lead_raw.wav"
            self.render_midi_to_wav(
                midi_path=flute_lead_midi,
                output_wav=flute_lead_raw_wav,
                soundfont_file=flute_sf,  # GeneralUser-GS.sf2 (Program 77 Shakuhachi / Flute)
                gain=cfg.get("gain_boost", 1.45),
                reverb_room_size=0.75,
                reverb_damp=0.25,
                reverb_level=0.75,
                reverb_width=preset["fs_width"],
                enable_chorus=False
            )

            # 4. De-Harsh & Warmth Filter Chain for Bamboo Flute (Airy, warm and bold)
            # - lowpass=f=8000: cắt bớt tần số siêu cao rít the thé, bảo toàn tiếng hơi thở sáo
            # - equalizer=f=4000:t=q:w=1.5:g=-2.5: giảm nhẹ dải gắt 4kHz
            # - equalizer=f=550:t=q:w=1.2:g=2.5: tăng dải ấm/dày thân sáo mộc
            # - aecho=0.9:0.85:40:0.25: ngân vang không gian thanh thoát
            # - volume=1.35: bù đắp âm lượng, giúp Sáo Trúc nổi bật trên nền hòa tấu
            flute_lead_warm_wav = temp_dir / f"{instrument_key}_lead_warm.wav"
            flute_deharsh_filter = (
                "lowpass=f=8000,"
                "equalizer=f=4000:t=q:w=1.5:g=-2.5,"
                "equalizer=f=550:t=q:w=1.2:g=2.5,"
                "aecho=0.9:0.85:40:0.25,"
                "volume=1.35"
            )
            cmd_warm = [
                self.ffmpeg_bin,
                "-y",
                "-i", str(flute_lead_raw_wav),
                "-af", flute_deharsh_filter,
                str(flute_lead_warm_wav)
            ]
            res_warm = subprocess.run(cmd_warm, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res_warm.returncode != 0:
                logger.warning(f"Flute de-harsh filter warning: {res_warm.stderr}. Using raw lead.")
                flute_lead_warm_wav = flute_lead_raw_wav

            # 5. Render Backing Chords Audio via FluidSynth (Gentle cushion)
            flute_backing_wav = temp_dir / f"{instrument_key}_backing.wav"
            backing_sf = self.studio_piano_sf if (soundfont_preset == "studio" and self.studio_piano_sf.exists()) else self.general_sf
            self.render_midi_to_wav(
                midi_path=flute_backing_midi,
                output_wav=flute_backing_wav,
                soundfont_file=backing_sf,
                gain=0.80,
                reverb_room_size=preset["fs_room_size"],
                reverb_damp=preset["fs_damp"],
                reverb_level=preset["fs_level"],
                reverb_width=preset["fs_width"],
                enable_chorus=preset["fs_chorus"]
            )

            # 6. Mix Flute Lead + Backing Chords into inst_wav
            # In Solo mode: lead flute is 100% predominant with minimal backing (0.20x)
            # In Chords mode: lead flute is 1.45x with subtle 0.40x backing (normalize=0)
            inst_wav = temp_dir / f"{instrument_key}_ensemble.wav"
            backing_vol = 0.20 if is_solo else 0.40
            lead_vol = 1.50 if is_solo else 1.45
            mix_filter = (
                f"[0:a]volume={lead_vol}[flute];"
                f"[1:a]volume={backing_vol}[backing];"
                "[flute][backing]amix=inputs=2:duration=longest:dropout_transition=2:normalize=0[mixed]"
            )
            cmd_mix = [
                self.ffmpeg_bin,
                "-y",
                "-i", str(flute_lead_warm_wav),
                "-i", str(flute_backing_wav),
                "-filter_complex", mix_filter,
                "-map", "[mixed]",
                "-ar", "44100",
                "-ac", "2",
                str(inst_wav)
            ]
            res_mix = subprocess.run(cmd_mix, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res_mix.returncode != 0:
                logger.error(f"FFmpeg flute ensemble mix failed: {res_mix.stderr}. Using lead.")
                inst_wav = flute_lead_warm_wav

            extra_eq = None  # Flute track already has de-harsh filter chain baked in

        elif is_solo and base_melody_midi and base_melody_midi.exists():
            self.combine_solo_fingerstyle_midi(
                chords_midi_path=base_chords_midi,
                melody_midi_path=base_melody_midi,
                output_midi_path=inst_midi,
                program_number=cfg["program"],
                transpose_semitones=transpose_semitones,
                min_pitch=cfg["min_pitch"],
                max_pitch=cfg["max_pitch"],
                min_duration_sec=min_duration_sec,
                enable_sustain=True,
                is_guitar=is_guitar,
                beat_times=beat_times
            )
        else:
            self.clean_and_prepare_midi(
                input_midi_path=base_chords_midi,
                output_midi_path=inst_midi,
                program_number=cfg["program"],
                transpose_semitones=transpose_semitones,
                min_pitch=cfg["min_pitch"],
                max_pitch=cfg["max_pitch"],
                min_duration_sec=min_duration_sec,
                enable_sustain=True,
                is_guitar=is_guitar,
                beat_times=beat_times
            )

        if not is_monophonic:
            # Synthesize WAV with FluidSynth using acoustic preset reverb & chorus
            inst_wav = temp_dir / f"{instrument_key}.wav"
            room_sz = reverb_room_size if reverb_room_size is not None else cfg.get("reverb_room_size", preset["fs_room_size"])
            reverb_damp = cfg.get("reverb_damp", preset["fs_damp"])
            reverb_lvl = cfg.get("reverb_level", preset["fs_level"])

            self.render_midi_to_wav(
                midi_path=inst_midi,
                output_wav=inst_wav,
                soundfont_file=sf,
                gain=cfg["gain_boost"],
                reverb_room_size=room_sz,
                reverb_damp=reverb_damp,
                reverb_level=reverb_lvl,
                reverb_width=preset["fs_width"],
                enable_chorus=preset["fs_chorus"]
            )
            extra_eq = cfg.get("extra_eq")

        # Encode Instrumental Master MP3 320kbps
        self.postprocess_and_encode_mp3(
            input_wav=inst_wav,
            output_mp3=output_inst_mp3,
            acoustic_preset=acoustic_preset,
            apply_reverb_enhancer=apply_reverb_enhancer,
            extra_eq=extra_eq
        )

        # Mix With Vocals Master MP3 320kbps
        voc_mp3_str = None
        if vocals_wav and output_voc_mp3:
            self.mix_vocals_and_instrument_to_mp3(
                vocals_wav=vocals_wav,
                instrument_wav=inst_wav,
                output_mp3=output_voc_mp3,
                vocal_volume=vocal_volume,
                inst_volume=inst_volume,
                transpose_semitones=transpose_semitones,
                acoustic_preset=acoustic_preset,
                apply_reverb_enhancer=apply_reverb_enhancer,
                extra_eq=extra_eq
            )
            voc_mp3_str = str(output_voc_mp3)

        return {
            "instrumental": str(output_inst_mp3.resolve()),
            "with_vocals": voc_mp3_str,
            "midi": str(inst_midi.resolve())
        }

    def process_sheet_music(
        self,
        input_path: str,
        selected_instruments: Optional[List[str]] = None,
        cover_mode: str = "chords",          # 'chords' or 'solo'
        transpose_semitones: int = 0,        # -6 to +6 semitones
        tempo_multiplier: float = 1.0,       # 0.6x to 1.5x time-stretching
        soundfont_preset: str = "studio",    # 'studio' or 'standard'
        acoustic_preset: str = "studio_warmth", # 'studio_warmth', 'concert_hall', 'lofi_chill'
        tempo_bpm: float = 120.0,
        inst_volume: float = 1.25,
        include_bass: bool = True,
        min_midi_duration_ms: float = 60.0,
        reverb_room_size: Optional[float] = None,
        reverb_level: float = 0.72,
        apply_reverb_enhancer: bool = True,
        preview_mode: bool = False,
        preview_start_sec: float = 0.0,
        preview_duration_sec: float = 30.0,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute the Sheet Music / MusicXML to Multi-Instrument Cover pipeline:
        1. OMR / MusicXML parsing into standard PrettyMIDI score.
        2. Bypass Demucs stem separation for ultra-fast, clean processing.
        3. Smart Sustain (CC 64), guitar strumming humanization, and beat-aware groove.
        4. Synthesize all selected instruments via FluidSynth SoundFonts.
        5. Master with FFmpeg acoustic presets and EBU R128 (-14 LUFS).
        6. Extract Real-Time Chord Progression (Chord HUD) and persist history.
        """
        def update_progress(val: float, msg: str):
            logger.info(f"[{int(val*100)}%] {msg}")
            if progress_callback:
                progress_callback(val, msg)

        # Normalize selected_instruments
        if not selected_instruments:
            selected_instruments = ["piano", "guitar_steel"]

        valid_instruments = [k for k in selected_instruments if k in INSTRUMENT_CONFIGS]
        if not valid_instruments:
            valid_instruments = ["piano", "guitar_steel"]
        selected_instruments = valid_instruments

        # Normalize acoustic preset
        clean_preset_str = acoustic_preset.lower()
        if "concert" in clean_preset_str or "nhà hát" in clean_preset_str or "khán phòng" in clean_preset_str:
            preset_key = "concert_hall"
        elif "lofi" in clean_preset_str or "lo-fi" in clean_preset_str or "cà phê" in clean_preset_str or "chill" in clean_preset_str:
            preset_key = "lofi_chill"
        else:
            preset_key = "studio_warmth"
        preset_info = ACOUSTIC_PRESETS[preset_key]

        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input sheet file not found: {input_path}")

        raw_title = input_file.stem
        clean_title = sanitize_filename(raw_title)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # History folder
        run_history_dir = self.history_dir / f"{timestamp}_{clean_title}_Sheet"
        run_history_dir.mkdir(parents=True, exist_ok=True)

        temp_dir = Path(tempfile.mkdtemp(prefix="sheet_proc_"))

        try:
            # Stage 1: Optical Music Recognition (OMR) or MusicXML parsing (0% -> 25%)
            update_progress(0.08, "🎼 Đang quét sheet nhạc và nhận diện ký âm (OMR / MusicXML Parser)...")
            sheet_raw_midi = temp_dir / "sheet_raw.mid"
            omr_res = self.omr_engine.process(
                input_file=input_file,
                output_midi_path=sheet_raw_midi,
                tempo_bpm=tempo_bpm,
                generate_overlay=True
            )
            detected_notes_count = omr_res.get("notes_count", 0)
            detected_key = omr_res.get("detected_key", "C Major")
            update_progress(0.25, f"✅ Nhận diện thành công {detected_notes_count} nốt nhạc (Tone: {detected_key})!")

            # Copy visual overlay if present
            overlay_file_history = None
            if omr_res.get("preview_image_path") and Path(omr_res["preview_image_path"]).exists():
                overlay_dest = run_history_dir / f"{clean_title}_OMR_Detection_Overlay.png"
                try:
                    shutil.copy2(omr_res["preview_image_path"], overlay_dest)
                    overlay_file_history = str(overlay_dest.resolve())
                except Exception:
                    pass

            # Stage 2: Time-Stretching / Tempo adjustment (25% -> 35%)
            if abs(tempo_multiplier - 1.0) >= 0.005:
                update_progress(0.30, f"⏱️ Đang co giãn thời gian nốt MIDI (BPM x{tempo_multiplier:.2f})...")
                stretch_midi_tempo(sheet_raw_midi, sheet_raw_midi, tempo_multiplier)

            # Stage 3: Synthesize Acoustic Upright Bass (35% -> 45%)
            bass_wav_history = run_history_dir / f"{clean_title}_Acoustic_Bass.wav"
            if include_bass:
                update_progress(0.35, "🎻 Đang trích xuất và tổng hợp Acoustic Upright Bass (GM Program 32)...")
                try:
                    self.render_acoustic_bass(
                        bass_source=sheet_raw_midi,
                        output_wav=bass_wav_history,
                        temp_dir=temp_dir,
                        soundfont_file=self.general_sf,
                        tempo_multiplier=tempo_multiplier,
                        transpose_semitones=transpose_semitones
                    )
                except Exception as bass_err:
                    logger.warning(f"Could not render Acoustic Bass from sheet: {bass_err}")

            # Stage 4: Acoustic Synthesis for all Selected Instruments (45% -> 88%)
            if soundfont_preset == "studio":
                sf_label = "Studio Hi-Fi SoundFonts (Nice Steinway & Guitars Universal)"
            else:
                sf_label = "Standard General MIDI (GeneralUser GS)"

            instrument_results = {}
            primary_inst_wav_history = run_history_dir / f"{clean_title}_Primary_Instrument.wav"
            total_inst = len(selected_instruments)

            for idx, inst_key in enumerate(selected_instruments):
                cfg = INSTRUMENT_CONFIGS[inst_key]
                prog_val = 0.45 + (0.43 * ((idx + 1) / total_inst))
                update_progress(prog_val, f"Đang render và master nhạc cụ {cfg['name']} [{preset_info['name']}] ({idx + 1}/{total_inst})...")

                inst_mp3 = run_history_dir / f"{clean_title}_{cfg['label']}_Instrumental.mp3"

                res = self.render_instrument(
                    base_chords_midi=sheet_raw_midi,
                    base_melody_midi=sheet_raw_midi,
                    instrument_key=inst_key,
                    output_inst_mp3=inst_mp3,
                    output_voc_mp3=None,
                    vocals_wav=None,
                    temp_dir=temp_dir,
                    cover_mode=cover_mode,
                    transpose_semitones=transpose_semitones,
                    soundfont_preset=soundfont_preset,
                    acoustic_preset=preset_key,
                    vocal_volume=0.0,
                    inst_volume=inst_volume,
                    reverb_room_size=reverb_room_size,
                    min_midi_duration_ms=min_midi_duration_ms,
                    apply_reverb_enhancer=apply_reverb_enhancer
                )
                instrument_results[inst_key] = res

                # Save primary instrument WAV for Mini-Mixer
                if idx == 0:
                    first_wav = temp_dir / f"{inst_key}.wav"
                    if not first_wav.exists():
                        first_wav = temp_dir / f"{inst_key}_ensemble.wav"
                    if first_wav.exists():
                        try:
                            shutil.copy2(first_wav, primary_inst_wav_history)
                        except Exception:
                            pass

            # Stage 5: Save clean MIDI & Extract Chord Progression (88% -> 95%)
            final_midi = run_history_dir / f"{clean_title}_transcribed.mid"
            shutil.copy2(sheet_raw_midi, final_midi)

            update_progress(0.92, "🎼 Đang phân tích chuỗi hợp âm bài hát (Chord Progression)...")
            chord_progression = extract_chord_progression_from_midi(final_midi)
            chord_txt_file = run_history_dir / f"{clean_title}_chords.txt"
            try:
                chord_txt_content = format_chord_progression_text(chord_progression, raw_title)
                with open(chord_txt_file, "w", encoding="utf-8") as cf:
                    cf.write(chord_txt_content)
            except Exception as chord_err:
                logger.warning(f"Could not write chord text file: {chord_err}")

            # Stage 6: Rich Metadata & File Dicts (95% -> 100%)
            mode_label = "Sheet Music / Ký Âm OMR (Multi-Instrument Cover)"
            transpose_label = f"{transpose_semitones:+d} nửa cung" if transpose_semitones != 0 else "Tone gốc (0)"

            file_sizes = {}
            files_dict = {
                "midi_file": str(final_midi.resolve()),
                "chord_txt_file": str(chord_txt_file.resolve()) if chord_txt_file.exists() else None,
                "acoustic_bass_wav": str(bass_wav_history.resolve()) if bass_wav_history.exists() else None,
                "vocals_wav": None,
                "primary_instrument_wav": str(primary_inst_wav_history.resolve()) if primary_inst_wav_history.exists() else None,
                "omr_overlay_image": overlay_file_history
            }
            instruments_dict = {}

            for ik, iv in instrument_results.items():
                inst_p = Path(iv["instrumental"])
                instruments_dict[ik] = {
                    "instrumental": str(inst_p.resolve()),
                    "with_vocals": None
                }
                files_dict[f"{ik}_instrumental"] = str(inst_p.resolve())
                if inst_p.exists():
                    file_sizes[f"{ik}_instrumental_mb"] = round(inst_p.stat().st_size / (1024 * 1024), 2)

            if "piano" in instrument_results:
                files_dict["piano_instrumental"] = str(instrument_results["piano"]["instrumental"])
                file_sizes["piano_instrumental_mb"] = file_sizes.get("piano_instrumental_mb", 0)

            guitar_main = "guitar_steel" if "guitar_steel" in instrument_results else ("guitar_nylon" if "guitar_nylon" in instrument_results else None)
            if guitar_main:
                files_dict["guitar_instrumental"] = str(instrument_results[guitar_main]["instrumental"])
                file_sizes["guitar_instrumental_mb"] = file_sizes.get(f"{guitar_main}_instrumental_mb", 0)

            if "flute" in instrument_results:
                files_dict["bamboo_flute_instrumental"] = str(instrument_results["flute"]["instrumental"])
                file_sizes["bamboo_flute_instrumental_mb"] = file_sizes.get("flute_instrumental_mb", 0)
                instruments_dict["bamboo_flute"] = instruments_dict["flute"]
            elif "bamboo_flute" in instrument_results:
                files_dict["flute_instrumental"] = str(instrument_results["bamboo_flute"]["instrumental"])
                file_sizes["flute_instrumental_mb"] = file_sizes.get("bamboo_flute_instrumental_mb", 0)
                instruments_dict["flute"] = instruments_dict["bamboo_flute"]

            metadata = {
                "id": run_history_dir.name,
                "song_title": raw_title,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "preview_mode": False,
                "source_type": "sheet_music",
                "omr_summary": omr_res.get("summary", ""),
                "notes_count": detected_notes_count,
                "mode": mode_label,
                "transpose": transpose_label,
                "transpose_semitones": transpose_semitones,
                "tempo_multiplier": tempo_multiplier,
                "detected_key": detected_key,
                "detected_key_name": detected_key,
                "detected_key_confidence": 0.95,
                "estimated_tempo_bpm": round(tempo_bpm * tempo_multiplier, 1),
                "soundfont_preset": sf_label,
                "acoustic_preset": preset_info["name"],
                "selected_instruments": selected_instruments,
                "instruments": instruments_dict,
                "files": files_dict,
                "file_sizes": file_sizes,
                "chord_progression": chord_progression
            }
            with open(run_history_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            update_progress(1.0, f"🎉 Chuyển đổi Sheet nhạc thành công! Đã tạo {len(selected_instruments)} bản phối nhạc cụ phòng thu.")

            return {
                "history_id": run_history_dir.name,
                "song_title": raw_title,
                "mode": mode_label,
                "transpose": transpose_label,
                "detected_key": detected_key,
                "estimated_tempo_bpm": round(tempo_bpm * tempo_multiplier, 1),
                "acoustic_preset": preset_info["name"],
                "instruments": instruments_dict,
                "files": files_dict,
                "file_sizes": file_sizes,
                "midi_file": str(final_midi.resolve()),
                "chord_txt_file": str(chord_txt_file.resolve()) if chord_txt_file.exists() else None,
                "chord_progression": chord_progression,
                "omr_summary": omr_res.get("summary", ""),
                "preview_image_path": overlay_file_history
            }
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def process(
        self,
        input_path: str,
        selected_instruments: Optional[List[str]] = None,
        cover_mode: str = "chords",          # 'chords' or 'solo'
        transpose_semitones: int = 0,        # -6 to +6 semitones
        tempo_multiplier: float = 1.0,       # 0.6x to 1.5x time-stretching
        soundfont_preset: str = "studio",    # 'studio' or 'standard'
        acoustic_preset: str = "studio_warmth", # 'studio_warmth', 'concert_hall', 'lofi_chill'
        guitar_type: str = "steel",          # kept for backwards compat
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        min_note_length_ms: float = 110.0,
        vocal_volume: float = 0.85,
        inst_volume: float = 1.25,
        include_bass: bool = True,
        min_pitch: int = 36,               # C2 (36)
        min_midi_duration_ms: float = 60.0,# 60ms
        reverb_room_size: Optional[float] = None,
        reverb_level: float = 0.72,
        apply_reverb_enhancer: bool = True,
        preview_mode: bool = False,
        preview_start_sec: float = 0.0,
        preview_duration_sec: float = 30.0,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute the full cover transformation pipeline for all selected instruments
        and save results to outputs/history/.
        Supports preview_mode to process only a slice (e.g. 30s) for ultra-fast testing.
        """
        def update_progress(val: float, msg: str):
            logger.info(f"[{int(val*100)}%] {msg}")
            if progress_callback:
                progress_callback(val, msg)

        # Normalize selected_instruments
        if not selected_instruments:
            if guitar_type == "nylon":
                selected_instruments = ["piano", "guitar_nylon"]
            else:
                selected_instruments = ["piano", "guitar_steel"]

        valid_instruments = [k for k in selected_instruments if k in INSTRUMENT_CONFIGS]
        if not valid_instruments:
            valid_instruments = ["piano", "guitar_steel"]
        selected_instruments = valid_instruments

        # Normalize acoustic preset
        clean_preset_str = acoustic_preset.lower()
        if "concert" in clean_preset_str or "nhà hát" in clean_preset_str or "khán phòng" in clean_preset_str:
            preset_key = "concert_hall"
        elif "lofi" in clean_preset_str or "lo-fi" in clean_preset_str or "cà phê" in clean_preset_str or "chill" in clean_preset_str:
            preset_key = "lofi_chill"
        else:
            preset_key = "studio_warmth"
        preset_info = ACOUSTIC_PRESETS[preset_key]

        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Automatic Routing for Sheet Music / MusicXML / MIDI input
        if is_sheet_music_file(input_file):
            return self.process_sheet_music(
                input_path=str(input_file),
                selected_instruments=selected_instruments,
                cover_mode=cover_mode,
                transpose_semitones=transpose_semitones,
                tempo_multiplier=tempo_multiplier,
                soundfont_preset=soundfont_preset,
                acoustic_preset=acoustic_preset,
                inst_volume=inst_volume,
                include_bass=include_bass,
                min_midi_duration_ms=min_midi_duration_ms,
                reverb_room_size=reverb_room_size,
                reverb_level=reverb_level,
                apply_reverb_enhancer=apply_reverb_enhancer,
                preview_mode=preview_mode,
                preview_start_sec=preview_start_sec,
                preview_duration_sec=preview_duration_sec,
                progress_callback=progress_callback
            )

        raw_title = input_file.stem
        if preview_mode:
            raw_title = f"{raw_title} [Preview {int(preview_start_sec)}s-{int(preview_start_sec + preview_duration_sec)}s]"
        clean_title = sanitize_filename(raw_title)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Dedicated History Folder for this conversion
        run_history_dir = self.history_dir / f"{timestamp}_{clean_title}"
        run_history_dir.mkdir(parents=True, exist_ok=True)

        # Temporary workspace for intermediate files
        temp_dir = Path(tempfile.mkdtemp(prefix="cover_proc_"))
        
        try:
            # Stage 1: Audio Normalization (0% -> 10%)
            start_slice = preview_start_sec if preview_mode else None
            dur_slice = preview_duration_sec if preview_mode else None
            slice_desc = f" - Đoạn nghe thử {preview_start_sec:.0f}s..{preview_start_sec+dur_slice:.0f}s" if preview_mode else ""
            update_progress(0.05, f"Đang chuẩn hóa âm thanh (FFmpeg WAV 44.1kHz{slice_desc})...")
            normalized_wav = temp_dir / "normalized_input.wav"
            self.normalize_to_wav(input_file, normalized_wav, start_sec=start_slice, duration_sec=dur_slice)

            # Stage 2: Stem Separation (10% -> 45%) - 4 Stems (htdemucs --shifts=1)
            update_progress(0.10, "Đang tách 4 Stems (Vocal, Trống, Bass, Nhạc đệm) bằng Demucs...")
            stems = self.separate_stems(normalized_wav, temp_dir)

            # Stage 3: Harmonic Accompaniment Pre-conditioning (45% -> 50%)
            update_progress(0.45, "Đang cách ly hòa âm (loại bỏ 100% tiếng trống kick/snare)...")
            clean_accomp_wav = temp_dir / "harmonic_accompaniment.wav"
            self.prepare_harmonic_accompaniment(
                other_wav=stems["other"],
                bass_wav=stems["bass"],
                output_wav=clean_accomp_wav,
                include_bass=include_bass,
                bass_highpass_freq=75,
                bass_volume=0.5
            )

            # Stage 3.5: Auto Key Detection & Beat Tracking
            update_progress(0.48, "🔍 Đang tự động nhận diện Tone bài hát và dò tìm nhịp phách...")
            key_info = detect_musical_key(clean_accomp_wav)
            est_tempo_bpm, beat_times = extract_beats_and_tempo(clean_accomp_wav)
            logger.info(f"Auto Key Detection: {key_info['label']} (Confidence: {int(key_info['confidence']*100)}%) | Tempo: {est_tempo_bpm:.1f} BPM")

            # Stage 4: Polyphonic Transcription (50% -> 68%)
            update_progress(0.50, "Đang bóc tách nốt hợp âm sang MIDI (Spotify Basic Pitch)...")
            midi_dir = temp_dir / "midi"
            midi_dir.mkdir(parents=True, exist_ok=True)
            chords_raw_midi = self.audio_to_midi(
                clean_accomp_wav,
                midi_dir / "chords",
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                min_note_length_ms=min_note_length_ms
            )

            # Apply Time-Stretching to MIDI if requested
            if abs(tempo_multiplier - 1.0) >= 0.005:
                update_progress(0.65, f"⏱️ Đang co giãn thời gian nốt MIDI (BPM x{tempo_multiplier:.2f})...")
                stretch_midi_tempo(chords_raw_midi, chords_raw_midi, tempo_multiplier)

            # Stage 5: Solo Mode or Monophonic Flute Melody Extraction
            is_solo_mode = "solo" in cover_mode.lower() or "fingerstyle" in cover_mode.lower()
            needs_vocal_melody = is_solo_mode or any(
                INSTRUMENT_CONFIGS.get(k, {}).get("is_monophonic", False)
                for k in selected_instruments
            )
            melody_raw_midi = None
            if needs_vocal_melody and stems.get("vocals") and stems["vocals"].exists():
                update_progress(0.68, "🎼 Đang bóc tách nốt giai điệu từ Vocal (Basic Pitch)...")
                melody_raw_midi = self.audio_to_midi(
                    stems["vocals"],
                    midi_dir / "melody",
                    onset_threshold=0.45,
                    frame_threshold=0.28,
                    min_note_length_ms=80.0
                )
                if abs(tempo_multiplier - 1.0) >= 0.005 and melody_raw_midi and melody_raw_midi.exists():
                    stretch_midi_tempo(melody_raw_midi, melody_raw_midi, tempo_multiplier)

            # Handle Vocal time-stretching if tempo is adjusted
            active_vocals_wav = None
            if stems.get("vocals") and stems["vocals"].exists():
                if abs(tempo_multiplier - 1.0) >= 0.005:
                    update_progress(0.70, f"⏱️ Đang co giãn thời gian Vocal (atempo x{tempo_multiplier:.2f}) bảo toàn cao độ...")
                    stretched_voc = temp_dir / "stretched_vocals.wav"
                    time_stretch_audio(stems["vocals"], stretched_voc, tempo_multiplier, self.ffmpeg_bin)
                    active_vocals_wav = stretched_voc
                else:
                    active_vocals_wav = stems["vocals"]

            # Save clean vocal copy to history for Mini-Mixer
            vocals_wav_history = run_history_dir / f"{clean_title}_Vocals.wav"
            if active_vocals_wav and active_vocals_wav.exists():
                try:
                    shutil.copy2(active_vocals_wav, vocals_wav_history)
                except Exception as voc_err:
                    logger.warning(f"Could not copy vocals to history: {voc_err}")

            # Synthesize Acoustic Upright Bass (GM Program 32)
            update_progress(0.71, "🎻 Đang trích xuất và tổng hợp Acoustic Upright Bass (GM Program 32)...")
            bass_wav_history = run_history_dir / f"{clean_title}_Acoustic_Bass.wav"
            try:
                if stems.get("bass") and stems["bass"].exists():
                    self.render_acoustic_bass(
                        bass_source=stems["bass"],
                        output_wav=bass_wav_history,
                        temp_dir=temp_dir,
                        soundfont_file=self.general_sf,
                        tempo_multiplier=tempo_multiplier,
                        transpose_semitones=transpose_semitones
                    )
            except Exception as bass_err:
                logger.warning(f"Could not render Acoustic Bass: {bass_err}")

            # Stage 6: Acoustic Synthesis for all Selected Instruments (72% -> 95%)
            if soundfont_preset == "studio":
                sf_label = "Studio Hi-Fi SoundFonts (Nice Steinway & Guitars Universal)"
            else:
                sf_label = "Standard General MIDI (GeneralUser GS)"

            instrument_results = {}
            primary_inst_wav_history = run_history_dir / f"{clean_title}_Primary_Instrument.wav"
            total_inst = len(selected_instruments)

            for idx, inst_key in enumerate(selected_instruments):
                cfg = INSTRUMENT_CONFIGS[inst_key]
                prog_val = 0.72 + (0.24 * ((idx + 1) / total_inst))
                update_progress(prog_val, f"Đang render và master nhạc cụ {cfg['name']} [{preset_info['name']}] ({idx + 1}/{total_inst})...")

                inst_mp3 = run_history_dir / f"{clean_title}_{cfg['label']}_Instrumental.mp3"
                voc_mp3 = run_history_dir / f"{clean_title}_{cfg['label']}_With_Vocals.mp3"

                res = self.render_instrument(
                    base_chords_midi=chords_raw_midi,
                    base_melody_midi=melody_raw_midi,
                    instrument_key=inst_key,
                    output_inst_mp3=inst_mp3,
                    output_voc_mp3=voc_mp3,
                    vocals_wav=active_vocals_wav,
                    temp_dir=temp_dir,
                    cover_mode=cover_mode,
                    transpose_semitones=transpose_semitones,
                    soundfont_preset=soundfont_preset,
                    acoustic_preset=preset_key,
                    vocal_volume=vocal_volume,
                    inst_volume=inst_volume,
                    reverb_room_size=reverb_room_size,
                    min_midi_duration_ms=min_midi_duration_ms,
                    apply_reverb_enhancer=apply_reverb_enhancer,
                    beat_times=beat_times
                )
                instrument_results[inst_key] = res

                # Save primary instrument WAV for Mini-Mixer
                if idx == 0:
                    first_wav = temp_dir / f"{inst_key}.wav"
                    if not first_wav.exists():
                        first_wav = temp_dir / f"{inst_key}_ensemble.wav"
                    if first_wav.exists():
                        try:
                            shutil.copy2(first_wav, primary_inst_wav_history)
                        except Exception:
                            pass

            # Stage 7: Save clean MIDI copy into history & Extract Chord Progression
            final_midi = run_history_dir / f"{clean_title}_transcribed.mid"
            chosen_midi = None
            if "piano" in instrument_results and Path(instrument_results["piano"]["midi"]).exists():
                chosen_midi = instrument_results["piano"]["midi"]
            elif selected_instruments and Path(instrument_results[selected_instruments[0]]["midi"]).exists():
                chosen_midi = instrument_results[selected_instruments[0]]["midi"]

            if chosen_midi:
                shutil.copy2(chosen_midi, final_midi)
            else:
                shutil.copy2(chords_raw_midi, final_midi)

            update_progress(0.97, "🎼 Đang phân tích chuỗi hợp âm bài hát (Chord Progression)...")
            chord_progression = extract_chord_progression_from_midi(final_midi)
            chord_txt_file = run_history_dir / f"{clean_title}_chords.txt"
            try:
                chord_txt_content = format_chord_progression_text(chord_progression, raw_title)
                with open(chord_txt_file, "w", encoding="utf-8") as cf:
                    cf.write(chord_txt_content)
            except Exception as chord_err:
                logger.warning(f"Could not write chord text file: {chord_err}")

            # Stage 8: Save rich metadata.json
            mode_label = "Độc Tấu / Fingerstyle (Solo Melody + Chords)" if is_solo_mode else "Đệm Hát Tiêu Chuẩn (Chords Only)"
            transpose_label = f"{transpose_semitones:+d} nửa cung" if transpose_semitones != 0 else "Tone gốc (0)"

            file_sizes = {}
            files_dict = {
                "midi_file": str(final_midi.resolve()),
                "chord_txt_file": str(chord_txt_file.resolve()) if chord_txt_file.exists() else None,
                "acoustic_bass_wav": str(bass_wav_history.resolve()) if bass_wav_history.exists() else None,
                "vocals_wav": str(vocals_wav_history.resolve()) if vocals_wav_history.exists() else None,
                "primary_instrument_wav": str(primary_inst_wav_history.resolve()) if primary_inst_wav_history.exists() else None
            }
            instruments_dict = {}

            for ik, iv in instrument_results.items():
                inst_p = Path(iv["instrumental"])
                voc_p = Path(iv["with_vocals"]) if iv.get("with_vocals") else None
                instruments_dict[ik] = {
                    "instrumental": str(inst_p.resolve()),
                    "with_vocals": str(voc_p.resolve()) if voc_p else None
                }
                files_dict[f"{ik}_instrumental"] = str(inst_p.resolve())
                if voc_p:
                    files_dict[f"{ik}_with_vocals"] = str(voc_p.resolve())
                if inst_p.exists():
                    file_sizes[f"{ik}_instrumental_mb"] = round(inst_p.stat().st_size / (1024 * 1024), 2)
                if voc_p and voc_p.exists():
                    file_sizes[f"{ik}_with_vocals_mb"] = round(voc_p.stat().st_size / (1024 * 1024), 2)

            # Backwards-compatible legacy keys for Piano & Guitar
            if "piano" in instrument_results:
                files_dict["piano_instrumental"] = str(instrument_results["piano"]["instrumental"])
                files_dict["piano_vocal"] = str(instrument_results["piano"]["with_vocals"])
                file_sizes["piano_instrumental_mb"] = file_sizes.get("piano_instrumental_mb", 0)
                file_sizes["piano_vocal_mb"] = file_sizes.get("piano_with_vocals_mb", 0)

            guitar_main = "guitar_steel" if "guitar_steel" in instrument_results else ("guitar_nylon" if "guitar_nylon" in instrument_results else None)
            if guitar_main:
                files_dict["guitar_instrumental"] = str(instrument_results[guitar_main]["instrumental"])
                files_dict["guitar_vocal"] = str(instrument_results[guitar_main]["with_vocals"])
                file_sizes["guitar_instrumental_mb"] = file_sizes.get(f"{guitar_main}_instrumental_mb", 0)
                file_sizes["guitar_vocal_mb"] = file_sizes.get(f"{guitar_main}_with_vocals_mb", 0)

            if "flute" in instrument_results:
                files_dict["bamboo_flute_instrumental"] = str(instrument_results["flute"]["instrumental"])
                files_dict["bamboo_flute_with_vocals"] = str(instrument_results["flute"]["with_vocals"])
                file_sizes["bamboo_flute_instrumental_mb"] = file_sizes.get("flute_instrumental_mb", 0)
                file_sizes["bamboo_flute_with_vocals_mb"] = file_sizes.get("flute_with_vocals_mb", 0)
                instruments_dict["bamboo_flute"] = instruments_dict["flute"]
            elif "bamboo_flute" in instrument_results:
                files_dict["flute_instrumental"] = str(instrument_results["bamboo_flute"]["instrumental"])
                files_dict["flute_with_vocals"] = str(instrument_results["bamboo_flute"]["with_vocals"])
                file_sizes["flute_instrumental_mb"] = file_sizes.get("bamboo_flute_instrumental_mb", 0)
                file_sizes["flute_with_vocals_mb"] = file_sizes.get("bamboo_flute_with_vocals_mb", 0)
                instruments_dict["flute"] = instruments_dict["bamboo_flute"]

            metadata = {
                "id": run_history_dir.name,
                "song_title": raw_title,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "preview_mode": preview_mode,
                "preview_range": f"{preview_start_sec:.0f}s - {preview_start_sec + preview_duration_sec:.0f}s" if preview_mode else "Full Song",
                "mode": mode_label,
                "transpose": transpose_label,
                "transpose_semitones": transpose_semitones,
                "tempo_multiplier": tempo_multiplier,
                "detected_key": key_info["label"],
                "detected_key_name": key_info["key"],
                "detected_key_confidence": key_info["confidence"],
                "estimated_tempo_bpm": round(est_tempo_bpm, 1),
                "soundfont_preset": sf_label,
                "acoustic_preset": preset_info["name"],
                "selected_instruments": selected_instruments,
                "instruments": instruments_dict,
                "files": files_dict,
                "file_sizes": file_sizes,
                "chord_progression": chord_progression
            }
            with open(run_history_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            update_progress(1.0, f"🎉 Chuyển đổi thành công {len(selected_instruments)} nhạc cụ [{preset_info['name']} | {mode_label} | Tone: {transpose_label} | Tone gốc: {key_info['label']} | {est_tempo_bpm:.1f} BPM]!")

            return {
                "history_id": run_history_dir.name,
                "song_title": raw_title,
                "acoustic_preset": preset_info["name"],
                "selected_instruments": selected_instruments,
                "instruments": instrument_results,
                "midi_file": str(final_midi.resolve()),
                "detected_key": key_info["label"],
                "detected_key_confidence": key_info["confidence"],
                "estimated_tempo_bpm": round(est_tempo_bpm, 1),
                "tempo_multiplier": tempo_multiplier,
                "chord_progression": chord_progression,
                "chord_txt_file": str(chord_txt_file.resolve()) if chord_txt_file.exists() else None,
                "acoustic_bass_wav": str(bass_wav_history.resolve()) if bass_wav_history.exists() else None,
                "vocals_wav": str(vocals_wav_history.resolve()) if vocals_wav_history.exists() else None,
                "primary_instrument_wav": str(primary_inst_wav_history.resolve()) if primary_inst_wav_history.exists() else None,
                # Backwards compatibility
                "piano_instrumental": instrument_results.get("piano", {}).get("instrumental"),
                "piano_vocal": instrument_results.get("piano", {}).get("with_vocals"),
                "guitar_instrumental": instrument_results.get("guitar_steel", instrument_results.get("guitar_nylon", {})).get("instrumental"),
                "guitar_vocal": instrument_results.get("guitar_steel", instrument_results.get("guitar_nylon", {})).get("with_vocals"),
            }

        finally:
            # Clean up all temporary files & stems immediately to avoid disk overflow
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Temporary workspace cleaned completely: {temp_dir}")
            except Exception as e:
                logger.warning(f"Error cleaning temporary files: {e}")

    def list_history_entries(self) -> List[Dict]:
        """
        List all saved history entries sorted from newest to oldest.
        """
        entries = []
        if not self.history_dir.exists():
            return entries

        for sub_dir in sorted(self.history_dir.iterdir(), reverse=True):
            if sub_dir.is_dir():
                meta_file = sub_dir / "metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        if "id" not in meta:
                            meta["id"] = sub_dir.name
                        entries.append(meta)
                    except Exception as e:
                        logger.warning(f"Could not load metadata from {meta_file}: {e}")
                else:
                    mp3_files = list(sub_dir.glob("*.mp3"))
                    midi_files = list(sub_dir.glob("*.mid"))
                    if mp3_files:
                        entries.append({
                            "id": sub_dir.name,
                            "song_title": sub_dir.name,
                            "created_at": "Trước đó",
                            "mode": "Tiêu chuẩn",
                            "transpose": "Tone gốc",
                            "selected_instruments": ["piano", "guitar_steel"],
                            "files": {
                                "piano_instrumental": str(mp3_files[0]),
                                "piano_vocal": str(mp3_files[1]) if len(mp3_files) > 1 else str(mp3_files[0]),
                                "guitar_instrumental": str(mp3_files[2]) if len(mp3_files) > 2 else str(mp3_files[0]),
                                "guitar_vocal": str(mp3_files[3]) if len(mp3_files) > 3 else str(mp3_files[0]),
                                "midi_file": str(midi_files[0]) if midi_files else "",
                            }
                        })
        return entries

    def get_history_entry_files(self, entry_id: str) -> Optional[Dict]:
        """
        Retrieve audio & MIDI file paths for a given history entry.
        """
        target_dir = self.history_dir / entry_id
        if not target_dir.exists() or not target_dir.is_dir():
            return None

        meta_file = target_dir / "metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if "files" in meta:
                    for k, v in meta["files"].items():
                        if v and not Path(v).is_absolute():
                            meta["files"][k] = str((PROJECT_ROOT / v).resolve())
                if "instruments" in meta:
                    for ik, iv in meta["instruments"].items():
                        for fk, fv in iv.items():
                            if fv and not Path(fv).is_absolute():
                                meta["instruments"][ik][fk] = str((PROJECT_ROOT / fv).resolve())
                if "midi_file" in meta and not Path(meta["midi_file"]).is_absolute():
                    meta["midi_file"] = str((PROJECT_ROOT / meta["midi_file"]).resolve())
                return meta
            except Exception:
                pass

        # Fallback search inside folder
        files = {
            "midi_file": None,
        }
        instruments = {}
        for f in target_dir.iterdir():
            if f.suffix == ".mid":
                files["midi_file"] = str(f.resolve())
            elif f.suffix == ".mp3":
                name_lower = f.name.lower()
                for ik, cfg in INSTRUMENT_CONFIGS.items():
                    lbl = cfg["label"].lower()
                    if lbl in name_lower or ik in name_lower:
                        if ik not in instruments:
                            instruments[ik] = {}
                        if "vocal" in name_lower:
                            instruments[ik]["with_vocals"] = str(f.resolve())
                            files[f"{ik}_with_vocals"] = str(f.resolve())
                        else:
                            instruments[ik]["instrumental"] = str(f.resolve())
                            files[f"{ik}_instrumental"] = str(f.resolve())

        if "piano" in instruments:
            files["piano_instrumental"] = instruments["piano"].get("instrumental")
            files["piano_vocal"] = instruments["piano"].get("with_vocals")
        if "guitar_steel" in instruments:
            files["guitar_instrumental"] = instruments["guitar_steel"].get("instrumental")
            files["guitar_vocal"] = instruments["guitar_steel"].get("with_vocals")

        return {
            "id": entry_id,
            "song_title": entry_id,
            "created_at": "N/A",
            "mode": "Tiêu chuẩn",
            "transpose": "Tone gốc",
            "selected_instruments": list(instruments.keys()),
            "instruments": instruments,
            "files": files
        }

    def delete_history_entry(self, entry_id: str) -> bool:
        """
        Delete a single history entry folder from outputs/history/.
        """
        if not entry_id:
            return False
        target_dir = self.history_dir / entry_id
        if target_dir.exists() and target_dir.is_dir():
            shutil.rmtree(target_dir, ignore_errors=True)
            logger.info(f"Deleted history folder: {target_dir}")
            return True
        return False

    def clear_all_history(self) -> int:
        """
        Delete all history folders inside outputs/history/.
        """
        count = 0
        if self.history_dir.exists():
            for sub_dir in list(self.history_dir.iterdir()):
                if sub_dir.is_dir():
                    shutil.rmtree(sub_dir, ignore_errors=True)
                    count += 1
        logger.info(f"Cleared all history: {count} folders removed.")
        return count


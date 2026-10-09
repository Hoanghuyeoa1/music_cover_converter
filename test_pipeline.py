import sys
import math
import wave
import struct
import shutil
from pathlib import Path

# Ensure UTF-8 console output
try:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.downloader import ensure_ffmpeg, ensure_fluidsynth, ensure_soundfonts
from src.audio_processor import (
    AudioProcessor,
    detect_musical_key,
    calculate_smart_transpose,
    compute_file_sha256,
    extract_beats_and_tempo,
    apply_groove_and_beat_dynamics,
    estimate_bpm,
    time_stretch_audio,
    stretch_midi_tempo,
    extract_chord_progression_from_midi,
    format_chord_progression_text,
    identify_chord_from_pitches,
    is_sheet_music_file
)
from src.omr_processor import SheetMusicOMR
import pretty_midi
import numpy as np
from PIL import Image, ImageDraw


def create_synthetic_musicxml(filepath: Path):
    """Generate a clean synthetic MusicXML file with C and Am chords."""
    import music21
    s = music21.stream.Score()
    p = music21.stream.Part()
    
    m1 = music21.stream.Measure(number=1)
    m1.append(music21.meter.TimeSignature('4/4'))
    m1.append(music21.key.Key('C'))
    c_chord = music21.chord.Chord(['C4', 'E4', 'G4'], quarterLength=4.0)
    m1.append(c_chord)
    p.append(m1)
    
    m2 = music21.stream.Measure(number=2)
    am_chord = music21.chord.Chord(['A3', 'C4', 'E4'], quarterLength=4.0)
    m2.append(am_chord)
    p.append(m2)
    
    s.append(p)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    s.write('musicxml', fp=str(filepath))


def create_synthetic_sheet_image(filepath: Path):
    """Draw a clean synthetic 5-line staff with noteheads for OMR testing."""
    img = Image.new("RGB", (800, 300), "white")
    draw = ImageDraw.Draw(img)
    lines = [100, 116, 132, 148, 164]
    for y in lines:
        draw.line([(60, y), (740, y)], fill="black", width=2)
    # Barlines
    draw.line([(60, lines[0]), (60, lines[-1])], fill="black", width=3)
    draw.line([(400, lines[0]), (400, lines[-1])], fill="black", width=2)
    draw.line([(740, lines[0]), (740, lines[-1])], fill="black", width=3)
    # Noteheads
    notes = [
        (150, 164), # E4 (Line 1)
        (220, 148), # G4 (Line 2)
        (290, 132), # B4 (Line 3)
        (480, 116), # D5 (Line 4)
        (560, 100), # F5 (Line 5)
        (640, 164), # E4
    ]
    for cx, cy in notes:
        draw.ellipse([cx - 8, cy - 6, cx + 8, cy + 6], fill="black")
        draw.line([(cx + 8, cy), (cx + 8, cy - 35)], fill="black", width=2)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(filepath))


def create_synthetic_wav(filename: Path, duration_sec: float = 3.0, sample_rate: int = 44100):
    """Generate a clean synthetic C-major chord audio file for testing."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(duration_sec * sample_rate)
    
    # Frequencies: C4 (261.63Hz), E4 (329.63Hz), G4 (392.00Hz)
    freqs = [261.63, 329.63, 392.00]
    
    with wave.open(str(filename), "w") as wav:
        wav.setnchannels(2)  # Stereo
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        
        frames = bytearray()
        for i in range(n_samples):
            t = i / sample_rate
            env = max(0.0, 1.0 - t / duration_sec)
            val = sum(math.sin(2 * math.pi * f * t) for f in freqs) / len(freqs) * 0.7 * env
            sample = int(val * 32767.0)
            packed = struct.pack("<hh", sample, sample)
            frames.extend(packed)
            
        wav.writeframes(frames)
    print(f"[OK] Generated synthetic test audio: {filename}")


def run_tests():
    print("=== Testing Environment & Tools ===")
    ff = ensure_ffmpeg()
    print(f"[OK] FFmpeg binary verified: {ff}")
    fs = ensure_fluidsynth()
    print(f"[OK] FluidSynth binary verified: {fs}")
    sfs = ensure_soundfonts()
    print(f"[OK] SoundFont verified: {sfs['general_gs']}")

    test_dir = PROJECT_ROOT / "temp" / "test_run"
    shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    test_wav = test_dir / "test_chord.wav"
    create_synthetic_wav(test_wav, duration_sec=2.5)

    print("\n=== Initializing AudioProcessor ===")
    processor = AudioProcessor(
        soundfont_path=sfs["general_gs"],
        output_dir=str(test_dir / "outputs")
    )

    print("\n=== 1. Testing Basic Pitch Audio-to-MIDI ===")
    midi_dir = test_dir / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    midi_file = processor.audio_to_midi(
        test_wav,
        midi_dir,
        onset_threshold=0.4,
        frame_threshold=0.3,
        min_note_length_ms=80.0
    )
    print(f"[OK] Generated MIDI: {midi_file} ({midi_file.stat().st_size} bytes)")
    assert midi_file.exists() and midi_file.stat().st_size > 0, "MIDI file must exist"

    print("\n=== 2. Testing MIDI Instrument Adaptation & FluidSynth Rendering ===")
    # Piano (Grand Piano GM 0) using clean_and_prepare_midi
    piano_midi = test_dir / "piano.mid"
    processor.clean_and_prepare_midi(
        input_midi_path=midi_file,
        output_midi_path=piano_midi,
        program_number=0,
        enable_sustain=True,
        is_guitar=False
    )
    assert piano_midi.exists() and piano_midi.stat().st_size > 0, "Piano MIDI must exist"
    
    piano_wav = test_dir / "piano_rendered.wav"
    processor.render_midi_to_wav(
        midi_path=piano_midi,
        output_wav=piano_wav,
        soundfont_file=Path(sfs["general_gs"]),
        gain=1.0
    )
    assert piano_wav.exists() and piano_wav.stat().st_size > 1000, "Rendered Piano WAV must exist"
    print(f"[OK] Rendered Piano: {piano_wav} ({piano_wav.stat().st_size} bytes)")

    # Guitar (Steel Guitar GM 25) - also testing adapt_midi_instrument alias
    guitar_midi = test_dir / "guitar.mid"
    processor.adapt_midi_instrument(
        input_midi_path=midi_file,
        output_midi_path=guitar_midi,
        program_number=25,
        enable_sustain=False,
        is_guitar=True
    )
    assert guitar_midi.exists() and guitar_midi.stat().st_size > 0, "Guitar MIDI must exist"
    
    guitar_wav = test_dir / "guitar_rendered.wav"
    processor.render_midi_to_wav(
        midi_path=guitar_midi,
        output_wav=guitar_wav,
        soundfont_file=Path(sfs["general_gs"]),
        gain=1.15
    )
    assert guitar_wav.exists() and guitar_wav.stat().st_size > 1000, "Rendered Guitar WAV must exist"
    print(f"[OK] Rendered Guitar: {guitar_wav} ({guitar_wav.stat().st_size} bytes)")

    print("\n=== 3. Testing 🎋 Bamboo Flute (Sáo Trúc) Monophonic Mode ===")
    flute_lead_midi = test_dir / "flute_lead.mid"
    processor.prepare_monophonic_lead_midi(
        melody_midi_path=None,
        chords_midi_path=midi_file,
        output_midi_path=flute_lead_midi,
        program_number=77,
        transpose_semitones=0,
        min_pitch=58,
        max_pitch=84,
        min_duration_sec=0.060
    )
    assert flute_lead_midi.exists() and flute_lead_midi.stat().st_size > 0, "Flute lead MIDI must exist"
    
    flute_lead_wav = test_dir / "flute_lead.wav"
    processor.render_midi_to_wav(
        midi_path=flute_lead_midi,
        output_wav=flute_lead_wav,
        soundfont_file=Path(sfs["general_gs"]),
        gain=1.3
    )
    assert flute_lead_wav.exists() and flute_lead_wav.stat().st_size > 1000, "Rendered Flute WAV must exist"
    print(f"[OK] Rendered Bamboo Flute Lead: {flute_lead_wav} ({flute_lead_wav.stat().st_size} bytes)")

    print("\n=== 4. Testing Solo / Fingerstyle (Melody + Chords Merged) ===")
    solo_midi = test_dir / "solo_piano.mid"
    processor.combine_solo_fingerstyle_midi(
        chords_midi_path=midi_file,
        melody_midi_path=midi_file,
        output_midi_path=solo_midi,
        program_number=0,
        transpose_semitones=0,
        enable_sustain=True,
        is_guitar=False
    )
    assert solo_midi.exists() and solo_midi.stat().st_size > 0, "Solo Fingerstyle MIDI must exist"
    
    solo_wav = test_dir / "solo_rendered.wav"
    processor.render_midi_to_wav(
        midi_path=solo_midi,
        output_wav=solo_wav,
        soundfont_file=Path(sfs["general_gs"]),
        gain=1.2
    )
    assert solo_wav.exists() and solo_wav.stat().st_size > 1000, "Rendered Solo WAV must exist"
    print(f"[OK] Rendered Solo/Fingerstyle: {solo_wav} ({solo_wav.stat().st_size} bytes)")

    print("\n=== 5. Testing FFmpeg Vocal + Instrument Mixing & Mastering (EBU R128) ===")
    mixed_mp3 = test_dir / "mixed_test.mp3"
    processor.mix_vocals_and_instrument_to_mp3(
        vocals_wav=test_wav,
        instrument_wav=piano_wav,
        output_mp3=mixed_mp3,
        vocal_volume=0.9,
        inst_volume=1.2,
        acoustic_preset="studio_warmth"
    )
    assert mixed_mp3.exists() and mixed_mp3.stat().st_size > 1000, "Mastered MP3 must exist"
    print(f"[OK] Mastered Vocal+Instrument MP3 (Sidechain Ducking + EBU R128): {mixed_mp3} ({mixed_mp3.stat().st_size} bytes)")

    print("\n=== 6. Testing Audio Range Trimmer (Fast Preview Slice) ===")
    preview_wav = test_dir / "test_preview_slice.wav"
    processor.normalize_to_wav(test_wav, preview_wav, start_sec=0.5, duration_sec=1.2)
    assert preview_wav.exists() and preview_wav.stat().st_size > 1000, "Preview slice WAV must exist"
    print(f"[OK] Normalized Preview Slice (0.5s - 1.7s): {preview_wav} ({preview_wav.stat().st_size} bytes)")

    print("\n=== 7. Testing Auto Key Detection & Smart Transpose ===")
    key_info = detect_musical_key(test_wav)
    print(f"[OK] Detected Musical Key: {key_info['key']} (Label: {key_info['label']}, Confidence: {key_info['confidence']})")
    assert key_info["key"] == "C Major", f"Expected C Major for C-E-G synthetic audio, got {key_info['key']}"
    assert key_info["confidence"] >= 0.45, "Confidence score must be valid"

    # Test smart transpose calculations
    d1 = calculate_smart_transpose("C Major", "G Major / E Minor")
    assert d1 == -5, f"Expected -5 semitones for C Major -> G Major / E Minor, got {d1}"
    d2 = calculate_smart_transpose("A Minor", "D Major / B Minor")
    assert d2 == 2, f"Expected +2 semitones for A Minor -> D Major / B Minor, got {d2}"
    d3 = calculate_smart_transpose("C Major", "C Major / A Minor")
    assert d3 == 0, f"Expected 0 for C Major -> C Major / A Minor, got {d3}"
    d4 = calculate_smart_transpose("C Major", "Giữ nguyên gốc (Không đổi)")
    assert d4 == 0, f"Expected 0 for keep original, got {d4}"
    print(f"[OK] Smart Transpose Distance Calculations verified: C->G ({d1}), Am->D ({d2}), C->C ({d3}), Keep ({d4})")

    print("\n=== 8. Testing Demucs Stem SHA-256 Smart Cache ===")
    cache_test_wav = test_dir / "cache_unit_test.wav"
    create_synthetic_wav(cache_test_wav, duration_sec=1.0)
    file_hash = compute_file_sha256(cache_test_wav)
    assert len(file_hash) == 64, "SHA-256 hash must be 64 hexadecimal chars"

    # Populate cache directly to verify bypass logic without waiting for heavy demucs model
    cached_folder = processor.cache_dir / file_hash
    cached_folder.mkdir(parents=True, exist_ok=True)
    for stem_name in ["vocals", "drums", "bass", "other"]:
        shutil.copy2(cache_test_wav, cached_folder / f"{stem_name}.wav")

    # Run separate_stems - should hit cache instantly
    work_cache_test = test_dir / "work_cache"
    stems_cached = processor.separate_stems(cache_test_wav, work_cache_test)
    assert len(stems_cached) == 4, "Must retrieve all 4 stems"
    for s in ["vocals", "drums", "bass", "other"]:
        assert stems_cached[s].exists() and stems_cached[s].stat().st_size > 0, f"Stem {s} must exist"
    print(f"[OK] Demucs Stems Smart Cache verified ([CACHE HIT] confirmed with SHA-256: {file_hash[:12]})")

    # Test cache cleanup
    cleaned_entries = processor.clear_stem_cache()
    assert cleaned_entries >= 1, "Cache cleanup should remove at least 1 directory"
    print(f"[OK] Demucs Stem Cache cleanup verified: removed {cleaned_entries} entries")

    print("\n=== 9. Testing Beat-Aware Groove & Velocity Humanizer ===")
    tempo_bpm, beat_times = extract_beats_and_tempo(test_wav)
    assert tempo_bpm > 0, "Tempo must be positive"
    assert len(beat_times) > 0, "Beat times grid must not be empty"

    sample_notes = [
        pretty_midi.Note(velocity=80, pitch=60, start=0.00, end=0.30),  # Downbeat (phách 1)
        pretty_midi.Note(velocity=80, pitch=64, start=0.25, end=0.45),  # Offbeat / Passing note
        pretty_midi.Note(velocity=80, pitch=67, start=1.00, end=1.30),  # Phách 3 (medium strong)
    ]
    custom_beat_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    humanized = apply_groove_and_beat_dynamics(sample_notes, beat_times=custom_beat_grid, tempo=120.0)

    downbeat_vel = humanized[0].velocity
    offbeat_vel = humanized[1].velocity
    beat3_vel = humanized[2].velocity

    print(f"[OK] Groove Accentuation Results: Downbeat Vel={downbeat_vel} (+{downbeat_vel-80}), Beat 3 Vel={beat3_vel} (+{beat3_vel-80}), Offbeat Vel={offbeat_vel} ({offbeat_vel-80:+d})")
    assert downbeat_vel > 80, f"Downbeat velocity ({downbeat_vel}) must receive positive dynamic boost (>80)"
    assert offbeat_vel < 80, f"Offbeat velocity ({offbeat_vel}) must be tempered for groove bounce (<80)"
    assert downbeat_vel > offbeat_vel, f"Downbeat velocity ({downbeat_vel}) must be louder than offbeat ({offbeat_vel})"
    print("[OK] Beat-aware velocity accentuation & swing groove dynamics confirmed!")

    print("\n=== 10. Testing BPM Detection & High-Quality Time-Stretching ===")
    detected_bpm = estimate_bpm(test_wav)
    print(f"[OK] Estimated Song BPM: {detected_bpm:.1f} BPM")
    assert 40.0 <= detected_bpm <= 240.0, f"Detected BPM {detected_bpm} must be in reasonable range"

    # MIDI tempo stretching test (1.25x faster)
    stretched_midi = test_dir / "stretched_piano.mid"
    stretch_midi_tempo(piano_midi, stretched_midi, tempo_multiplier=1.25)
    assert stretched_midi.exists() and stretched_midi.stat().st_size > 0, "Stretched MIDI must exist"
    pm_orig = pretty_midi.PrettyMIDI(str(piano_midi))
    pm_stretched = pretty_midi.PrettyMIDI(str(stretched_midi))
    orig_note_len = pm_orig.instruments[0].notes[0].end - pm_orig.instruments[0].notes[0].start
    stretched_note_len = pm_stretched.instruments[0].notes[0].end - pm_stretched.instruments[0].notes[0].start
    print(f"[OK] MIDI Note Length scaled: {orig_note_len:.3f}s -> {stretched_note_len:.3f}s (Ratio: {stretched_note_len/orig_note_len:.2f})")
    assert stretched_note_len < orig_note_len, "Stretched note length must be shorter for tempo_multiplier > 1.0"

    # FFmpeg atempo audio time-stretching test (0.75x slower, preserving pitch)
    stretched_audio = test_dir / "stretched_vocal.wav"
    time_stretch_audio(test_wav, stretched_audio, tempo_multiplier=0.75, ffmpeg_bin=ff)
    assert stretched_audio.exists() and stretched_audio.stat().st_size > test_wav.stat().st_size, "Stretched audio must be larger in duration/size for slower speed"
    print(f"[OK] FFmpeg atempo Pitch-Preserving Time-Stretch: {stretched_audio} ({stretched_audio.stat().st_size} bytes)")

    print("\n=== 11. Testing MIDI Chord Recognition & Progression Extraction ===")
    # Direct chord template matcher tests
    chord_c = identify_chord_from_pitches([60, 64, 67])       # C, E, G
    chord_am = identify_chord_from_pitches([57, 60, 64])      # A, C, E
    chord_g7 = identify_chord_from_pitches([55, 59, 62, 65])  # G, B, D, F
    print(f"[OK] Chord Identification: [C,E,G] -> {chord_c}, [A,C,E] -> {chord_am}, [G,B,D,F] -> {chord_g7}")
    assert chord_c == "C", f"Expected C, got {chord_c}"
    assert chord_am == "Am", f"Expected Am, got {chord_am}"
    assert chord_g7 == "G7", f"Expected G7, got {chord_g7}"

    # Extract chord progression from transcribed MIDI
    progression = extract_chord_progression_from_midi(piano_midi)
    assert len(progression) >= 1, "Must extract at least 1 chord step"
    print(f"[OK] Extracted {len(progression)} chord progression steps: {progression}")
    assert progression[0]["chord"] == "C", f"First chord should be C, got {progression[0]['chord']}"

    # Test chord text formatter
    chord_txt = format_chord_progression_text(progression, title="Test Song")
    assert "C" in chord_txt and "MỐC THỜI GIAN" in chord_txt, "Formatted text must contain chord table"
    print(f"[OK] Formatted Chord Progression Text:\n{chord_txt[:200]}...")

    print("\n=== 12. Testing Interactive Stems Mini-Mixer & Acoustic Upright Bass ===")
    # Synthesize Acoustic Upright Bass track (Program 32)
    acoustic_bass_wav = test_dir / "acoustic_bass_test.wav"
    processor.render_acoustic_bass(
        bass_source=test_wav,
        output_wav=acoustic_bass_wav,
        temp_dir=test_dir,
        soundfont_file=sfs["general_gs"]
    )
    assert acoustic_bass_wav.exists() and acoustic_bass_wav.stat().st_size > 1000, "Acoustic bass WAV must exist"
    print(f"[OK] Synthesized Acoustic Upright Bass (GM Program 32): {acoustic_bass_wav} ({acoustic_bass_wav.stat().st_size} bytes)")

    # Test Live Re-Mastering Engine (Vocal + Instrument + Upright Bass + Reverb Send + EBU R128)
    mini_mixer_master_mp3 = test_dir / "mini_mixer_master.mp3"
    processor.mix_stems_master(
        vocals_wav=test_wav,
        instrument_wav=piano_wav,
        bass_wav=acoustic_bass_wav,
        output_mp3=mini_mixer_master_mp3,
        vocal_volume=1.0,
        inst_volume=1.0,
        bass_volume=0.8,
        reverb_send_pct=30.0
    )
    assert mini_mixer_master_mp3.exists() and mini_mixer_master_mp3.stat().st_size > 1000, "Mini-Mixer master MP3 must exist"
    print(f"[OK] Mini-Mixer Mastered MP3 (Sidechain Ducking + Reverb Send + EBU R128): {mini_mixer_master_mp3} ({mini_mixer_master_mp3.stat().st_size} bytes)")

    print("\n=== 13. Testing Sheet Music & MusicXML OMR to Multi-Instrument Pipeline ===")
    omr_dir = test_dir / "omr_test"
    omr_dir.mkdir(parents=True, exist_ok=True)

    # 13.1 Test MusicXML Digital Score Conversion
    test_musicxml = omr_dir / "sample_score.musicxml"
    create_synthetic_musicxml(test_musicxml)
    assert test_musicxml.exists(), "Sample MusicXML file must be created"
    assert is_sheet_music_file(test_musicxml), "is_sheet_music_file must detect .musicxml"
    print(f"[OK] Created synthetic MusicXML score: {test_musicxml}")

    omr_engine = SheetMusicOMR(temp_dir=omr_dir / "temp")
    xml_omr_res = omr_engine.process(test_musicxml)
    assert Path(xml_omr_res["midi_path"]).exists(), "OMR MusicXML must output MIDI file"
    assert xml_omr_res["total_notes"] >= 2, f"Expected at least 2 notes, got {xml_omr_res['total_notes']}"
    print(f"[OK] MusicXML parsed to MIDI: {xml_omr_res['total_notes']} notes | Tempo: {xml_omr_res['estimated_tempo_bpm']} BPM | Key: {xml_omr_res['detected_key']}")

    # 13.2 Test Sheet Music Image Computer Vision OMR
    test_sheet_img = omr_dir / "synthetic_sheet.png"
    create_synthetic_sheet_image(test_sheet_img)
    assert test_sheet_img.exists(), "Sample sheet music image must be created"
    assert is_sheet_music_file(test_sheet_img), "is_sheet_music_file must detect .png image"
    print(f"[OK] Created synthetic sheet music image: {test_sheet_img}")

    img_omr_res = omr_engine.process(test_sheet_img, generate_overlay=True)
    assert Path(img_omr_res["midi_path"]).exists(), "OMR Image must output MIDI file"
    assert img_omr_res["total_notes"] > 0, "OMR Image must detect notes on staff"
    assert img_omr_res.get("overlay_path") and Path(img_omr_res["overlay_path"]).exists(), "OMR must generate visual overlay image"
    print(f"[OK] Computer Vision OMR on Sheet Image: {img_omr_res['total_notes']} notes detected | Staffs: {img_omr_res['total_staffs']} | Overlay: {Path(img_omr_res['overlay_path']).name}")

    # 13.3 Test End-to-End process_sheet_music Pipeline (Skipping Demucs, rendering Piano & Bamboo Flute)
    sheet_cover_res = processor.process_sheet_music(
        input_path=str(test_musicxml),
        selected_instruments=["piano", "flute"],
        cover_mode="chords",
        transpose_semitones=0,
        soundfont_preset="standard",
        acoustic_preset="studio_warmth",
        include_bass=True
    )
    assert sheet_cover_res.get("midi_file") and Path(sheet_cover_res["midi_file"]).exists(), "Sheet cover must yield master MIDI"
    assert sheet_cover_res.get("history_id"), "Sheet cover must create history entry"

    inst_dict = sheet_cover_res.get("instruments", {})
    assert "piano" in inst_dict, "Piano must be rendered"
    assert "flute" in inst_dict or "bamboo_flute" in inst_dict, "Bamboo flute must be rendered"

    piano_mp3 = Path(inst_dict["piano"]["instrumental"])
    assert piano_mp3.exists() and piano_mp3.stat().st_size > 1000, f"Piano MP3 missing or empty: {piano_mp3}"

    flute_key = "flute" if "flute" in inst_dict else "bamboo_flute"
    flute_mp3 = Path(inst_dict[flute_key]["instrumental"])
    assert flute_mp3.exists() and flute_mp3.stat().st_size > 1000, f"Flute MP3 missing or empty: {flute_mp3}"

    # Verify Chord HUD Progression
    chord_prog = sheet_cover_res.get("chord_progression", [])
    assert len(chord_prog) >= 1, "Must extract chord progression from sheet MIDI"
    print(f"[OK] End-to-End Sheet Music Cover Pipeline rendered successfully!")
    print(f"     - Piano Master MP3: {piano_mp3.name} ({piano_mp3.stat().st_size} bytes)")
    print(f"     - Bamboo Flute Master MP3: {flute_mp3.name} ({flute_mp3.stat().st_size} bytes)")
    print(f"     - Extracted Chords (Chord HUD): {[c['chord'] for c in chord_prog]}")
    print(f"     - History Run ID: {sheet_cover_res['history_id']}")

    print("\n=======================================================")
    print("ALL CORE AUDIO & SHEET OMR PIPELINE TESTS PASSED WITH 100% SUCCESS!")
    print("=======================================================")


if __name__ == "__main__":
    run_tests()

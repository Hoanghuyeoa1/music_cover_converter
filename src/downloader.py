import os
import sys
import shutil
import zipfile
import logging
from pathlib import Path
from typing import Tuple, Optional
import urllib.request
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Downloader")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = PROJECT_ROOT / "bin"
SOUNDFONTS_DIR = PROJECT_ROOT / "soundfonts"

FLUIDSYNTH_WINDOWS_URL = "https://github.com/FluidSynth/fluidsynth/releases/download/v2.6.1/fluidsynth-v2.6.1-win10-x64-cpp11.zip"

SOUNDFONT_SOURCES = [
    {
        "id": "general_gs",
        "name": "GeneralUser-GS.sf2",
        "url": "https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/master/GeneralUser-GS.sf2",
        "fallback_url": "https://archive.org/download/free-soundfonts-sf2-2019-04/GeneralUser%20GS%20v1.471.sf2",
        "description": "GeneralUser GS SoundFont (Standard General MIDI bank)"
    },
    {
        "id": "studio_piano",
        "name": "Nice-Steinway-Lite-v3.0.sf2",
        "url": "https://huggingface.co/datasets/projectlosangeles/soundfonts4u/resolve/main/Nice-Steinway-Lite-v3.0.sf2",
        "fallback_url": "https://huggingface.co/datasets/projectlosangeles/soundfonts4u/resolve/main/Abbey-Steinway-D-v1.9.sf2",
        "description": "Nice Steinway Lite Concert Grand Piano (74MB High-Fidelity Studio SoundFont)"
    },
    {
        "id": "studio_guitar",
        "name": "Guitars-Universal-V1.5.sf2",
        "url": "https://huggingface.co/datasets/projectlosangeles/soundfonts4u/resolve/main/Guitars-Universal-V1.5.sf2",
        "fallback_url": "https://huggingface.co/datasets/projectlosangeles/soundfonts4u/resolve/main/Best%20of%20Guitars-4U-v1.0.sf2",
        "description": "Guitars Universal Studio SoundFont (157MB Multi-Velocity Acoustic Steel & Nylon Guitars)"
    }
]


def download_file_with_progress(url: str, dest_path: Path, description: str = ""):
    """Download a file with console progress indication."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {description or dest_path.name} from {url}...")
    
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
            total_size = response.info().get("Content-Length")
            total_size = int(total_size) if total_size else None
            downloaded = 0
            block_size = 1024 * 64

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size:
                    percent = int(downloaded * 100 / total_size)
                    mb = downloaded / (1024 * 1024)
                    tot_mb = total_size / (1024 * 1024)
                    print(f"\r  Progress: {percent}% ({mb:.1f}/{tot_mb:.1f} MB)", end="", flush=True)
                else:
                    mb = downloaded / (1024 * 1024)
                    print(f"\r  Downloaded: {mb:.1f} MB", end="", flush=True)
            print()
        logger.info(f"Successfully downloaded to {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def ensure_ffmpeg() -> str:
    """
    Ensure ffmpeg binary is accessible.
    Checks PATH, local bin, and winget packages, updating PATH if found.
    Returns the path to ffmpeg executable or raises FileNotFoundError.
    """
    # 1. Check system PATH
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        logger.info(f"FFmpeg found in PATH: {ffmpeg_path}")
        return ffmpeg_path

    # 2. Check project bin directory
    local_ffmpeg = BIN_DIR / "ffmpeg.exe"
    if local_ffmpeg.exists():
        os.environ["PATH"] = str(BIN_DIR) + os.path.pathsep + os.environ.get("PATH", "")
        logger.info(f"FFmpeg found in project bin: {local_ffmpeg}")
        return str(local_ffmpeg)

    # 3. Check Winget installation directories on Windows
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        winget_pattern = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_pattern.exists():
            for f in winget_pattern.glob("**/ffmpeg.exe"):
                ffmpeg_bin_dir = str(f.parent)
                os.environ["PATH"] = ffmpeg_bin_dir + os.path.pathsep + os.environ.get("PATH", "")
                logger.info(f"FFmpeg located in WinGet package: {f}. Added to PATH.")
                return str(f)

    # 4. Check typical CapCut or other local paths if present
    possible_paths = [
        Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
    ]
    for p in possible_paths:
        if p.exists():
            os.environ["PATH"] = str(p.parent) + os.path.pathsep + os.environ.get("PATH", "")
            logger.info(f"FFmpeg found at {p}. Added to PATH.")
            return str(p)

    raise FileNotFoundError(
        "FFmpeg is not installed or not in PATH.\n"
        "Please install it using:\n"
        "  Windows: winget install Gyan.FFmpeg\n"
        "  macOS:   brew install ffmpeg\n"
        "  Linux:   sudo apt install ffmpeg"
    )


def ensure_fluidsynth() -> str:
    """
    Ensure fluidsynth binary is accessible.
    If not in PATH and on Windows, downloads and extracts the official release.
    Returns path to fluidsynth executable.
    """
    # 1. Check system PATH
    fluidsynth_path = shutil.which("fluidsynth")
    if fluidsynth_path:
        logger.info(f"FluidSynth found in PATH: {fluidsynth_path}")
        return fluidsynth_path

    # 2. Check project bin directory
    fs_subdirs = list(BIN_DIR.glob("**/fluidsynth.exe"))
    if fs_subdirs:
        fs_exe = fs_subdirs[0]
        os.environ["PATH"] = str(fs_exe.parent) + os.path.pathsep + os.environ.get("PATH", "")
        logger.info(f"FluidSynth found in {fs_exe}. Added to PATH.")
        return str(fs_exe)

    # 3. If on Windows, auto-download fluidsynth binary
    if sys.platform.startswith("win"):
        logger.info("FluidSynth not found. Automatically downloading Windows x64 binary...")
        BIN_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = BIN_DIR / "fluidsynth.zip"
        dest_dir = BIN_DIR / "fluidsynth"

        success = download_file_with_progress(
            FLUIDSYNTH_WINDOWS_URL,
            zip_path,
            description="FluidSynth Windows x64 release"
        )
        if success:
            logger.info("Extracting FluidSynth archive...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
            zip_path.unlink(missing_ok=True)

            # Re-locate fluidsynth.exe
            found = list(dest_dir.glob("**/fluidsynth.exe"))
            if found:
                fs_exe = found[0]
                os.environ["PATH"] = str(fs_exe.parent) + os.path.pathsep + os.environ.get("PATH", "")
                logger.info(f"FluidSynth successfully setup at: {fs_exe}")
                return str(fs_exe)

    raise FileNotFoundError(
        "FluidSynth is not found and could not be configured automatically.\n"
        "Please install FluidSynth:\n"
        "  Windows: Download from https://github.com/FluidSynth/fluidsynth/releases and place in bin/fluidsynth\n"
        "  macOS:   brew install fluidsynth\n"
        "  Linux:   sudo apt install fluidsynth"
    )


def ensure_soundfonts() -> dict:
    """
    Ensure required SoundFont files are downloaded.
    Returns dict mapping soundfont identifiers to absolute file paths.
    """
    SOUNDFONTS_DIR.mkdir(parents=True, exist_ok=True)
    soundfonts = {}

    for sf in SOUNDFONT_SOURCES:
        sf_path = SOUNDFONTS_DIR / sf["name"]
        if not sf_path.exists() or sf_path.stat().st_size < 1000000:  # Valid sf2 is > 1MB
            logger.info(f"SoundFont {sf['name']} missing or incomplete. Downloading...")
            # Try primary URL
            success = download_file_with_progress(sf["url"], sf_path, sf["description"])
            # Fallback if primary fails
            if not success and "fallback_url" in sf:
                logger.info("Attempting fallback URL...")
                success = download_file_with_progress(sf["fallback_url"], sf_path, sf["description"])

            if not success or not sf_path.exists():
                raise RuntimeError(f"Could not download SoundFont: {sf['name']}")
        else:
            logger.info(f"SoundFont verified: {sf_path} ({sf_path.stat().st_size / (1024*1024):.1f} MB)")

        soundfonts[sf.get("id", sf["name"])] = str(sf_path)

    return soundfonts


def download_audio_from_url(
    url: str,
    output_dir: Optional[Path] = None,
    preferred_codec: str = "wav"
) -> Tuple[Path, str]:
    """
    Download highest quality audio from a supported URL (YouTube, TikTok, SoundCloud, etc.)
    using yt-dlp and extract to a clean audio file.
    Returns (downloaded_file_path, extracted_title).
    """
    if not url or not url.strip():
        raise ValueError("URL không được để trống.")
    
    clean_url = url.strip()
    if output_dir is None:
        output_dir = PROJECT_ROOT / "temp" / "downloads"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import yt_dlp
    except ImportError:
        raise ImportError(
            "Thư viện yt-dlp chưa được cài đặt. "
            "Vui lòng cài đặt bằng lệnh: uv pip install yt-dlp hoặc pip install yt-dlp"
        )

    ffmpeg_bin = ensure_ffmpeg()
    ffmpeg_dir = str(Path(ffmpeg_bin).parent)

    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_tmpl = str(output_dir / f"{time_tag}_%(title).60s.%(ext)s")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_tmpl,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': preferred_codec,
            'preferredquality': '192',
        }],
        'ffmpeg_location': ffmpeg_dir,
        'noplaylist': True,
        'quiet': False,
        'no_warnings': True,
    }

    logger.info(f"Downloading audio from URL: {clean_url}...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(clean_url, download=True)
        raw_title = info.get('title', 'URL_Track')
        
        # yt-dlp replaces extension with preferred_codec
        prepared_name = ydl.prepare_filename(info)
        expected_path = Path(prepared_name).with_suffix(f".{preferred_codec}")
        
        if expected_path.exists():
            logger.info(f"Audio downloaded and extracted: {expected_path}")
            return expected_path, raw_title
        
        # Fallback check for newly created files in output_dir
        recent_files = sorted(output_dir.glob(f"{time_tag}*.{preferred_codec}"), key=os.path.getmtime, reverse=True)
        if recent_files:
            return recent_files[0], raw_title
            
        raise FileNotFoundError(f"Không tìm thấy file audio sau khi tải từ URL: {clean_url}")


if __name__ == "__main__":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=== Checking & Downloading Required Tools & Assets ===")
    ff = ensure_ffmpeg()
    print(f"[OK] FFmpeg: {ff}")
    fs = ensure_fluidsynth()
    print(f"[OK] FluidSynth: {fs}")
    sfs = ensure_soundfonts()
    for k, v in sfs.items():
        print(f"[OK] SoundFont [{k}]: {v}")
    print("\nAll prerequisites are ready!")


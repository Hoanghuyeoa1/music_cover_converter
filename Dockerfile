# ====================================================================
# Music Cover Studio - Dockerfile (NVIDIA CUDA 12.1 + PyTorch + Demucs)
# ====================================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV SERVER_NAME=0.0.0.0
ENV PORT=7860

# Install system dependencies (Python 3.11 from deadsnakes, ffmpeg, fluidsynth, git, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    ffmpeg \
    fluidsynth \
    fluid-soundfont-gm \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3.11-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python & python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set working directory
WORKDIR /app

# Copy dependency specifications first for layer caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 wheels, then remaining python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code and SoundFonts into image
COPY src/ /app/src/
COPY app.py /app/app.py
COPY test_pipeline.py /app/test_pipeline.py
COPY soundfonts/ /app/soundfonts/

# Ensure soundfonts exist or download fallbacks, and initialize necessary directories
RUN python3 -c "from src.downloader import ensure_soundfonts; ensure_soundfonts()" \
    && mkdir -p /app/outputs/history /app/temp/cache_stems

# Expose Gradio web server port
EXPOSE 7860

# Healthcheck for Gradio service
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Launch Gradio server
ENTRYPOINT ["python3", "app.py"]

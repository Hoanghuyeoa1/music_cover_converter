# 🎹 AI Multi-Instrument Cover Converter & Studio 🎸🎻🎷🎋

Hệ thống AI cấp cao chuyển đổi bài hát bất kỳ (MP3, WAV, FLAC, MP4, M4A, YouTube URL) thành các bản Cover nhạc cụ mộc tự nhiên (**Piano Grand**, **Acoustic Guitars**, **Sáo Trúc Hòa Tấu**, **Dàn Dây Strings**, **Alto Sax**, **Electric Piano Rhodes**, **Harp**, **Music Box**) và **Acoustic Upright Bass**, tích hợp nhận diện nhịp BPM, co giãn tốc độ giữ nguyên cao độ, bộ nhận diện hợp âm thời gian thực và bàn trộn Stems Mini-Mixer chuẩn mastering EBU R128.

---

## 🌟 Tính Năng Đột Phá Nổi Bật

### 1. ⚡ Nhận Diện Nhịp & Co Giãn Tốc Độ (BPM Detector & High-Quality Time-Stretching)
- **Tự động nhận diện BPM**: Ứng dụng `librosa.beat.beat_track` phân tích xung nhịp bài hát gốc hiển thị trực quan lên giao diện.
- **Thanh trượt Tempo Multiplier (0.6x - 1.5x)**: Co giãn trục thời gian của toàn bộ nốt MIDI trước khi nạp vào FluidSynth render.
- **Bảo toàn cao độ giọng hát (Pitch-Preserving Vocal Stretch)**: Tự động ghép chuỗi bộ lọc FFmpeg `atempo` đa tầng trên `vocals.wav`, giúp giọng hát khớp chính xác với nhịp mới của dàn nhạc cụ mà hoàn toàn không bị biến dạng tone/cao độ.

### 2. 🎛️ Trình Trộn Âm Tương Tác Trực Quan (Interactive Stems Mini-Mixer)
- **Acoustic Upright Bass (GM Program 32)**: Tách stem `bass.wav` qua Demucs, bóc tách cao độ bằng Basic Pitch và render tiếng Bass mộc dày dặn, ấm áp bằng FluidSynth.
- **Bàn Mixer 4 Kênh Trực Tiếp**:
  - `🎤 Vocal Volume` (0.0 - 1.5)
  - `🎹 Cover Instrument Volume` (0.0 - 1.5)
  - `🎻 Acoustic Bass Volume` (0.0 - 1.5)
  - `🏛️ Master Reverb Send` (0% - 100%)
- **Live Re-Mastering Engine**: FFmpeg Filtergraph thời gian thực kết hợp EQ Carving, Sidechain Ducking giữa Vocal & Instrument, Reverb tổng thể và xuất ra file MP3 320kbps chuẩn quốc tế **EBU R128 (-14 LUFS, True Peak -1.0 dBFS)** khi bấm **"🎧 Mix & Xuất Bản Phối Này"**.

### 3. 🎸 Bộ Nhận Diện Hợp Âm & Hiển Thị Thời Gian Thực (Real-Time Chord HUD)
- **Thuật toán gom cụm & khớp mẫu hợp âm**: Phân tích cao độ theo phách từ đối tượng PrettyMIDI, nhận diện đầy đủ: Major triad, Minor triad, Dominant 7th, Minor 7th, Major 7th, Sus2, Sus4, Diminished, Augmented, Power Chord...
- **Real-time Active Chord Glow**: Tích hợp JavaScript lắng nghe sự kiện `timeupdate` từ Trình phát Audio để **highlight phát sáng rực rỡ hợp âm đang vang lên** theo thời gian thực.
- **Click-to-Seek**: Nhấp chuột vào bất kỳ ô hợp âm nào trên HUD để tua ngay trình phát đến đúng thời điểm đó.
- **Xuất bảng hợp âm**: Tải file `.txt` danh sách vòng hợp âm theo ô nhịp để in ra hoặc lưu trữ luyện tập.

### 4. 🎼 Chuyển Đổi Sheet Nhạc Sang Bản Phối Đa Nhạc Cụ (Sheet Music OMR & MusicXML)
- **Đa dạng nguồn đầu vào ký âm**: Hỗ trợ trực tiếp tệp ký âm số hóa **MusicXML (`.musicxml`, `.mxl`, `.xml`)**, tệp **MIDI (`.mid`)** hoặc ảnh chụp bản nhạc thực tế **(`.png`, `.jpg`, `.jpeg`, `.pdf`)**.
- **Công nghệ OMR cục bộ (Offline Optical Music Recognition)**: Thuật toán thị giác máy tính phân tích biểu đồ chiếu ngang (Horizontal Projection Profile) để định vị 5 dòng kẻ khuông nhạc, phân loại khóa Sol (Treble) / khóa Fa (Bass), bóc tách đầu nốt (connected components), kiểm tra đuôi nốt (stem analysis) và phân chia ô nhịp.
- **Tối ưu tốc độ vượt trội (Bypass Demucs)**: Tự động bỏ qua các bước tách stem tốn kém tài nguyên, đưa thẳng nốt nhạc số hóa vào bộ engine nắn nốt humanize, sinh Smart Sustain (CC 64), gán độ trễ tỉa phím Guitar Strumming, render qua 9 bộ nhạc cụ phòng thu SoundFonts và chuẩn hóa EBU R128 (-14 LUFS).
- **Xem trước tương tác (Interactive Sheet Preview)**: Trích xuất tự động trang 1 của PDF hoặc hiển thị trực quan bản chụp sheet nhạc ngay trên giao diện trước khi render.

### 5. 🚀 Đóng Gói Docker 1-Click (Cloud & GPU Serverless Ready)
- Hỗ trợ triển khai container hóa toàn diện với **NVIDIA CUDA 12.1**, **Ubuntu 22.04**, **Python 3.11**, **FluidSynth**, và **FFmpeg**.
- Khởi chạy ngay lập tức chỉ với 1 dòng lệnh duy nhất!

---

## 🐳 Hướng Dẫn Khởi Chạy Nhanh Bằng Docker 1-Click

### Yêu Cầu
- Đã cài đặt [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/).
- (Khuyên dùng) Máy tính hoặc VPS có GPU NVIDIA và đã cài [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### Khởi Chạy (1 Command)
```bash
docker compose up -d
```
Docker sẽ tự động:
1. Kéo image nền `nvidia/cuda:12.1.1-runtime-ubuntu22.04`.
2. Cài đặt Python 3.11, FFmpeg, FluidSynth và các thư viện trong `requirements.txt`.
3. Chuẩn bị các bộ SoundFonts chất lượng cao.
4. Mount thư mục kết quả `./outputs` và bộ nhớ đệm `./temp/cache_stems` vào máy chủ host (dữ liệu không bị mất khi restart).
5. Phơi cổng `7860`.

Truy cập giao diện Web Studio tại:
👉 **http://localhost:7860**

### Dừng Container
```bash
docker compose down
```

---

## 💻 Hướng Dẫn Chạy Cục Bộ (Local Machine)

### Cách 1: Chạy bằng file `run.bat` (Khuyên dùng trên Windows)
Nhấp đúp chuột vào file:
```cmd
run.bat
```

### Cách 2: Khởi chạy bằng Virtual Environment
```bash
# 1. Kích hoạt môi trường ảo
.venv\Scripts\activate      # Trên Windows
# source .venv/bin/activate # Trên Linux / macOS

# 2. Cài đặt thư viện (nếu chưa cài)
pip install -r requirements.txt

# 3. Khởi chạy giao diện Gradio Studio
python app.py
```
Mở trình duyệt tại: 👉 **http://127.0.0.1:7860**

---

## 🧪 Chạy Bộ Kiểm Thử Tự Động (Automated Test Suite)

Dự án tích hợp bộ kiểm thử 13 bài test toàn diện kiểm tra từ xử lý tín hiệu đến các thuật toán AI:
```bash
python test_pipeline.py
```
**Danh sách 13 bài test:**
1. Spotify Basic Pitch Audio-to-MIDI polyphonic transcription.
2. MIDI Instrument Adaptation & FluidSynth multi-soundfont rendering.
3. Bamboo Flute (Sáo Trúc) Monophonic lead extraction.
4. Solo / Fingerstyle Melody + Chords dynamic merging.
5. FFmpeg Vocal + Instrument Mastering (Sidechain Ducking + EBU R128).
6. Audio Range Trimmer (Fast Preview Slice).
7. Auto Key Detection & Smart Transpose distance calculation.
8. Demucs Stem SHA-256 Smart Cache (Zero-redundancy stem separation).
9. Beat-Aware Groove & Velocity Accentuation Humanizer.
10. **BPM Detection & High-Quality Pitch-Preserving Time-Stretching**.
11. **MIDI Chord Recognition & Real-time HUD Progression Exporter**.
12. **Interactive Stems Mini-Mixer & Acoustic Upright Bass (GM 32) Mastering**.
13. **Sheet Music & MusicXML OMR to Multi-Instrument Acoustic Cover Pipeline**.

---

## 📂 Cấu Trúc Mã Nguồn

```
music_cover_converter/
├── Dockerfile                 # Docker build NVIDIA CUDA 12.1 + Python 3.11 + FluidSynth
├── docker-compose.yml         # Cấu hình GPU reservation & volume persistence
├── .dockerignore              # Danh sách loại trừ khi build container
├── app.py                     # Giao diện Web Gradio Studio (Real-time HUD, Mini-Mixer)
├── test_pipeline.py           # Bộ test tự động 13 hạng mục (100% Passed)
├── requirements.txt           # Danh mục thư viện Python
├── run.bat                    # Script chạy 1-click Windows
├── soundfonts/                # Thư mục chứa các bộ SoundFonts (.sf2)
│   ├── GeneralUser-GS.sf2
│   ├── Nice-Steinway-Lite-v3.0.sf2
│   └── Guitars-Universal-V1.5.sf2
├── src/
│   ├── __init__.py
│   ├── downloader.py          # Quản lý tự động tải binary & SoundFonts
│   ├── omr_processor.py       # Engine nhận diện ký âm OMR & phân tích MusicXML sang MIDI
│   └── audio_processor.py     # Lõi xử lý âm thanh, OMR routing, Demucs cache, Basic Pitch, Chord HUD, Mixer
├── outputs/                   # Kết quả xuất ra (MP3, WAV, MIDI, Text Chord Sheet)
│   └── history/               # Lịch sử chi tiết từng bản thu
└── temp/
    └── cache_stems/           # Demucs Stem Smart Cache SHA-256
```

---

## 📜 Giấy Phép & Tín Chỉ (License & Credits)
- **Demucs**: Facebook / Meta Research (HTDemucs Hybrid Transformer).
- **Basic Pitch**: Spotify Audio Intelligence Lab.
- **SoundFonts**: GeneralUser GS (S. Christian Collins), Nice Steinway Lite & Guitars Universal (Soundfonts4U).
- **Audio Processing**: FFmpeg Project & FluidSynth Team.

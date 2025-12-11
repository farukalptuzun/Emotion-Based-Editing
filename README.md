# 🎬 Emotion-Based Editing

**Amaç:** Videonun duygu ve enerjisine göre otomatik görsel efekt uygulamak

**Hedef Etki:** İzleyici dikkatinin videodan kopmasını engellemek → **retention +25–45%**

---

## 📋 Proje Özeti

Emotion-Based Editing, videolardaki duygusal ve enerjik içeriği analiz ederek, izleyici dikkatini artırmak için otomatik görsel efektler uygulayan bir video düzenleme sistemi. Sistem, ses ve transkript analizi yaparak duygu tespiti gerçekleştirir, enerji seviyelerini hesaplar ve bu verilere göre zoom efektleri, renk düzenlemeleri ve geçiş optimizasyonları uygular.

---

## 🏗️ Proje Yapısı

```
Emotion-Based-Editing/
├── src/
│   ├── emotion_detection/      # EB-1: Duygu ve enerji tespiti
│   │   ├── audio_analyzer.py      # Audio → features extraction
│   │   ├── text_emotion.py        # Transcript → emotion classification
│   │   ├── energy_calculator.py   # Energy scoring (amplitude + speaking rate)
│   │   └── timeline_generator.py # Emotion timeline JSON output
│   │
│   ├── video_effects/          # EB-2, EB-3, EB-4: Görsel efektler
│   │   ├── zoom_effect.py         # EB-2: Auto camera zoom (Ken Burns)
│   │   ├── color_grading.py       # EB-3: Emotion-based color grading
│   │   └── transitions.py         # EB-4: Smooth jump-cut replacer
│   │
│   ├── face_tracking/          # Yüz takibi (zoom için)
│   │   └── face_tracker.py        # MediaPipe face detection & tracking
│   │
│   └── pipeline/               # Ana iş akışı
│       └── main_pipeline.py       # Orchestration & rendering
│
├── models/                     # ML modelleri (emotion detection)
├── luts/                       # LUT dosyaları (.cube format)
├── config/                     # Konfigürasyon dosyaları
│   └── emotion_config.yaml
├── main.py                     # Entry point
├── requirements.txt
└── README.md
```

---

## 🔄 Pipeline Akışı

### **1. EB-1 — Emotion + Energy Detection Model Entegrasyonu**

**Açıklama:**
- Audio + transcript'den duygu analizi çıkarır
- Enerji seviyesini (amplitude + speaking rate) hesaplar
- Timecode bazlı "emotion timeline" üretir
- Duygusal pikleri tespit eder (ör.: sevinç, vurgu, heyecan)

**İş Akışı:**
```
Video Input
    ↓
Audio Extraction (WAV)
    ↓
┌─────────────────┬─────────────────┐
│  Audio Analysis │  Speech-to-Text │
│  (librosa)      │  (Whisper)      │
│  - MFCC         │                 │
│  - Amplitude    │  → Transcript   │
│  - Spectral     │                 │
└────────┬────────┴────────┬────────┘
         │                 │
         ↓                 ↓
    Audio Emotion    Text Emotion
    (ML Model)       (BERT-based)
         │                 │
         └────────┬────────┘
                  ↓
         Emotion Fusion
                  ↓
         Energy Calculation
         (0.4*amplitude + 0.3*speaking_rate + 0.3*spectral)
                  ↓
         Timeline Generation
         (sliding window: 0.5s)
                  ↓
    Emotion Timeline JSON
```

**Çıktı Formatı:**
```json
[
  { "start": 4.10, "end": 5.90, "emotion": "excitement", "energy": 0.92 },
  { "start": 12.00, "end": 12.80, "emotion": "anger", "energy": 0.88 },
  { "start": 18.50, "end": 20.30, "emotion": "humor", "energy": 0.75 }
]
```

---

### **2. EB-2 — Energy Surge → Auto Camera Zoom**

**Açıklama:**
- Yüksek enerji seviyesine sahip segmentlerde smooth zoom-in (Ken Burns effect) uygular
- Zoom yoğunluğu "energy score" ile orantılıdır

**İş Akışı:**
```
Emotion Timeline
    ↓
Filter: energy > 0.75
    ↓
For each high-energy segment:
    ├─→ Face Detection (MediaPipe)
    ├─→ Calculate Face Center
    ├─→ Zoom Factor = 1.0 + (energy - 0.75) * 0.48
    │   (Max 12% crop → zoom_factor max 1.12)
    └─→ Apply FFmpeg zoompan filter
        zoompan=z='min(zoom+0.0015,{zoom_factor})':d={duration}
            :x='{face_x}':y='{face_y}':s={resolution}
```

**Kabul Kriterleri:**
- ✅ Yüz merkezde kalmalı (face-tracking bağımlılığı)
- ✅ Zoom max %12 crop
- ✅ Minimum jumpcut hissi

---

### **3. EB-3 — Emotion-Based Color Grading**

**Açıklama:**
Duygu → renk tonu eşleşmesi:

| Emotion    | Renk/Stil                          |
|------------|------------------------------------|
| Excitement | Saturation ↑ + Warm tones          |
| Tension    | Contrast ↑ + Cold tones            |
| Humor      | Slight vibrance + playful overlay  |
| Sadness    | Desaturation + vignette            |

**İş Akışı:**
```
Emotion Timeline
    ↓
For each segment:
    ├─→ Map emotion → LUT file
    │   excitement → warm_vibrant.cube
    │   tension → cold_contrast.cube
    │   humor → playful_vibrant.cube
    │   sadness → desaturated_vignette.cube
    │
    └─→ Apply FFmpeg filter chain
        - Option 1: lut3d filter (LUT file)
        - Option 2: Direct filters (eq, curves, colorbalance)
```

**FFmpeg Örnekleri:**
```bash
# Excitement: Saturation ↑ + Warm tones
eq=saturation=1.3:gamma=1.1,
curves=preset=lighter,
colorbalance=rs=0.1:gs=-0.05:bs=-0.1

# Tension: Contrast ↑ + Cold tones
eq=contrast=1.2,
colorbalance=rs=-0.1:gs=0.05:bs=0.15

# Sadness: Desaturation + Vignette
eq=saturation=0.6,
vignette=angle=PI/4
```

---

### **4. EB-4 — Transition Optimizer (Smooth Jump-Cut Replacer)**

**Açıklama:**
- Duygusal pikten daha sakin bölgeye geçişlerde hızlı fade, motion blur veya dynamic zoom geçişi uygular
- Jump-cut algısı azaltılır

**İş Akışı:**
```
Emotion Timeline
    ↓
For each segment pair (i, i+1):
    ├─→ Calculate energy_drop = segment[i].energy - segment[i+1].energy
    │
    ├─→ If energy_drop > 0.3:  # High → Low transition
    │   ├─→ Choose transition type:
    │   │   - energy_drop > 0.5 → fade (0.3s)
    │   │   - energy_drop 0.3-0.5 → zoom_blur (0.2s)
    │   │   - else → motion_blur (0.15s)
    │   │
    │   └─→ Apply FFmpeg transition filter
    │       xfade=transition=fade:duration={duration}:offset={time}
    │
    └─→ If energy_drop < -0.3:  # Low → High transition
        └─→ Apply quick zoom-in transition
```

**Başarı Kriteri:**
- ✅ "TikTok jump-cut vibe" = Kabul edilir
- ❌ "Amatör kesik hissi" = Kabul edilmez

---

## 🎯 Tam Pipeline Akışı

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Video File                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Audio Extraction                                   │
│  - Extract audio track (WAV format)                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: EB-1 - Emotion + Energy Detection                  │
│  ├─ Audio Analysis → Audio Emotion                          │
│  ├─ Speech-to-Text → Text Emotion                           │
│  ├─ Energy Calculation                                      │
│  └─ Timeline Generation → emotion_timeline.json             │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Face Tracking Preparation                          │
│  - Detect faces in video frames                             │
│  - Track face positions over time                           │
│  - Generate face_tracking_data.json                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Effect Application (Parallel Processing)            │
│  ├─ EB-2: Auto Zoom (high-energy segments)                  │
│  ├─ EB-3: Color Grading (emotion-based)                     │
│  └─ EB-4: Transitions (energy drop segments)                │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: FFmpeg Rendering                                   │
│  - Combine all filters in single pass                       │
│  - Render final video                                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT:                                  │
│  ├─ emotion_timeline.json                                   │
│  ├─ rendered_video.mp4                                      │
│  └─ processing_report.json                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Çıktılar

1. **Emotion Timeline JSON** (`emotion_timeline.json`)
   - Timecode bazlı duygu ve enerji verileri
   - Segment bazlı analiz sonuçları

2. **Efektli Render** (`rendered_video.mp4`)
   - Tüm efektlerin uygulandığı final video
   - Demo amaçlı test çıktısı

3. **UI Ekran + Ayar Paneli** (Gelecek versiyon)
   - Parametre ayarlama arayüzü
   - Preview ve export özellikleri

4. **Teknik Rapor** (`processing_report.json`)
   - Performans metrikleri
   - Doğruluk analizi
   - İşleme süreleri

---

## 🚀 Kurulum (Yakında)

```bash
# 1. Repository'yi klonla
git clone <repository-url>
cd Emotion-Based-Editing

# 2. Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. FFmpeg kurulumu (sistem gereksinimi)
# macOS: brew install ffmpeg
# Ubuntu: sudo apt-get install ffmpeg
# Windows: https://ffmpeg.org/download.html
```

---

## 📝 Kullanım (Yakında)

```bash
python main.py --input video.mp4 --output edited_video.mp4
```

---

## 🔧 Teknoloji Stack

- **Audio Processing:** `librosa`, `whisper`, `soundfile`
- **Emotion Detection:** `transformers`, `torch`, `emotion-recognition`
- **Face Tracking:** `mediapipe`, `dlib`
- **Video Processing:** `moviepy`, `ffmpeg-python`, `opencv-python`
- **Scene Detection:** `scenedetect`
- **Utilities:** `numpy`, `pydub`, `pyyaml`

---

## 📊 Performans Hedefleri

- **Emotion Detection Accuracy:** F1-score > 0.75
- **Energy Correlation:** Manual annotation ile r > 0.80
- **Face Tracking:** Yüz merkezde kalma oranı > 90%
- **Processing Speed:** 1x real-time (30fps video için ~30 fps processing)
- **Retention Improvement:** +25-45% (A/B test ile doğrulanacak)

---

## 📄 Lisans

[Lisans bilgisi eklenecek]

---

## 👥 Katkıda Bulunanlar

[Katkıda bulunanlar listesi eklenecek]

---

## 📞 İletişim

[İletişim bilgileri eklenecek]


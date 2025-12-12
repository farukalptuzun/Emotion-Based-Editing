#!/usr/bin/env python3
"""
EB-2 Zoom Effect Test Script (Debug)
"""

import sys

print("Step 1: Starting imports...")
sys.stdout.flush()

print("Step 2: Importing VideoProcessor...")
sys.stdout.flush()
from src.video_processing import VideoProcessor
print("Step 3: VideoProcessor imported successfully")
sys.stdout.flush()

# Dosya yolları
video_path = "SaveVid.Net_AQPJ7_bD5mZH851wHMWlg9mQOSURuDKf91SyReKyejLdWqFNKzKuG2xYYMGvzixeIVrQzUG_ez5k7rtoSWsMSvEQiOGkWmwsQtRow2A.mp4"
timeline_path = "SaveVid.Net_AQPJ7_bD5mZH851wHMWlg9mQOSURuDKf91SyReKyejLdWqFNKzKuG2xYYMGvzixeIVrQzUG_ez5k7rtoSWsMSvEQiOGkWmwsQtRow2A_emotion_timeline.json"
output_path = "output_zoom_test.mp4"

print("="*60)
print("EB-2 Zoom Effect Test")
print("="*60)
print(f"Video: {video_path}")
print(f"Timeline: {timeline_path}")
print(f"Output: {output_path}\n")
sys.stdout.flush()

print("Step 4: Creating VideoProcessor...")
sys.stdout.flush()

# Video processor oluştur
# energy_threshold=0.4 (düşük threshold - test için)
# Normal kullanımda 0.75 olmalı
try:
    processor = VideoProcessor(
        energy_threshold=0.4,  # Test için düşük threshold (timeline'da energy düşük)
        face_sample_rate=2.0  # Her 2 saniyede bir frame (daha hızlı, daha az hassas)
    )
    print("Step 5: VideoProcessor created successfully")
    sys.stdout.flush()
except Exception as e:
    print(f"❌ Error creating VideoProcessor: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Zoom efektlerini uygula
print("Step 6: Applying zoom effects...")
sys.stdout.flush()

try:
    processor.apply_zoom_effects(
        video_path=video_path,
        timeline_path=timeline_path,
        output_path=output_path
    )
    print(f"\n✅ Success! Zoom effect applied. Output: {output_path}")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()



"""
Emotion-Based Editing - Main Entry Point

EB-1: Emotion + Energy Detection
"""

import sys
import argparse
from src.emotion_detection import EmotionEnergyDetector


def main():
    parser = argparse.ArgumentParser(
        description="Emotion-Based Editing - EB-1: Emotion + Energy Detection"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input video file path"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output timeline JSON path (default: <input>_emotion_timeline.json)"
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default=None,
        help="Language code for transcription (None = auto-detect, 'en' = English, 'tr' = Turkish)"
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep extracted audio file after processing"
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = EmotionEnergyDetector(whisper_model=args.whisper_model)
    
    # Process video
    timeline = detector.process_video(
        video_path=args.input,
        output_timeline_path=args.output,
        language=args.language,
        keep_audio=args.keep_audio
    )
    
    print(f"\n✅ Processing complete! Generated {len(timeline)} timeline segments.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


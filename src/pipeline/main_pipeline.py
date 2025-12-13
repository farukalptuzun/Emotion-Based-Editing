"""
Emotion-Based Editing - Main Pipeline
EB-1: Emotion + Energy Detection
EB-2: Auto Camera Zoom
EB-3: Emotion-Based Color Grading
"""

import sys
import argparse
import json
import shutil
from src.emotion_detection import EmotionEnergyDetector
from src.video_processing.video_processor import VideoProcessor
from src.video_effects.color_grading import ColorGradingProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Emotion-Based Editing - Full Pipeline"
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
        help="Output video path (default: <input>_edited.mp4)"
    )
    parser.add_argument(
        "--skip-eb1",
        action="store_true",
        help="Skip EB-1 (emotion detection), use existing timeline"
    )
    parser.add_argument(
        "--skip-eb2",
        action="store_true",
        help="Skip EB-2 (zoom effects)"
    )
    parser.add_argument(
        "--skip-eb3",
        action="store_true",
        help="Skip EB-3 (color grading)"
    )
    parser.add_argument(
        "--timeline",
        type=str,
        default=None,
        help="Timeline JSON path (if skip-eb1, use this)"
    )
    
    args = parser.parse_args()
    
    # Output path
    if args.output is None:
        base_name = args.input.rsplit('.', 1)[0]
        args.output = f"{base_name}_edited.mp4"
    
    # Timeline path
    if args.timeline is None:
        base_name = args.input.rsplit('.', 1)[0]
        args.timeline = f"{base_name}_emotion_timeline.json"
    
    # EB-1: Emotion Detection
    if not args.skip_eb1:
        print("\n" + "="*60)
        print("EB-1: Emotion + Energy Detection")
        print("="*60)
        detector = EmotionEnergyDetector()
        timeline = detector.process_video(
            video_path=args.input,
            output_timeline_path=args.timeline
        )
    else:
        print(f"Loading existing timeline: {args.timeline}")
        with open(args.timeline, 'r', encoding='utf-8') as f:
            timeline = json.load(f)
    
    # Intermediate video path
    intermediate_video = args.input
    
    # EB-2: Zoom Effects
    if not args.skip_eb2:
        print("\n" + "="*60)
        print("EB-2: Auto Camera Zoom")
        print("="*60)
        zoom_output = args.output.replace('.mp4', '_zoom.mp4')
        processor = VideoProcessor()
        intermediate_video = processor.apply_zoom_effects(
            video_path=intermediate_video,
            timeline_path=args.timeline,
            output_path=zoom_output
        )
    
    # EB-3: Color Grading
    if not args.skip_eb3:
        print("\n" + "="*60)
        print("EB-3: Emotion-Based Color Grading")
        print("="*60)
        color_processor = ColorGradingProcessor()
        
        # Get video info for color grading
        from moviepy import VideoFileClip
        video = VideoFileClip(intermediate_video)
        video_info = {
            "width": video.w,
            "height": video.h,
            "fps": video.fps,
            "duration": video.duration
        }
        video.close()
        
        final_output = color_processor.apply_color_grading(
            video_path=intermediate_video,
            timeline_path=args.timeline,
            output_path=args.output,
            video_info=video_info
        )
    else:
        # No color grading, just copy
        shutil.copy(intermediate_video, args.output)
    
    print("\n" + "="*60)
    print(f"✅ Processing complete! Output: {args.output}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


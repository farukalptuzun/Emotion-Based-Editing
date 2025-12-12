"""
EB-1: Ana orchestrator
Tüm modülleri birleştirerek emotion timeline üretir
"""

from .audio_analyzer import AudioAnalyzer
from .text_emotion import TextEmotionDetector
from .energy_calculator import EnergyCalculator
from .timeline_generator import TimelineGenerator
import os
from typing import List, Dict, Optional


class EmotionEnergyDetector:
    """
    EB-1: Ana orchestrator
    Tüm modülleri birleştirerek emotion timeline üretir
    """
    
    def __init__(self,
                 whisper_model: str = "base",
                 emotion_model: str = "cardiffnlp/twitter-roberta-base-emotion",
                 amplitude_weight: float = 0.4,
                 speaking_rate_weight: float = 0.3,
                 spectral_weight: float = 0.3):
        """
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            emotion_model: HuggingFace emotion model
            amplitude_weight: Energy calculation için amplitude ağırlığı
            speaking_rate_weight: Energy calculation için speaking rate ağırlığı
            spectral_weight: Energy calculation için spectral centroid ağırlığı
        """
        print("Initializing EmotionEnergyDetector...")
        self.audio_analyzer = AudioAnalyzer()
        self.text_emotion = TextEmotionDetector(
            emotion_model=emotion_model,
            whisper_model=whisper_model
        )
        self.energy_calculator = EnergyCalculator(
            amplitude_weight=amplitude_weight,
            speaking_rate_weight=speaking_rate_weight,
            spectral_weight=spectral_weight
        )
        self.timeline_generator = TimelineGenerator()
    
    def process_video(self, 
                     video_path: str, 
                     output_timeline_path: Optional[str] = None,
                     language: Optional[str] = None,
                     keep_audio: bool = False) -> List[Dict]:
        """
        Video'yu işler ve emotion timeline üretir
        
        Args:
            video_path: Input video file path
            output_timeline_path: Output JSON path (None = auto-generate)
            language: Language code for transcription (None = auto-detect)
            keep_audio: Keep extracted audio file after processing
            
        Returns:
            List of timeline segments
        """
        print(f"\n{'='*60}")
        print(f"Processing video: {video_path}")
        print(f"{'='*60}\n")
        
        # 1. Audio extraction
        print("STEP 1: Extracting audio from video...")
        audio_path = self.audio_analyzer.extract_audio_from_video(video_path)
        
        try:
            # 2. Audio analysis
            print("\nSTEP 2: Analyzing audio features...")
            audio_results = self.audio_analyzer.analyze_full_audio(audio_path)
            
            # 3. Text emotion detection
            print("\nSTEP 3: Transcribing and detecting text emotions...")
            text_results = self.text_emotion.analyze_segments(audio_path, language=language)
            
            # 4. Energy calculation & fusion
            print("\nSTEP 4: Calculating energy scores and fusing data...")
            fused_results = self.energy_calculator.fuse_audio_text_data(
                audio_results, 
                text_results
            )
            
            # 5. Timeline generation
            print("\nSTEP 5: Generating emotion timeline...")
            timeline = self.timeline_generator.generate_timeline(fused_results)
            
            # 6. Save timeline
            if output_timeline_path is None:
                base_name = os.path.splitext(video_path)[0]
                output_timeline_path = f"{base_name}_emotion_timeline.json"
            
            self.timeline_generator.save_timeline(timeline, output_timeline_path)
            
            # 7. Generate summary
            summary = self.timeline_generator.get_emotion_summary(timeline)
            print("\n" + "="*60)
            print("EMOTION TIMELINE SUMMARY")
            print("="*60)
            print(f"Total segments: {summary['total_segments']}")
            print(f"Total duration: {summary['total_duration']:.2f} seconds")
            print(f"Average energy: {summary['avg_energy']:.2f}")
            print("\nEmotion distribution:")
            for emotion, stats in summary['emotion_distribution'].items():
                print(f"  {emotion}: {stats['count']} segments, "
                      f"{stats['total_duration']:.2f}s, "
                      f"avg energy: {stats['avg_energy']:.2f}")
            print("="*60 + "\n")
            
            return timeline
            
        finally:
            # Cleanup
            if not keep_audio and os.path.exists(audio_path):
                print(f"Cleaning up temporary audio file: {audio_path}")
                os.remove(audio_path)
    
    def process_audio_only(self,
                          audio_path: str,
                          output_timeline_path: Optional[str] = None,
                          language: Optional[str] = None) -> List[Dict]:
        """
        Sadece audio dosyası için emotion timeline üretir (video extraction olmadan)
        
        Args:
            audio_path: Input audio file path
            output_timeline_path: Output JSON path
            language: Language code for transcription
            
        Returns:
            List of timeline segments
        """
        print(f"\n{'='*60}")
        print(f"Processing audio: {audio_path}")
        print(f"{'='*60}\n")
        
        # 1. Audio analysis
        print("STEP 1: Analyzing audio features...")
        audio_results = self.audio_analyzer.analyze_full_audio(audio_path)
        
        # 2. Text emotion detection
        print("\nSTEP 2: Transcribing and detecting text emotions...")
        text_results = self.text_emotion.analyze_segments(audio_path, language=language)
        
        # 3. Energy calculation & fusion
        print("\nSTEP 3: Calculating energy scores and fusing data...")
        fused_results = self.energy_calculator.fuse_audio_text_data(
            audio_results,
            text_results
        )
        
        # 4. Timeline generation
        print("\nSTEP 4: Generating emotion timeline...")
        timeline = self.timeline_generator.generate_timeline(fused_results)
        
        # 5. Save timeline
        if output_timeline_path is None:
            base_name = os.path.splitext(audio_path)[0]
            output_timeline_path = f"{base_name}_emotion_timeline.json"
        
        self.timeline_generator.save_timeline(timeline, output_timeline_path)
        
        # 6. Generate summary
        summary = self.timeline_generator.get_emotion_summary(timeline)
        print("\n" + "="*60)
        print("EMOTION TIMELINE SUMMARY")
        print("="*60)
        print(f"Total segments: {summary['total_segments']}")
        print(f"Total duration: {summary['total_duration']:.2f} seconds")
        print(f"Average energy: {summary['avg_energy']:.2f}")
        print("="*60 + "\n")
        
        return timeline


# Usage example
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python emotion_energy_detector.py <video_path> [output_path]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    detector = EmotionEnergyDetector()
    timeline = detector.process_video(video_path, output_timeline_path=output_path)
    
    print(f"\nTimeline preview (first 5 segments):")
    for segment in timeline[:5]:
        print(f"  {segment['start']:.2f}s - {segment['end']:.2f}s: "
              f"{segment['emotion']} (energy: {segment['energy']:.2f})")


"""
EB-1: Emotion + Energy Detection Module

Bu modül video'dan duygu ve enerji analizi yapar:
- Audio feature extraction
- Text emotion detection (Whisper + Cardiff NLP)
- Energy calculation
- Timeline generation
"""

from .audio_analyzer import AudioAnalyzer
from .text_emotion import TextEmotionDetector
from .energy_calculator import EnergyCalculator
from .timeline_generator import TimelineGenerator
from .emotion_energy_detector import EmotionEnergyDetector

__all__ = [
    "AudioAnalyzer",
    "TextEmotionDetector",
    "EnergyCalculator",
    "TimelineGenerator",
    "EmotionEnergyDetector"
]


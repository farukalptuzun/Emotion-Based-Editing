import whisper
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from scipy.special import softmax
from typing import Dict, List
import json


class TextEmotionDetector:
    """
    Speech-to-text ve text-based emotion detection
    Cardiff NLP Twitter RoBERTa Emotion Model kullanır
    """
    
    def __init__(self, 
                 emotion_model: str = "cardiffnlp/twitter-roberta-base-emotion",
                 whisper_model: str = "base"):
        """
        Args:
            emotion_model: HuggingFace emotion classification model
            whisper_model: Whisper model size (tiny, base, small, medium, large)
        """
        # Whisper model (speech-to-text)
        print(f"Loading Whisper model: {whisper_model}...")
        self.whisper_model = whisper.load_model(whisper_model)
        
        # Emotion classifier (Cardiff NLP Twitter RoBERTa)
        print(f"Loading emotion model: {emotion_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(emotion_model)
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model)
        
        # Model labels (Cardiff NLP model output)
        self.model_labels = ['anger', 'joy', 'optimism', 'sadness']
        
        # Emotion mapping (model output → our categories)
        self.emotion_mapping = {
            "joy": "excitement",
            "optimism": "excitement",  # Optimism → excitement (pozitif enerji)
            "anger": "anger",
            "sadness": "sadness"
        }
        
        # Device - Metal Performance Shaders (MPS) for Mac, CUDA for NVIDIA, CPU fallback
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("Using Metal Performance Shaders (MPS) for acceleration")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("Using CUDA for acceleration")
        else:
            self.device = "cpu"
            print("Using CPU (no GPU acceleration available)")
        
        self.emotion_model.to(self.device)
        self.emotion_model.eval()
    
    def preprocess_text(self, text: str) -> str:
        """
        Twitter metinleri için preprocessing
        (Cardiff NLP model Twitter için optimize edilmiş)
        """
        if not text:
            return ""
        
        new_text = []
        for t in text.split(" "):
            # @mention → @user
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            # URL → http
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    
    def detect_emotion_from_text(self, text: str) -> Dict[str, float]:
        """
        Text'ten emotion detection yapar (Cardiff NLP model)
        
        Args:
            text: Input text
            
        Returns:
            {
                "emotion": "excitement" | "anger" | "sadness" | "neutral",
                "confidence": 0.0-1.0,
                "all_scores": {...}  # Tüm emotion skorları
            }
        """
        if not text or len(text.strip()) == 0:
            return {
                "emotion": "neutral",
                "confidence": 1.0,
                "all_scores": {}
            }
        
        # Preprocess
        processed_text = self.preprocess_text(text)
        
        # Tokenize
        encoded_input = self.tokenizer(processed_text, return_tensors='pt', 
                                       truncation=True, max_length=512)
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        # Predict
        with torch.no_grad():
            output = self.emotion_model(**encoded_input)
            scores = output[0][0].cpu().detach().numpy()
            scores = softmax(scores)  # Convert to probabilities
        
        # Get all scores
        all_scores = {}
        for i, label in enumerate(self.model_labels):
            all_scores[label] = float(scores[i])
        
        # Get top emotion
        top_idx = np.argmax(scores)
        top_label = self.model_labels[top_idx]
        top_confidence = float(scores[top_idx])
        
        # Map to our emotion categories
        mapped_emotion = self.emotion_mapping.get(top_label, "neutral")
        
        return {
            "emotion": mapped_emotion,
            "confidence": top_confidence,
            "all_scores": all_scores,
            "raw_emotion": top_label  # Original model output
        }
    
    def transcribe_audio(self, audio_path: str, language: str = None) -> List[Dict]:
        """
        Audio'yu transcribe eder ve time-aligned segments döner
        
        Args:
            audio_path: Audio file path
            language: Language code (None = auto-detect, "en" = English, "tr" = Turkish)
            
        Returns:
            List of segments with start, end, text
        """
        print(f"Transcribing audio: {audio_path}...")
        
        # Transcribe with word timestamps
        result = self.whisper_model.transcribe(
            audio_path,
            word_timestamps=True,
            language=language,  # None = auto-detect
            verbose=False
        )
        
        segments = []
        for segment in result["segments"]:
            segments.append({
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": segment["text"].strip()
            })
        
        print(f"Transcribed {len(segments)} segments")
        return segments
    
    def analyze_segments(self, audio_path: str, language: str = None) -> List[Dict]:
        """
        Audio'yu transcribe eder ve her segment için emotion detection yapar
        
        Args:
            audio_path: Audio file path
            language: Language code (None = auto-detect)
            
        Returns:
            List of segments with emotion analysis
        """
        # 1. Transcribe
        segments = self.transcribe_audio(audio_path, language)
        
        # 2. Emotion detection for each segment
        print("Detecting emotions from text...")
        results = []
        
        for i, segment in enumerate(segments):
            if i % 10 == 0:
                print(f"Processing segment {i+1}/{len(segments)}...")
            
            emotion_result = self.detect_emotion_from_text(segment["text"])
            
            results.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "emotion": emotion_result["emotion"],
                "emotion_confidence": emotion_result["confidence"],
                "raw_emotion": emotion_result.get("raw_emotion", "neutral"),
                "all_emotion_scores": emotion_result.get("all_scores", {})
            })
        
        print(f"Emotion detection completed for {len(results)} segments")
        return results
    
    def analyze_text_only(self, text: str) -> Dict:
        """
        Sadece text input için emotion detection (transcription olmadan)
        
        Args:
            text: Input text
            
        Returns:
            Emotion analysis result
        """
        return self.detect_emotion_from_text(text)


# Usage example
if __name__ == "__main__":
    # Initialize detector
    detector = TextEmotionDetector()
    
    # Test with text only
    test_text = "This is amazing! I'm so excited!"
    result = detector.analyze_text_only(test_text)
    print(f"Text: {test_text}")
    print(f"Emotion: {result['emotion']} (confidence: {result['confidence']:.2f})")
    print(f"All scores: {result['all_scores']}")


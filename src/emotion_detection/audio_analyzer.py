import librosa
import numpy as np
from typing import Dict, List, Tuple
import soundfile as sf
from moviepy import VideoFileClip
import os


class AudioAnalyzer:
    """
    Audio'dan feature extraction yapar:
    - Amplitude envelope
    - Speaking rate (syllables/second)
    - Spectral features (MFCC, chroma, spectral centroid)
    """
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.window_size = 0.5  # seconds
    
    def extract_audio_from_video(self, video_path: str, output_audio_path: str = None) -> str:
        """Video'dan audio track'i çıkarır"""
        print(f"Extracting audio from video: {video_path}...")
        video = VideoFileClip(video_path)
        if output_audio_path is None:
            output_audio_path = video_path.replace('.mp4', '_audio.wav').replace('.mov', '_audio.wav')
        # MoviePy 2.x doesn't support verbose and logger parameters
        video.audio.write_audiofile(output_audio_path)
        video.close()
        print(f"Audio extracted to: {output_audio_path}")
        return output_audio_path
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Audio dosyasını yükler"""
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        return y, sr
    
    def calculate_amplitude_envelope(self, y: np.ndarray, frame_length: int = 2048) -> np.ndarray:
        """Amplitude envelope hesaplar (RMS energy)"""
        amplitude = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=self.hop_length)[0]
        return amplitude
    
    def calculate_speaking_rate(self, y: np.ndarray, sr: int) -> float:
        """
        Speaking rate hesaplar (syllables/second approximation)
        Onset detection kullanarak konuşma hızını tahmin eder
        """
        # Onset detection (konuşma başlangıçları)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=self.hop_length, units='time')
        
        if len(onsets) < 2:
            return 0.0
        
        # Onset'ler arası ortalama süre
        intervals = np.diff(onsets)
        avg_interval = np.mean(intervals)
        
        # Syllables/second approximation
        speaking_rate = 1.0 / avg_interval if avg_interval > 0 else 0.0
        
        # Normalize (typical range: 2-5 syllables/second)
        speaking_rate = np.clip(speaking_rate / 5.0, 0.0, 1.0)
        
        return speaking_rate
    
    def extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Spectral features çıkarır"""
        # MFCC (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)
        
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        
        # Chroma (pitch class)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
        
        return {
            "mfcc": mfcc,
            "spectral_centroid": spectral_centroid,
            "chroma": chroma
        }
    
    def analyze_time_window(self, y: np.ndarray, sr: int, start_time: float, end_time: float) -> Dict:
        """Belirli bir zaman aralığı için analiz yapar"""
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        y_segment = y[start_sample:end_sample]
        
        if len(y_segment) == 0:
            return {
                "time": start_time,
                "amplitude": 0.0,
                "speaking_rate": 0.0,
                "spectral_centroid": 0.0
            }
        
        # Amplitude
        amplitude = self.calculate_amplitude_envelope(y_segment)
        amplitude_mean = float(np.mean(amplitude)) if len(amplitude) > 0 else 0.0
        
        # Speaking rate
        speaking_rate = self.calculate_speaking_rate(y_segment, sr)
        
        # Spectral features
        spectral_features = self.extract_spectral_features(y_segment, sr)
        spectral_centroid_mean = float(np.mean(spectral_features["spectral_centroid"])) if len(spectral_features["spectral_centroid"]) > 0 else 0.0
        
        # Normalize spectral centroid (typical range: 1000-5000 Hz)
        spectral_centroid_norm = np.clip(spectral_centroid_mean / 5000.0, 0.0, 1.0)
        
        return {
            "time": start_time,
            "amplitude": amplitude_mean,
            "speaking_rate": speaking_rate,
            "spectral_centroid": spectral_centroid_norm
        }
    
    def analyze_full_audio(self, audio_path: str, window_size: float = 0.5, overlap: float = 0.25) -> List[Dict]:
        """
        Tüm audio'yu sliding window ile analiz eder
        window_size: Analiz penceresi (saniye)
        overlap: Pencere örtüşmesi (saniye)
        """
        print(f"Analyzing audio features: {audio_path}...")
        y, sr = self.load_audio(audio_path)
        duration = len(y) / sr
        
        print(f"Audio duration: {duration:.2f} seconds")
        
        results = []
        current_time = 0.0
        step = window_size - overlap
        
        total_windows = int((duration - window_size) / step) + 1
        print(f"Processing {total_windows} time windows...")
        
        window_count = 0
        while current_time < duration:
            end_time = min(current_time + window_size, duration)
            
            analysis = self.analyze_time_window(y, sr, current_time, end_time)
            analysis["end_time"] = end_time
            results.append(analysis)
            
            current_time += step
            window_count += 1
            
            if window_count % 50 == 0:
                print(f"Processed {window_count}/{total_windows} windows...")
        
        print(f"Audio analysis completed: {len(results)} windows analyzed")
        return results


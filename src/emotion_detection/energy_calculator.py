import numpy as np
from typing import Dict, List
from scipy.signal import find_peaks


class EnergyCalculator:
    """
    Audio ve text verilerini birleştirerek energy score hesaplar
    """
    
    def __init__(self, 
                 amplitude_weight: float = 0.4,
                 speaking_rate_weight: float = 0.3,
                 spectral_weight: float = 0.3):
        self.amplitude_weight = amplitude_weight
        self.speaking_rate_weight = speaking_rate_weight
        self.spectral_weight = spectral_weight
    
    def calculate_energy(self, audio_features: Dict) -> float:
        """
        Energy score hesaplar
        Formula: energy = 0.4*amplitude + 0.3*speaking_rate + 0.3*spectral_centroid
        """
        amplitude = audio_features.get("amplitude", 0.0)
        speaking_rate = audio_features.get("speaking_rate", 0.0)
        spectral_centroid = audio_features.get("spectral_centroid", 0.0)
        
        energy = (
            self.amplitude_weight * amplitude +
            self.speaking_rate_weight * speaking_rate +
            self.spectral_weight * spectral_centroid
        )
        
        # Normalize to [0, 1]
        energy = np.clip(energy, 0.0, 1.0)
        
        return float(energy)
    
    def fuse_audio_text_data(self, 
                             audio_results: List[Dict],
                             text_results: List[Dict]) -> List[Dict]:
        """
        Audio ve text sonuçlarını time-aligned olarak birleştirir
        """
        print("Fusing audio and text data...")
        fused_results = []
        
        # Audio results'ı time bazlı index'e çevir
        audio_dict = {}
        for audio in audio_results:
            time_key = round(audio["time"], 2)
            audio_dict[time_key] = audio
        
        # Text results'ı time bazlı index'e çevir
        text_dict = {}
        for text in text_results:
            # Segment ortası zamanı
            mid_time = (text["start"] + text["end"]) / 2
            time_key = round(mid_time, 2)
            # Multiple text segments can map to same time, keep the one with highest confidence
            if time_key not in text_dict or text.get("emotion_confidence", 0) > text_dict[time_key].get("emotion_confidence", 0):
                text_dict[time_key] = text
        
        # Tüm zaman noktalarını birleştir
        all_times = sorted(set(list(audio_dict.keys()) + list(text_dict.keys())))
        
        print(f"Merging {len(audio_dict)} audio windows with {len(text_dict)} text segments...")
        
        for time in all_times:
            audio_data = audio_dict.get(time, {})
            text_data = text_dict.get(time, {})
            
            # Energy hesapla
            energy = self.calculate_energy(audio_data)
            
            # Emotion (text'ten, yoksa neutral)
            emotion = text_data.get("emotion", "neutral")
            emotion_confidence = text_data.get("emotion_confidence", 0.5)
            
            # Start ve end time'ı belirle
            start_time = audio_data.get("time", time)
            end_time = audio_data.get("end_time", time + 0.5)
            
            # Eğer text segment varsa, onun zaman aralığını kullan
            if text_data:
                start_time = min(start_time, text_data.get("start", start_time))
                end_time = max(end_time, text_data.get("end", end_time))
            
            fused_results.append({
                "time": time,
                "start": start_time,
                "end": end_time,
                "energy": energy,
                "emotion": emotion,
                "emotion_confidence": emotion_confidence,
                "amplitude": audio_data.get("amplitude", 0.0),
                "speaking_rate": audio_data.get("speaking_rate", 0.0),
                "spectral_centroid": audio_data.get("spectral_centroid", 0.0),
                "text": text_data.get("text", "")
            })
        
        print(f"Fused data: {len(fused_results)} time-aligned segments")
        return fused_results
    
    def detect_energy_peaks(self, fused_results: List[Dict], threshold: float = 0.75) -> List[Dict]:
        """
        Energy peaks tespit eder
        """
        if not fused_results:
            return []
        
        energies = [r["energy"] for r in fused_results]
        
        # Peak detection
        # distance: minimum distance between peaks (5 samples = ~2.5 seconds)
        peaks, properties = find_peaks(energies, height=threshold, distance=5)
        
        peak_results = []
        for idx in peaks:
            result = fused_results[idx].copy()
            result["is_peak"] = True
            peak_results.append(result)
        
        print(f"Detected {len(peak_results)} energy peaks (threshold: {threshold})")
        return peak_results


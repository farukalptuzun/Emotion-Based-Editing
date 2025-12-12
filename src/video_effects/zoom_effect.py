from typing import Dict, List, Optional
import json


class ZoomEffectCalculator:
    """
    Energy bazlı zoom efekt hesaplamaları
    """
    
    def __init__(self, 
                 energy_threshold: float = 0.75,
                 max_zoom_factor: float = 1.12,  # Max 12% crop
                 min_zoom_factor: float = 1.0):
        self.energy_threshold = energy_threshold
        self.max_zoom_factor = max_zoom_factor
        self.min_zoom_factor = min_zoom_factor
    
    def calculate_zoom_factor(self, energy: float) -> Optional[float]:
        """
        Energy'ye göre zoom faktörü hesaplar
        Formula: zoom = 1.0 + (energy - threshold) * 0.48
        Max: 1.12 (12% crop)
        """
        if energy < self.energy_threshold:
            return None  # Zoom uygulanmayacak
        
        # Zoom faktörü hesapla
        # energy 0.75'ten 1.0'a kadar → zoom 1.0'dan 1.12'ye kadar
        # (1.0 - 0.75) = 0.25 range
        # (1.12 - 1.0) = 0.12 range
        # multiplier = 0.12 / 0.25 = 0.48
        zoom_factor = self.min_zoom_factor + (energy - self.energy_threshold) * 0.48
        
        # Max limit
        zoom_factor = min(zoom_factor, self.max_zoom_factor)
        
        return zoom_factor
    
    def calculate_crop_coordinates(self,
                                  zoom_factor: float,
                                  video_width: int,
                                  video_height: int,
                                  face_center_x: Optional[float] = None,
                                  face_center_y: Optional[float] = None) -> Dict:
        """
        Zoom için crop koordinatlarını hesaplar
        Yüz merkezde kalacak şekilde crop yapar
        """
        # Eğer yüz tespit edilmediyse, video center kullan
        if face_center_x is None or face_center_y is None:
            face_center_x = video_width / 2
            face_center_y = video_height / 2
        
        # Crop boyutları
        crop_width = video_width / zoom_factor
        crop_height = video_height / zoom_factor
        
        # Crop başlangıç koordinatları (yüz merkezde kalacak)
        crop_x = face_center_x - (crop_width / 2)
        crop_y = face_center_y - (crop_height / 2)
        
        # Sınırları kontrol et
        crop_x = max(0, min(crop_x, video_width - crop_width))
        crop_y = max(0, min(crop_y, video_height - crop_height))
        
        return {
            "crop_x": crop_x,
            "crop_y": crop_y,
            "crop_width": crop_width,
            "crop_height": crop_height,
            "zoom_factor": zoom_factor
        }
    
    def generate_zoompan_filter(self,
                                start_time: float,
                                end_time: float,
                                zoom_factor: float,
                                crop_x: float,
                                crop_y: float,
                                video_width: int,
                                video_height: int,
                                fps: float = 30.0) -> str:
        """
        FFmpeg zoompan filter string'i oluşturur
        Ken Burns effect (smooth zoom-in)
        
        Not: FFmpeg zoompan filter'ı segment bazlı çalışmaz,
        bu yüzden daha kompleks bir yaklaşım gerekebilir.
        Bu basit versiyon tüm video için tek bir zoom uygular.
        """
        duration_frames = int((end_time - start_time) * fps)
        
        # Smooth zoom: başlangıçtan zoom_factor'a kadar
        # zoompan filter formatı:
        # zoompan=z='min(zoom+0.0015,{zoom_factor})':d={duration}:x={x}:y={y}:s={size}
        
        # x ve y: crop koordinatları (yüz merkezde kalacak)
        # Basit versiyon: video center'a zoom
        filter_str = (
            f"zoompan=z='min(zoom+0.0015,{zoom_factor})':"
            f"d={duration_frames}:"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':"
            f"s={video_width}x{video_height}"
        )
        
        return filter_str
    
    def interpolate_face_position(self,
                                  face_positions: List[Dict],
                                  target_time: float) -> Dict:
        """
        Face positions listesinden belirli bir zaman için yüz pozisyonunu bulur
        """
        if not face_positions:
            return {
                "face_detected": False,
                "face_center_x": None,
                "face_center_y": None,
                "confidence": 0.0
            }
        
        # En yakın iki frame'i bul
        before = None
        after = None
        
        for pos in face_positions:
            if pos.get("time", 0) <= target_time:
                before = pos
            elif pos.get("time", 0) > target_time:
                after = pos
                break
        
        # Eğer tam zamanında frame varsa
        if before and abs(before.get("time", 0) - target_time) < 0.01:
            return before
        
        # Interpolation
        if before and after:
            t1 = before.get("time", 0)
            t2 = after.get("time", 0)
            if t2 > t1:
                alpha = (target_time - t1) / (t2 - t1)
                
                if (before.get("face_detected") and after.get("face_detected") and
                    before.get("face_center_x") is not None and after.get("face_center_x") is not None):
                    return {
                        "face_detected": True,
                        "face_center_x": before["face_center_x"] * (1 - alpha) + after["face_center_x"] * alpha,
                        "face_center_y": before["face_center_y"] * (1 - alpha) + after["face_center_y"] * alpha,
                        "confidence": (before.get("confidence", 0) + after.get("confidence", 0)) / 2
                    }
        
        # Fallback
        if before and before.get("face_detected"):
            return before
        if after and after.get("face_detected"):
            return after
        
        return {
            "face_detected": False,
            "face_center_x": None,
            "face_center_y": None,
            "confidence": 0.0
        }
    
    def process_timeline_segments(self,
                                  timeline: List[Dict],
                                  face_positions: List[Dict],
                                  video_width: int,
                                  video_height: int,
                                  fps: float = 30.0) -> List[Dict]:
        """
        Timeline segmentlerini işler ve zoom efektlerini hesaplar
        """
        zoom_segments = []
        
        for segment in timeline:
            energy = segment.get("energy", 0.0)
            start_time = segment["start"]
            end_time = segment["end"]
            
            # Zoom faktörü hesapla
            zoom_factor = self.calculate_zoom_factor(energy)
            
            if zoom_factor is None:
                continue  # Bu segment'e zoom uygulanmayacak
            
            # Yüz pozisyonunu bul (segment ortası)
            mid_time = (start_time + end_time) / 2
            face_data = self.interpolate_face_position(face_positions, mid_time)
            
            # Crop koordinatları
            crop_coords = self.calculate_crop_coordinates(
                zoom_factor,
                video_width,
                video_height,
                face_data.get("face_center_x"),
                face_data.get("face_center_y")
            )
            
            # FFmpeg filter (basit versiyon - geliştirilecek)
            filter_str = self.generate_zoompan_filter(
                start_time,
                end_time,
                zoom_factor,
                crop_coords["crop_x"],
                crop_coords["crop_y"],
                video_width,
                video_height,
                fps
            )
            
            zoom_segments.append({
                "start": start_time,
                "end": end_time,
                "energy": energy,
                "zoom_factor": zoom_factor,
                "crop_x": crop_coords["crop_x"],
                "crop_y": crop_coords["crop_y"],
                "filter": filter_str,
                "face_detected": face_data.get("face_detected", False),
                "face_center_x": face_data.get("face_center_x"),
                "face_center_y": face_data.get("face_center_y")
            })
        
        return zoom_segments
    
    def save_zoom_segments(self, zoom_segments: List[Dict], output_path: str):
        """Zoom segments'ı JSON olarak kaydeder"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(zoom_segments, f, indent=2, ensure_ascii=False)
        print(f"Zoom segments saved to: {output_path}")



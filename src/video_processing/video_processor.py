import json
import subprocess
import os
from typing import List, Dict, Optional
# MoviePy lazy import (yavaş yüklenebilir)
# from moviepy import VideoFileClip  # Lazy import
# FaceTracker lazy import (MediaPipe yüklemesi yavaş olabilir)
# from src.face_tracking.face_tracker import FaceTracker  # Lazy import
from src.video_effects.zoom_effect import ZoomEffectCalculator


class VideoProcessor:
    """
    FFmpeg ile video işleme orchestrator
    EB-2: Auto Camera Zoom uygular
    """
    
    def __init__(self,
                 energy_threshold: float = 0.75,
                 face_sample_rate: float = 1.0):  # Default: Her 1 saniyede bir frame (hız/doğruluk dengesi)
        # Lazy load: FaceTracker'ı sadece gerektiğinde yükle (MediaPipe yüklemesi yavaş olabilir)
        self._face_sample_rate = face_sample_rate
        self._face_tracker = None
        self.zoom_calculator = ZoomEffectCalculator(energy_threshold=energy_threshold)
    
    @property
    def face_tracker(self):
        """Lazy load FaceTracker (MediaPipe sadece gerektiğinde yüklenir)"""
        if self._face_tracker is None:
            print("Initializing FaceTracker (loading MediaPipe - this may take a moment)...")
            # Lazy import: Sadece gerektiğinde MediaPipe yükle
            from src.face_tracking.face_tracker import FaceTracker
            self._face_tracker = FaceTracker(sample_rate=self._face_sample_rate)
            print("FaceTracker initialized")
        return self._face_tracker
    
    def get_video_info(self, video_path: str) -> Dict:
        """Video bilgilerini al"""
        from moviepy import VideoFileClip  # Lazy import
        video = VideoFileClip(video_path)
        info = {
            "width": video.w,
            "height": video.h,
            "fps": video.fps,
            "duration": video.duration
        }
        video.close()
        return info
    
    def apply_zoom_effects(self,
                          video_path: str,
                          timeline_path: str,
                          output_path: str,
                          face_positions_path: Optional[str] = None) -> str:
        """
        Timeline JSON'u okuyup zoom efektlerini uygular
        
        Args:
            video_path: Input video file
            timeline_path: Emotion timeline JSON path
            output_path: Output video path
            face_positions_path: Optional cached face positions JSON
        """
        print(f"\n{'='*60}")
        print("EB-2: Applying Auto Camera Zoom Effects")
        print(f"{'='*60}\n")
        
        # 1. Video bilgileri
        print("STEP 1: Getting video info...")
        video_info = self.get_video_info(video_path)
        print(f"Video: {video_info['width']}x{video_info['height']}, "
              f"{video_info['fps']:.2f} fps, {video_info['duration']:.2f}s")
        
        # 2. Timeline yükle
        print("\nSTEP 2: Loading emotion timeline...")
        with open(timeline_path, 'r', encoding='utf-8') as f:
            timeline = json.load(f)
        print(f"Loaded {len(timeline)} timeline segments")
        
        # 3. Face tracking
        print("\nSTEP 3: Face tracking...")
        if face_positions_path and os.path.exists(face_positions_path):
            print(f"Loading cached face positions from: {face_positions_path}")
            face_positions = self.face_tracker.load_face_positions(face_positions_path)
        else:
            face_positions = self.face_tracker.track_faces_in_video(video_path)
            if face_positions_path:
                self.face_tracker.save_face_positions(face_positions, face_positions_path)
        
        # 4. Zoom efektlerini hesapla
        print("\nSTEP 4: Calculating zoom effects...")
        zoom_segments = self.zoom_calculator.process_timeline_segments(
            timeline,
            face_positions,
            video_info["width"],
            video_info["height"],
            video_info["fps"]
        )
        
        print(f"Found {len(zoom_segments)} segments for zoom effect")
        if zoom_segments:
            print(f"Zoom range: {min(s['zoom_factor'] for s in zoom_segments):.2f} - "
                  f"{max(s['zoom_factor'] for s in zoom_segments):.2f}")
        
        # 5. FFmpeg filter chain oluştur
        print("\nSTEP 5: Building FFmpeg filter chain...")
        filter_complex = self.build_filter_complex(zoom_segments, video_info, timeline)
        
        # 6. FFmpeg komutu çalıştır
        print("\nSTEP 6: Rendering video with FFmpeg...")
        self.render_video(video_path, output_path, filter_complex, video_info)
        
        print(f"\n{'='*60}")
        print(f"✅ Video rendered: {output_path}")
        print(f"{'='*60}\n")
        
        return output_path
    
    def build_filter_complex(self, 
                           zoom_segments: List[Dict],
                           video_info: Dict,
                           timeline: List[Dict]) -> str:
        """
        FFmpeg filter_complex string'i oluşturur
        
        Segment bazlı dinamik zoom:
        - Yüksek enerji → zoom in (yakınlaşma)
        - Düşük enerji → zoom out (uzaklaşma)
        - Timeline'daki her segment için farklı zoom faktörü
        """
        if not timeline:
            return ""  # Timeline yok
        
        width = video_info['width']
        height = video_info['height']
        fps = video_info['fps']
        
        # Timeline'daki TÜM segmentler için zoom faktörü hesapla
        # ÖNEMLİ: Energy değerlerini normalize et (JSON'daki energy 0.0-0.55 arası olabilir)
        # Önce max energy'yi bul
        all_energies = [seg.get("energy", 0.0) for seg in timeline]
        max_energy = max(all_energies) if all_energies else 1.0
        min_energy = min(all_energies) if all_energies else 0.0
        energy_range = max_energy - min_energy if max_energy > min_energy else 1.0
        
        print(f"Energy range in timeline: {min_energy:.2f} - {max_energy:.2f}")
        
        segment_zooms = []
        
        for segment in timeline:
            energy = segment.get("energy", 0.0)
            start_time = segment["start"]
            end_time = segment["end"]
            
            # Energy'yi normalize et (0.0-1.0 aralığına)
            if energy_range > 0:
                normalized_energy = (energy - min_energy) / energy_range
            else:
                normalized_energy = 0.0
            
            # Normalize edilmiş energy'ye göre zoom faktörü hesapla
            # Düşük enerji (0.0-0.3) → zoom out (0.95) - daha görünür uzaklaşma
            # Orta enerji (0.3-0.6) → zoom 1.0-1.15 - daha görünür zoom
            # Yüksek enerji (0.6-1.0) → zoom 1.15-1.25 - çok daha görünür zoom
            if normalized_energy < 0.3:
                zoom_factor = 0.95  # Zoom out (uzaklaşma) - %5 zoom out
            elif normalized_energy < 0.6:
                # Orta enerji: 0.3-0.6 → zoom 1.0-1.15
                zoom_factor = 1.0 + (normalized_energy - 0.3) * 0.5  # 0.0-0.15 range
            else:
                # Yüksek enerji: 0.6-1.0 → zoom 1.15-1.25
                zoom_factor = 1.15 + (normalized_energy - 0.6) * 0.25  # 0.15-0.25 range
            
            segment_zooms.append({
                "start": start_time,
                "end": end_time,
                "zoom": zoom_factor,
                "energy": energy
            })
        
        # FFmpeg expression oluştur: segment bazlı zoom
        # Format: if(between(t,start,end),zoom_factor,if(between(t,start2,end2),zoom_factor2,...))
        # ÖNEMLİ: Segment'leri sıralayıp overlap'leri kaldırmalıyız
        if len(segment_zooms) == 0:
            return ""  # Segment yok
        
        # Segment'leri zaman sırasına göre sırala
        sorted_segments = sorted(segment_zooms, key=lambda x: x['start'])
        
        # Overlap'leri kaldır ve segment'leri birleştir
        # Önce overlap'leri kaldır: Eğer iki segment overlap ediyorsa, zoom faktörü yüksek olanı kullan
        cleaned_segments = []
        for seg in sorted_segments:
            # Bu segment'i eklemeden önce, mevcut segment'lerle overlap var mı kontrol et
            should_add = True
            for i, existing in enumerate(cleaned_segments):
                # Overlap kontrolü
                if not (seg['end'] <= existing['start'] or seg['start'] >= existing['end']):
                    # Overlap var! Yüksek zoom faktörüne sahip olanı kullan
                    if seg['zoom'] > existing['zoom']:
                        # Mevcut segment'i kaldır, yeni segment'i ekle
                        cleaned_segments[i] = seg.copy()
                    should_add = False
                    break
            
            if should_add:
                cleaned_segments.append(seg.copy())
        
        # Aynı zoom faktörüne sahip komşu segment'leri birleştir
        merged_segments = []
        current_seg = None
        
        for seg in sorted(cleaned_segments, key=lambda x: x['start']):
            if current_seg is None:
                current_seg = seg.copy()
            elif (abs(current_seg['zoom'] - seg['zoom']) < 0.001 and 
                  current_seg['end'] >= seg['start'] - 0.1):  # 0.1s tolerance for merging
                # Birleştir: end time'i güncelle
                current_seg['end'] = max(current_seg['end'], seg['end'])
            else:
                # Yeni segment başlat
                merged_segments.append(current_seg)
                current_seg = seg.copy()
        
        if current_seg is not None:
            merged_segments.append(current_seg)
        
        # Eğer hala çok fazla segment varsa, sadece en önemli olanları kullan (max 10 segment)
        if len(merged_segments) > 10:
            # En yüksek enerji farkına sahip segment'leri seç
            merged_segments.sort(key=lambda x: x['energy'], reverse=True)
            merged_segments = merged_segments[:10]
            merged_segments.sort(key=lambda x: x['start'])
        
        # Segment bazlı dinamik zoom için FFmpeg'de trim + crop + scale + concat kullanıyoruz
        # crop filter'ı nested if() expression'larını parse edemiyor
        # Bu yüzden her segment için ayrı trim + crop + scale uygulayıp concat ile birleştiriyoruz
        
        if len(merged_segments) == 0:
            return ""  # Segment yok
        
        # Segment'leri zaman sırasına göre sırala
        merged_segments = sorted(merged_segments, key=lambda x: x['start'])
        
        # Video süresini al
        video_duration = video_info.get('duration', 0.0)
        if video_duration <= 0:
            # Timeline'dan video süresini tahmin et
            if timeline:
                video_duration = max(seg.get('end', 0.0) for seg in timeline)
            else:
                video_duration = 60.0  # Default 60 saniye
        
        # Face center (şimdilik video merkezi)
        face_center_x = width / 2
        face_center_y = height / 2
        
        # Default zoom (segment dışındaki zamanlar için)
        default_zoom = 1.0
        
        # Tüm video'yu segment'lere böl (segment'ler arasındaki boşlukları da dahil et)
        all_segments = []
        current_time = 0.0
        
        for seg in merged_segments:
            seg_start = seg['start']
            seg_end = seg['end']
            seg_zoom = seg['zoom']
            
            # Eğer current_time ile seg_start arasında boşluk varsa, default zoom segment'i ekle
            if current_time < seg_start:
                all_segments.append({
                    'start': current_time,
                    'end': seg_start,
                    'zoom': default_zoom,
                    'type': 'default'
                })
            
            # Zoom segment'ini ekle
            all_segments.append({
                'start': seg_start,
                'end': seg_end,
                'zoom': seg_zoom,
                'type': 'zoom'
            })
            
            current_time = seg_end
        
        # Video sonuna kadar default zoom segment'i ekle
        if current_time < video_duration:
            all_segments.append({
                'start': current_time,
                'end': video_duration,
                'zoom': default_zoom,
                'type': 'default'
            })
        
        # Segment sayısını sınırla (çok fazla segment filter'ı çok uzun yapar)
        if len(all_segments) > 20:
            # En yüksek zoom farkına sahip segment'leri seç
            segments_with_diff = []
            for seg in all_segments:
                diff = abs(seg['zoom'] - default_zoom)
                segments_with_diff.append((diff, seg))
            segments_with_diff.sort(reverse=True, key=lambda x: x[0])
            # En önemli segment'leri al, ama zaman sırasını koru
            important_segments = [seg for _, seg in segments_with_diff[:15]]
            important_segments.sort(key=lambda x: x['start'])
            all_segments = important_segments
            print(f"⚠️  Too many segments, using top {len(all_segments)} segments with highest zoom difference")
        
        # Her segment için trim + crop + scale filter'ı oluştur
        filter_parts = []
        output_labels = []
        
        for i, seg in enumerate(all_segments):
            seg_start = seg['start']
            seg_end = seg['end']
            seg_zoom = seg['zoom']
            
            # Crop koordinatlarını hesapla
            crop_w = int(width / seg_zoom)
            crop_h = int(height / seg_zoom)
            crop_x = int((width - crop_w) / 2)
            crop_y = int((height - crop_h) / 2)
            
            # Output label
            output_label = f"v{i}"
            output_labels.append(output_label)
            
            # Filter: trim + setpts + crop + scale
            filter_part = (
                f"[0:v]trim=start={seg_start:.3f}:end={seg_end:.3f},"
                f"setpts=PTS-STARTPTS,"
                f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
                f"scale={width}:{height}:flags=lanczos[{output_label}]"
            )
            filter_parts.append(filter_part)
        
        # Concat filter: Tüm segment'leri birleştir
        concat_inputs = "".join([f"[{label}]" for label in output_labels])
        concat_filter = f"{concat_inputs}concat=n={len(output_labels)}:v=1[outv]"
        
        # Tüm filter'ları birleştir
        filter_str = ";".join(filter_parts) + ";" + concat_filter
        
        zoom_range = [seg["zoom"] for seg in merged_segments]
        print(f"Segment-based dynamic zoom: {len(all_segments)} segments (merged from {len(segment_zooms)} original)")
        print(f"Zoom range: {min(zoom_range):.2f} - {max(zoom_range):.2f}")
        print(f"Using trim+crop+scale+concat approach with {len(all_segments)} segments")
        print(f"Energy-based: Low energy → zoom out (0.95), High energy → zoom in (1.25)")
        
        # Debug: Filter string uzunluğunu kontrol et
        if len(filter_str) > 5000:
            print(f"⚠️  Warning: Filter string is very long ({len(filter_str)} chars), may cause FFmpeg parsing issues")
        
        # Debug: Filter string'in ilk 300 karakterini yazdır
        print(f"\n🔍 DEBUG: FFmpeg filter string (first 300 chars):")
        print(f"{filter_str[:300]}...")
        print(f"Total filter string length: {len(filter_str)} chars\n")
        
        return filter_str
    
    def render_video(self, 
                    input_path: str,
                    output_path: str,
                    filter_complex: str,
                    video_info: Dict):
        """FFmpeg ile video render"""
        if not filter_complex:
            # Filter yoksa, video'yu kopyala
            print("No zoom effects to apply, copying video...")
            cmd = [
                "ffmpeg", "-i", input_path,
                "-c", "copy",
                output_path, "-y"
            ]
        else:
            # Use VideoToolbox hardware encoder on macOS for Metal acceleration
            import platform
            use_hardware_encoder = platform.system() == "Darwin"  # macOS
            
            # Segment bazlı dinamik zoom için trim+crop+scale+concat kullanıyoruz
            # filter_complex kullanıyoruz çünkü multiple input/output var
            print("Using software encoder (trim+crop+scale+concat approach)")
            cmd = [
                "ffmpeg", "-i", input_path,
                "-filter_complex", filter_complex,
                "-map", "[outv]",  # Concat output'unu map et (video)
                "-map", "0:a",  # Audio stream'i de map et (SES İÇİN GEREKLİ!)
                "-c:v", "libx264",  # Software encoder (daha stabil)
                "-preset", "ultrafast",  # En hızlı encoding
                "-crf", "28",  # Quality (18-28 arası, 28 = hızlı encoding, kabul edilebilir kalite)
                "-threads", "0",  # Tüm CPU core'ları kullan
                "-c:a", "copy",
                output_path, "-y"
            ]
        
        print(f"Running FFmpeg command...")
        print(f"Optimized settings: segment-based dynamic crop+scale, ultrafast preset, CRF 28, threads=0")
        print(f"Estimated: 1-2 min for 60s video (crop+scale is fast)")
        
        # Debug: FFmpeg komutunu yazdır
        print(f"\n🔍 DEBUG: FFmpeg command:")
        print(f"  Filter: {filter_complex[:200]}..." if len(filter_complex) > 200 else f"  Filter: {filter_complex}")
        print()
        
        try:
            # Basit yaklaşım: stderr'i direkt terminal'e yönlendir (progress gösterir)
            # FFmpeg progress bilgisini stderr'e yazar, bu yüzden direkt gösteriyoruz
            result = subprocess.run(
                cmd,
                check=True,
                stderr=None  # stderr'i terminal'e yönlendir (progress gösterir)
            )
            print("\n✅ FFmpeg completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ FFmpeg error (return code: {e.returncode})")
            raise


# Usage example
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python video_processor.py <video_path> <timeline_path> <output_path> [face_positions_path]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    timeline_path = sys.argv[2]
    output_path = sys.argv[3]
    face_positions_path = sys.argv[4] if len(sys.argv) > 4 else None
    
    processor = VideoProcessor()
    processor.apply_zoom_effects(
        video_path=video_path,
        timeline_path=timeline_path,
        output_path=output_path,
        face_positions_path=face_positions_path
    )


import json
import subprocess
from typing import List, Dict, Optional
from enum import Enum


class EmotionColorStyle(Enum):
    """Emotion → Color Grading Style Mapping"""
    EXCITEMENT = "excitement"  # Saturation ↑ Warm tones
    TENSION = "tension"  # Contrast ↑ Cold tones (anger → tension)
    HUMOR = "humor"  # Slight vibrance + playful overlay
    SADNESS = "sadness"  # Desaturation + vignette
    NEUTRAL = "neutral"  # No effect (or subtle)


class ColorGradingProcessor:
    """
    EB-3: Emotion-Based Color Grading
    FFmpeg + LUT pipeline ile duygu bazlı renk düzenleme
    """
    
    def __init__(self):
        # Emotion → Color Style Mapping
        self.emotion_mapping = {
            "excitement": EmotionColorStyle.EXCITEMENT,
            "anger": EmotionColorStyle.TENSION,  # Anger → Tension (cold tones)
            "sadness": EmotionColorStyle.SADNESS,
            "neutral": EmotionColorStyle.NEUTRAL,
            # Humor detection: excitement + low energy → humor
            "humor": EmotionColorStyle.HUMOR
        }
        
        # Color grading parameters per emotion
        self.grading_params = {
            EmotionColorStyle.EXCITEMENT: {
                "saturation": 1.3,  # ↑ Saturation
                "contrast": 1.1,
                "brightness": 1.05,
                "warmth": 1.15,  # Warm tones (red/orange shift)
                "vibrance": 1.2
            },
            EmotionColorStyle.TENSION: {
                "saturation": 0.95,
                "contrast": 1.4,  # ↑ Contrast
                "brightness": 0.95,
                "warmth": 0.85,  # Cold tones (blue/cyan shift)
                "vibrance": 0.9
            },
            EmotionColorStyle.HUMOR: {
                "saturation": 1.1,
                "contrast": 1.05,
                "brightness": 1.08,
                "warmth": 1.1,
                "vibrance": 1.15,  # Slight vibrance
                "playful_overlay": True  # Playful overlay (subtle color shift)
            },
            EmotionColorStyle.SADNESS: {
                "saturation": 0.5,  # Daha soluk (desaturation)
                "contrast": 0.85,  # Daha düşük kontrast
                "brightness": 0.85,  # Daha karanlık
                "warmth": 0.80,  # Daha soğuk tonlar (blue/cyan shift)
                "vignette": True,  # Vignette effect
                "vignette_intensity": 0.5,  # Daha belirgin vignette
                "blue_tint": True  # Blue/cyan tint ekle
            },
            EmotionColorStyle.NEUTRAL: {
                "saturation": 1.0,
                "contrast": 1.0,
                "brightness": 1.0,
                "warmth": 1.0
            }
        }
    
    def detect_humor(self, emotion: str, energy: float) -> bool:
        """
        Humor detection: excitement + low energy → humor
        """
        return emotion == "excitement" and energy < 0.3
    
    def get_color_style_for_segment(self, segment: Dict) -> EmotionColorStyle:
        """
        Segment'ten color style belirler
        """
        emotion = segment.get("emotion", "neutral")
        energy = segment.get("energy", 0.5)
        
        # Special case: humor detection
        if self.detect_humor(emotion, energy):
            return EmotionColorStyle.HUMOR
        
        # Map emotion to color style
        return self.emotion_mapping.get(emotion, EmotionColorStyle.NEUTRAL)
    
    def build_color_grading_filter(self, 
                                  timeline: List[Dict],
                                  video_info: Dict) -> str:
        """
        Timeline'dan FFmpeg color grading filter chain'i oluşturur
        
        Returns:
            FFmpeg filter_complex string
        """
        if not timeline:
            return ""  # No timeline
        
        width = video_info.get('width', 1920)
        height = video_info.get('height', 1080)
        fps = video_info.get('fps', 30.0)
        
        # Segment'leri zaman sırasına göre sırala
        sorted_timeline = sorted(timeline, key=lambda x: x['start'])
        
        # Her segment için color grading parametrelerini hesapla
        color_segments = []
        for segment in sorted_timeline:
            style = self.get_color_style_for_segment(segment)
            params = self.grading_params[style]
            
            color_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "style": style,
                "params": params
            })
        
        # Video süresini al
        video_duration = video_info.get('duration', 0.0)
        if video_duration <= 0 and timeline:
            video_duration = max(seg.get('end', 0.0) for seg in timeline)
        
        # Segment'leri birleştir (overlap'leri kaldır)
        merged_segments = self._merge_color_segments(color_segments, video_duration)
        
        # FFmpeg filter string'i oluştur
        return self._build_segment_based_filter(merged_segments, video_info)
    
    def _merge_color_segments(self, 
                             color_segments: List[Dict],
                             video_duration: float) -> List[Dict]:
        """
        Overlap eden segment'leri birleştir
        Overlap durumunda: dominant emotion'ı kullan (daha uzun süre veya daha yüksek priority)
        """
        if not color_segments:
            return []
        
        # Zaman sırasına göre sırala
        sorted_segments = sorted(color_segments, key=lambda x: x['start'])
        
        # Style priority: excitement/humor > tension/sadness > neutral
        # Sadness priority artırıldı (tension ile aynı seviye)
        style_priority = {
            EmotionColorStyle.EXCITEMENT: 4,
            EmotionColorStyle.HUMOR: 4,
            EmotionColorStyle.TENSION: 3,
            EmotionColorStyle.SADNESS: 3,  # Increased from 2 to 3 (same as tension)
            EmotionColorStyle.NEUTRAL: 1
        }
        
        # Tüm zaman noktalarını topla
        time_points = set()
        for seg in sorted_segments:
            time_points.add(seg['start'])
            time_points.add(seg['end'])
        time_points.add(0.0)
        time_points.add(video_duration)
        time_points = sorted(list(time_points))
        
        # Her zaman aralığı için dominant style'ı bul
        merged = []
        for i in range(len(time_points) - 1):
            start_time = time_points[i]
            end_time = time_points[i + 1]
            
            # Bu zaman aralığında aktif olan segment'leri bul
            active_segments = []
            for seg in sorted_segments:
                seg_start = seg['start']
                seg_end = seg['end']
                # Overlap kontrolü: segment bu zaman aralığıyla kesişiyor mu?
                if not (seg_end <= start_time or seg_start >= end_time):
                    active_segments.append(seg)
            
            if not active_segments:
                # Hiç segment yok, neutral
                style = EmotionColorStyle.NEUTRAL
            else:
                # Neutral olmayan segment'leri tercih et
                # Eğer başka bir emotion varsa, neutral'ı ignore et
                non_neutral = [s for s in active_segments 
                              if s['style'] != EmotionColorStyle.NEUTRAL]
                
                if non_neutral:
                    # Neutral olmayan segment varsa, en yüksek priority'ye sahip olanı seç
                    best_seg = max(non_neutral, 
                                  key=lambda s: style_priority.get(s['style'], 0))
                    style = best_seg['style']
                else:
                    # Sadece neutral varsa, neutral kullan
                    style = EmotionColorStyle.NEUTRAL
            
            # Eğer önceki segment ile aynı style ise birleştir
            if merged and merged[-1]['style'] == style:
                merged[-1]['end'] = end_time
            else:
                # Yeni segment oluştur
                # Çok kısa segment'leri de koru (sadness gibi non-neutral için önemli)
                segment_duration = end_time - start_time
                min_duration = 0.1  # Minimum 0.1 saniye
                
                # Non-neutral segment'leri her zaman ekle (ne kadar kısa olursa olsun)
                # Neutral segment'ler için minimum duration kontrolü yap
                if segment_duration >= min_duration or style != EmotionColorStyle.NEUTRAL:
                    merged.append({
                        "start": start_time,
                        "end": end_time,
                        "style": style,
                        "params": self.grading_params[style]
                    })
        
        return merged
    
    def _create_color_filter(self, 
                            params: Dict,
                            style: EmotionColorStyle,
                            width: int,
                            height: int) -> str:
        """
        FFmpeg color grading filter string'i oluştur
        Using curves filter for warmth and brightness (YUV compatible)
        """
        filters = []
        
        # Saturation - WORKING ✅
        sat = params.get("saturation", 1.0)
        if sat != 1.0:
            filters.append(f"eq=saturation={sat:.2f}")
        
        # Contrast - WORKING ✅
        contrast = params.get("contrast", 1.0)
        if contrast != 1.0:
            filters.append(f"eq=contrast={contrast:.2f}")
        
        # Warmth (color temperature) - Using curves filter (YUV compatible)
        warmth = params.get("warmth", 1.0)
        if warmth != 1.0:
            # Warm tones: increase red, decrease blue
            # Cold tones: increase blue, decrease red
            # curves filter: all=master, r=red, g=green, b=blue
            if warmth > 1.0:
                # Warm: boost red channel, reduce blue channel
                warm_boost = (warmth - 1.0) * 0.15  # Scale to 0-0.15 range
                # Red channel: slight boost in midtones
                # Blue channel: slight reduction in midtones
                filters.append(f"curves=all='0/0 0.5/0.5 1/1':r='0/0 0.5/{0.5+warm_boost:.3f} 1/1':b='0/0 0.5/{0.5-warm_boost:.3f} 1/1'")
            else:
                # Cold: boost blue channel, reduce red channel
                cold_boost = (1.0 - warmth) * 0.15
                filters.append(f"curves=all='0/0 0.5/0.5 1/1':r='0/0 0.5/{0.5-cold_boost:.3f} 1/1':b='0/0 0.5/{0.5+cold_boost:.3f} 1/1'")
        
        # Vibrance - Using saturation boost (vibrance filter not available in FFmpeg)
        # Vibrance preserves skin tones while boosting other colors
        # For now, using additional saturation as approximation
        vibrance = params.get("vibrance", 1.0)
        if vibrance != 1.0 and abs(vibrance - sat) > 0.05:  # Only if different from saturation
            # Apply vibrance as additional saturation boost
            vibrance_boost = vibrance
            filters.append(f"eq=saturation={vibrance_boost:.2f}")
        
        # Brightness - Using curves filter (alternative to eq=brightness)
        brightness = params.get("brightness", 1.0)
        if brightness != 1.0:
            # Brightness adjustment using curves (YUV compatible)
            # Lift all channels by brightness amount
            if brightness > 1.0:
                # Increase brightness: lift shadows and midtones
                bright_lift = (brightness - 1.0) * 0.15  # Scale to reasonable range
                filters.append(f"curves=all='0/{bright_lift:.3f} 0.5/{0.5+bright_lift:.3f} 1/1'")
            else:
                # Decrease brightness: lower shadows and midtones
                dark_lift = (1.0 - brightness) * 0.15
                filters.append(f"curves=all='0/{dark_lift:.3f} 0.5/{0.5-dark_lift:.3f} 1/1'")
        
        # Blue tint - ENABLED (for sadness style, additional blue/cyan shift)
        if params.get("blue_tint", False):
            # Additional blue/cyan tint for sadness (beyond warmth adjustment)
            # Boost blue channel slightly, reduce red channel
            filters.append(f"curves=all='0/0 0.5/0.5 1/1':r='0/0 0.5/0.48 1/1':b='0/0 0.5/0.52 1/1'")
        
        # Vignette - ENABLED (for sadness style)
        if params.get("vignette", False):
            intensity = params.get("vignette_intensity", 0.3)
            # Vignette: darken edges
            # FFmpeg vignette parameters: angle, x0, y0, mode
            # x0, y0: center coordinates (0-1 range, 0.5 = center)
            # angle: vignette angle (smaller angle = more intense/darker vignette)
            # mode: 0=black vignette, 1=white vignette
            # Intensity kontrolü için angle'i ayarla
            # intensity 0.3 -> PI/4 (less intense), intensity 0.5 -> PI/6 (more intense)
            if intensity <= 0.3:
                vignette_angle = "PI/4"  # Default, less intense
            elif intensity <= 0.4:
                vignette_angle = "PI/5"  # Medium intensity
            else:
                vignette_angle = "PI/6"  # High intensity (0.5)
            filters.append(f"vignette=angle={vignette_angle}:x0=0.5:y0=0.5:mode=0")
        
        # Playful overlay - ENABLED (for humor style)
        if params.get("playful_overlay", False):
            # Subtle hue shift for playful effect
            filters.append(f"hue=h=5")
        
        if not filters:
            return ""
        
        return ",".join(filters)
    
    def _build_segment_based_filter(self,
                                   merged_segments: List[Dict],
                                   video_info: Dict) -> str:
        """
        Segment bazlı trim + color grading + concat yaklaşımı
        (zoom effect'teki gibi)
        """
        if not merged_segments:
            return ""
        
        width = video_info.get('width', 1920)
        height = video_info.get('height', 1080)
        
        filter_parts = []
        output_labels = []
        
        for i, seg in enumerate(merged_segments):
            seg_start = seg["start"]
            seg_end = seg["end"]
            params = seg["params"]
            style = seg["style"]
            
            # Color grading filter
            color_filter = self._create_color_filter(params, style, width, height)
            
            # Output label
            output_label = f"v{i}"
            output_labels.append(output_label)
            
            # Filter: trim + setpts + color grading
            if color_filter:
                filter_part = (
                    f"[0:v]trim=start={seg_start:.3f}:end={seg_end:.3f},"
                    f"setpts=PTS-STARTPTS,"
                    f"{color_filter}[{output_label}]"
                )
            else:
                # No color grading needed
                filter_part = (
                    f"[0:v]trim=start={seg_start:.3f}:end={seg_end:.3f},"
                    f"setpts=PTS-STARTPTS[{output_label}]"
                )
            
            filter_parts.append(filter_part)
        
        # Concat filter: Tüm segment'leri birleştir
        concat_inputs = "".join([f"[{label}]" for label in output_labels])
        concat_filter = f"{concat_inputs}concat=n={len(output_labels)}:v=1[outv]"
        
        # Tüm filter'ları birleştir
        filter_str = ";".join(filter_parts) + ";" + concat_filter
        
        print(f"\n{'='*60}")
        print(f"📊 Color Grading Debug Info")
        print(f"{'='*60}")
        print(f"Total segments: {len(merged_segments)}")
        print(f"Styles used: {set(seg['style'].value for seg in merged_segments)}")
        print(f"\n📋 Segment Details (first 15 segments):")
        print(f"{'Index':<6} {'Start':<8} {'End':<8} {'Duration':<10} {'Style':<12} {'Saturation':<12} {'Contrast':<10} {'Warmth':<10}")
        print(f"{'-'*80}")
        
        for i, seg in enumerate(merged_segments[:15]):  # İlk 15 segment
            style = seg['style'].value
            params = seg['params']
            duration = seg['end'] - seg['start']
            
            sat = params.get('saturation', 1.0)
            contrast = params.get('contrast', 1.0)
            warmth = params.get('warmth', 1.0)
            
            # Style'a göre emoji ekle
            emoji = {
                'excitement': '🔥',
                'tension': '❄️',
                'humor': '😄',
                'sadness': '😢',
                'neutral': '⚪'
            }.get(style, '❓')
            
            print(f"{i+1:<6} {seg['start']:<8.2f} {seg['end']:<8.2f} {duration:<10.2f} "
                  f"{emoji} {style:<10} {sat:<12.2f} {contrast:<10.2f} {warmth:<10.2f}")
        
        if len(merged_segments) > 15:
            print(f"\n... and {len(merged_segments) - 15} more segments")
        
        # Style istatistikleri
        style_counts = {}
        style_durations = {}
        for seg in merged_segments:
            style = seg['style'].value
            duration = seg['end'] - seg['start']
            style_counts[style] = style_counts.get(style, 0) + 1
            style_durations[style] = style_durations.get(style, 0.0) + duration
        
        print(f"\n📈 Style Statistics:")
        for style, count in sorted(style_counts.items(), key=lambda x: x[1], reverse=True):
            duration = style_durations[style]
            percentage = (duration / sum(style_durations.values())) * 100 if sum(style_durations.values()) > 0 else 0
            print(f"  {style}: {count} segments, {duration:.2f}s ({percentage:.1f}%)")
        
        # Sadness segment'lerini özellikle kontrol et
        sadness_segments = [s for s in merged_segments if s['style'] == EmotionColorStyle.SADNESS]
        if sadness_segments:
            print(f"\n😢 SADNESS SEGMENTS DETECTED: {len(sadness_segments)}")
            for seg in sadness_segments:
                duration = seg['end'] - seg['start']
                print(f"   {seg['start']:.2f}s - {seg['end']:.2f}s (duration: {duration:.2f}s)")
                print(f"      Params: saturation={seg['params'].get('saturation', 1.0):.2f}, "
                      f"contrast={seg['params'].get('contrast', 1.0):.2f}, "
                      f"vignette={seg['params'].get('vignette', False)}")
        else:
            print(f"\n⚠️  NO SADNESS SEGMENTS FOUND in merged segments")
            # Original timeline'da sadness var mı kontrol et
            original_sadness = [s for s in merged_segments if 'sadness' in str(s.get('style', '')).lower()]
            if original_sadness:
                print(f"   (But sadness was detected in original timeline - check merge logic)")
        
        print(f"{'='*60}\n")
        
        return filter_str
    
    def apply_color_grading(self,
                           video_path: str,
                           timeline_path: str,
                           output_path: str,
                           video_info: Optional[Dict] = None) -> str:
        """
        Timeline JSON'u okuyup color grading uygular
        
        Args:
            video_path: Input video file
            timeline_path: Emotion timeline JSON path
            output_path: Output video path
            video_info: Optional video info dict (width, height, fps, duration)
        """
        print(f"\n{'='*60}")
        print("EB-3: Applying Emotion-Based Color Grading")
        print(f"{'='*60}\n")
        
        # 1. Timeline yükle
        print("STEP 1: Loading emotion timeline...")
        with open(timeline_path, 'r', encoding='utf-8') as f:
            timeline = json.load(f)
        print(f"Loaded {len(timeline)} timeline segments")
        
        # Timeline debug: Emotion dağılımı
        emotion_counts = {}
        for seg in timeline:
            emotion = seg.get("emotion", "neutral")
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"  Emotion distribution: {emotion_counts}")
        
        # Humor detection preview
        humor_detected = sum(1 for seg in timeline 
                           if self.detect_humor(seg.get("emotion", "neutral"), 
                                               seg.get("energy", 0.5)))
        if humor_detected > 0:
            print(f"  🎭 Detected {humor_detected} potential humor segments (excitement + low energy)")
        
        # Timeline debug: Emotion dağılımı
        emotion_counts = {}
        for seg in timeline:
            emotion = seg.get("emotion", "neutral")
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"  Emotion distribution: {emotion_counts}")
        
        # Humor detection preview
        humor_detected = sum(1 for seg in timeline 
                           if self.detect_humor(seg.get("emotion", "neutral"), 
                                               seg.get("energy", 0.5)))
        if humor_detected > 0:
            print(f"  🎭 Detected {humor_detected} potential humor segments (excitement + low energy)")
        
        # 2. Video info
        if video_info is None:
            print("STEP 2: Getting video info...")
            from moviepy import VideoFileClip
            video = VideoFileClip(video_path)
            video_info = {
                "width": video.w,
                "height": video.h,
                "fps": video.fps,
                "duration": video.duration
            }
            video.close()
        else:
            print("STEP 2: Using provided video info...")
        
        print(f"Video: {video_info['width']}x{video_info['height']}, "
              f"{video_info['fps']:.2f} fps, {video_info['duration']:.2f}s")
        
        # 3. Color grading filter oluştur
        print("\nSTEP 3: Building color grading filter...")
        filter_complex = self.build_color_grading_filter(timeline, video_info)
        
        if not filter_complex:
            print("No color grading to apply, copying video...")
            import shutil
            shutil.copy(video_path, output_path)
            return output_path
        
        # 4. FFmpeg ile render
        print("\nSTEP 4: Rendering video with FFmpeg...")
        self._render_video(video_path, output_path, filter_complex, video_info)
        
        print(f"\n{'='*60}")
        print(f"✅ Color graded video: {output_path}")
        print(f"{'='*60}\n")
        
        return output_path
    
    def _render_video(self,
                     input_path: str,
                     output_path: str,
                     filter_complex: str,
                     video_info: Dict):
        """FFmpeg ile video render"""
        cmd = [
            "ffmpeg", "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]",  # Color grading output
            "-map", "0:a",  # Audio stream
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-threads", "0",
            "-c:a", "copy",
            output_path, "-y"
        ]
        
        print(f"Running FFmpeg command...")
        print(f"Color grading: segment-based trim+color+concat")
        
        try:
            result = subprocess.run(cmd, check=True, stderr=None)
            print("\n✅ FFmpeg completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ FFmpeg error (return code: {e.returncode})")
            raise


# Usage example
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python color_grading.py <video_path> <timeline_path> <output_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    timeline_path = sys.argv[2]
    output_path = sys.argv[3]
    
    processor = ColorGradingProcessor()
    processor.apply_color_grading(
        video_path=video_path,
        timeline_path=timeline_path,
        output_path=output_path
    )


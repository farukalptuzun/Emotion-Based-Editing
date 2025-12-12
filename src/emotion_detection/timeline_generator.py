import json
from typing import List, Dict
from datetime import timedelta


class TimelineGenerator:
    """
    Final emotion timeline JSON'u üretir
    """
    
    def __init__(self, min_segment_duration: float = 0.5, merge_threshold: float = 0.1):
        self.min_segment_duration = min_segment_duration
        self.merge_threshold = merge_threshold  # Merge segments if gap < threshold
    
    def merge_similar_segments(self, fused_results: List[Dict]) -> List[Dict]:
        """
        Benzer emotion ve energy'ye sahip komşu segmentleri birleştirir
        """
        if not fused_results:
            return []
        
        print("Merging similar segments...")
        merged = []
        current_segment = fused_results[0].copy()
        
        for next_segment in fused_results[1:]:
            # Aynı emotion ve benzer energy
            same_emotion = current_segment["emotion"] == next_segment["emotion"]
            energy_diff = abs(current_segment["energy"] - next_segment["energy"])
            time_gap = next_segment["start"] - current_segment["end"]
            
            if same_emotion and energy_diff < 0.2 and time_gap < self.merge_threshold:
                # Merge segments
                current_segment["end"] = next_segment["end"]
                # Average energy
                current_segment["energy"] = (current_segment["energy"] + next_segment["energy"]) / 2
                # Update emotion confidence (average)
                current_segment["emotion_confidence"] = (
                    current_segment.get("emotion_confidence", 0.5) + 
                    next_segment.get("emotion_confidence", 0.5)
                ) / 2
            else:
                # Save current and start new
                if current_segment["end"] - current_segment["start"] >= self.min_segment_duration:
                    merged.append(current_segment)
                current_segment = next_segment.copy()
        
        # Add last segment
        if current_segment["end"] - current_segment["start"] >= self.min_segment_duration:
            merged.append(current_segment)
        
        print(f"Merged {len(fused_results)} segments into {len(merged)} segments")
        return merged
    
    def generate_timeline(self, fused_results: List[Dict]) -> List[Dict]:
        """
        Final timeline JSON formatını üretir
        """
        # Merge similar segments
        merged = self.merge_similar_segments(fused_results)
        
        # Format output
        timeline = []
        for segment in merged:
            timeline.append({
                "start": round(segment["start"], 2),
                "end": round(segment["end"], 2),
                "emotion": segment["emotion"],
                "energy": round(segment["energy"], 2)
            })
        
        print(f"Generated timeline with {len(timeline)} segments")
        return timeline
    
    def save_timeline(self, timeline: List[Dict], output_path: str):
        """Timeline'ı JSON olarak kaydeder"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(timeline, f, indent=2, ensure_ascii=False)
        print(f"Timeline saved to: {output_path}")
    
    def filter_peaks_only(self, timeline: List[Dict], min_energy: float = 0.75) -> List[Dict]:
        """Sadece yüksek energy segmentlerini döner"""
        filtered = [s for s in timeline if s["energy"] >= min_energy]
        print(f"Filtered {len(filtered)} high-energy segments (energy >= {min_energy})")
        return filtered
    
    def get_emotion_summary(self, timeline: List[Dict]) -> Dict:
        """Timeline'dan emotion istatistikleri çıkarır"""
        emotion_counts = {}
        total_energy = 0.0
        total_duration = 0.0
        
        for segment in timeline:
            emotion = segment["emotion"]
            duration = segment["end"] - segment["start"]
            energy = segment["energy"]
            
            if emotion not in emotion_counts:
                emotion_counts[emotion] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "avg_energy": 0.0,
                    "total_energy": 0.0
                }
            
            emotion_counts[emotion]["count"] += 1
            emotion_counts[emotion]["total_duration"] += duration
            emotion_counts[emotion]["total_energy"] += energy * duration
            
            total_energy += energy * duration
            total_duration += duration
        
        # Calculate averages
        for emotion in emotion_counts:
            if emotion_counts[emotion]["total_duration"] > 0:
                emotion_counts[emotion]["avg_energy"] = (
                    emotion_counts[emotion]["total_energy"] / 
                    emotion_counts[emotion]["total_duration"]
                )
        
        return {
            "emotion_distribution": emotion_counts,
            "total_duration": total_duration,
            "avg_energy": total_energy / total_duration if total_duration > 0 else 0.0,
            "total_segments": len(timeline)
        }


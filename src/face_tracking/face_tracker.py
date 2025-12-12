import os
# Matplotlib backend'ini MediaPipe import'undan ÖNCE ayarla
# MediaPipe'in drawing_utils modülü matplotlib kullanıyor
os.environ['MPLBACKEND'] = 'Agg'  # Non-interactive backend (GUI gerektirmez)

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple, Optional
# MoviePy lazy import (yavaş yüklenebilir)
# from moviepy import VideoFileClip  # Lazy import
import json


class FaceTracker:
    """
    MediaPipe ile yüz tespiti ve tracking
    """
    
    def __init__(self, 
                 sample_rate: float = 1.0,  # Her 1 saniyede bir frame (hız/doğruluk dengesi)
                 min_detection_confidence: float = 0.5):
        self.sample_rate = sample_rate
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0=short-range, 1=full-range
            min_detection_confidence=min_detection_confidence
        )
    
    def detect_face_in_frame(self, frame: np.ndarray) -> Dict:
        """Tek bir frame'de yüz tespiti"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            detection = results.detections[0]  # İlk yüzü al
            bbox = detection.location_data.relative_bounding_box
            
            h, w = frame.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Yüz merkezi
            face_center_x = x + width / 2
            face_center_y = y + height / 2
            
            return {
                "face_detected": True,
                "face_center_x": float(face_center_x),
                "face_center_y": float(face_center_y),
                "face_bbox": [x, y, x + width, y + height],
                "confidence": float(detection.score[0])
            }
        else:
            return {
                "face_detected": False,
                "face_center_x": None,
                "face_center_y": None,
                "face_bbox": None,
                "confidence": 0.0
            }
    
    def track_faces_in_video(self, video_path: str) -> List[Dict]:
        """
        Video'da yüz tracking yapar
        Her sample_rate saniyede bir frame analiz eder
        """
        from moviepy import VideoFileClip  # Lazy import
        video = VideoFileClip(video_path)
        fps = video.fps
        duration = video.duration
        
        face_positions = []
        
        print(f"Tracking faces in video: {video_path}")
        print(f"Duration: {duration:.2f}s, FPS: {fps:.2f}, Sample rate: {self.sample_rate}s")
        
        # Her sample_rate saniyede bir frame al
        times = np.arange(0, duration, self.sample_rate)
        total_frames = len(times)
        
        for i, t in enumerate(times):
            try:
                frame = video.get_frame(t)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                face_data = self.detect_face_in_frame(frame_bgr)
                face_data["time"] = float(t)
                face_data["frame_number"] = int(t * fps)
                
                face_positions.append(face_data)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{total_frames} frames...")
            except Exception as e:
                print(f"Error processing frame at {t:.2f}s: {e}")
                # Fallback: no face detected
                face_positions.append({
                    "time": float(t),
                    "face_detected": False,
                    "face_center_x": None,
                    "face_center_y": None,
                    "confidence": 0.0
                })
        
        video.close()
        
        detected_count = sum(1 for pos in face_positions if pos.get("face_detected", False))
        print(f"Face tracking completed: {detected_count}/{len(face_positions)} frames with face detected")
        
        return face_positions
    
    def interpolate_face_position(self, 
                                   face_positions: List[Dict], 
                                   target_time: float) -> Dict:
        """
        İki frame arasında yüz pozisyonunu interpolate eder
        (Kalman filter yerine basit linear interpolation)
        """
        if not face_positions:
            return {
                "time": target_time,
                "face_detected": False,
                "face_center_x": None,
                "face_center_y": None,
                "confidence": 0.0
            }
        
        # En yakın iki frame'i bul
        before = None
        after = None
        
        for pos in face_positions:
            if pos["time"] <= target_time:
                before = pos
            elif pos["time"] > target_time:
                after = pos
                break
        
        # Eğer tam zamanında frame varsa
        if before and abs(before["time"] - target_time) < 0.01:
            return before
        
        # Interpolation
        if before and after:
            t1, t2 = before["time"], after["time"]
            alpha = (target_time - t1) / (t2 - t1) if t2 > t1 else 0
            
            if (before.get("face_detected") and after.get("face_detected") and
                before.get("face_center_x") is not None and after.get("face_center_x") is not None):
                return {
                    "time": target_time,
                    "face_detected": True,
                    "face_center_x": before["face_center_x"] * (1 - alpha) + after["face_center_x"] * alpha,
                    "face_center_y": before["face_center_y"] * (1 - alpha) + after["face_center_y"] * alpha,
                    "confidence": (before.get("confidence", 0) + after.get("confidence", 0)) / 2
                }
            elif before.get("face_detected") and before.get("face_center_x") is not None:
                return before
            elif after.get("face_detected") and after.get("face_center_x") is not None:
                return after
        
        # Fallback: en yakın detected face'i kullan
        if before and before.get("face_detected"):
            return before
        if after and after.get("face_detected"):
            return after
        
        # Son fallback: video center (face yok)
        return {
            "time": target_time,
            "face_detected": False,
            "face_center_x": None,
            "face_center_y": None,
            "confidence": 0.0
        }
    
    def save_face_positions(self, face_positions: List[Dict], output_path: str):
        """Face positions'ı JSON olarak kaydeder"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(face_positions, f, indent=2, ensure_ascii=False)
        print(f"Face positions saved to: {output_path}")
    
    def load_face_positions(self, input_path: str) -> List[Dict]:
        """Face positions'ı JSON'dan yükler"""
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)


# Usage example
if __name__ == "__main__":
    tracker = FaceTracker(sample_rate=0.5)
    face_positions = tracker.track_faces_in_video("test_video.mp4")
    tracker.save_face_positions(face_positions, "face_positions.json")



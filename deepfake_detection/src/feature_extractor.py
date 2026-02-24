import numpy as np
import cv2
from typing import List, Dict, Tuple
from face_detector import FaceDetector
from landmark_detector import LandmarkDetector
from eye_analyzer import EyeAnalyzer




# from face_detector import FaceDetector
# from landmark_detector import LandmarkDetector
# from eye_analyzer import EyeAnalyzer

import os

class FeatureExtractor:
    def __init__(self):
        """Initialize feature extractor with all components"""
        self.face_detector = FaceDetector()
        self.landmark_detector = LandmarkDetector()
        self.eye_analyzer = EyeAnalyzer()
    
    def extract_video_features(self, video_path: str) -> Dict:
        """
        Extract features from a video file
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        left_ear_sequence = []
        right_ear_sequence = []
        timestamps = []
        
        print(f"Processing video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps
            
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            largest_face = self.face_detector.get_largest_face(faces)
            
            if largest_face is not None:
                # Extract face region
                x, y, w, h = largest_face
                face_region = frame[y:y+h, x:x+w]
                
                # Detect landmarks
                landmarks = self.landmark_detector.detect_landmarks(face_region)
                
                if landmarks is not None:
                    # Adjust landmarks to original frame coordinates
                    landmarks[:, 0] += x
                    landmarks[:, 1] += y
                    
                    # Get eye landmarks
                    left_eye, right_eye = self.landmark_detector.get_eye_landmarks(landmarks)
                    
                    # Calculate EAR for both eyes
                    left_ear = self.eye_analyzer.calculate_eye_aspect_ratio(left_eye[:6])
                    right_ear = self.eye_analyzer.calculate_eye_aspect_ratio(right_eye[:6])
                    
                    left_ear_sequence.append(left_ear)
                    right_ear_sequence.append(right_ear)
                    timestamps.append(timestamp)
            
            frame_count += 1
            
            # Print progress
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        
        if len(left_ear_sequence) == 0:
            print(f"Warning: No valid frames found in {video_path}")
            return self._get_default_features()
        
        # Calculate features for both eyes
        left_features = self.eye_analyzer.calculate_blink_features(left_ear_sequence, timestamps)
        right_features = self.eye_analyzer.calculate_blink_features(right_ear_sequence, timestamps)
        
        # Combine features
        combined_features = {}
        
        # Add left eye features
        for key, value in left_features.items():
            combined_features[f'left_{key}'] = value
        
        # Add right eye features
        for key, value in right_features.items():
            combined_features[f'right_{key}'] = value
        
        # Add combined features
        avg_ear_sequence = [(l + r) / 2 for l, r in zip(left_ear_sequence, right_ear_sequence)]
        avg_features = self.eye_analyzer.calculate_blink_features(avg_ear_sequence, timestamps)
        
        for key, value in avg_features.items():
            combined_features[f'avg_{key}'] = value
        
        # Add video-level features
        combined_features['video_duration'] = timestamps[-1] if timestamps else 0
        combined_features['total_frames'] = len(timestamps)
        combined_features['fps'] = fps
        
        return combined_features
    
    def _get_default_features(self) -> Dict:
        """Return default features when no face is detected"""
        feature_names = [
            'blink_rate', 'avg_blink_duration', 'std_blink_duration',
            'max_blink_duration', 'min_blink_duration', 'avg_blink_cycle',
            'std_blink_cycle', 'avg_ear', 'std_ear', 'min_ear', 'max_ear',
            'avg_blink_completeness'
        ]
        
        features = {}
        for prefix in ['left_', 'right_', 'avg_']:
            for name in feature_names:
                features[f'{prefix}{name}'] = 0.0
        
        features['video_duration'] = 0
        features['total_frames'] = 0
        features['fps'] = 0
        
        return features
#!/usr/bin/env python3
"""
Simple FastAPI backend for Deepfake Detection - Fixed Version
"""

import os
import sys
import uuid
import tempfile
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

# Global variables
classifier = None
feature_extractor = None
model_loaded = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global classifier, feature_extractor, model_loaded
    try:
        print("Loading ML components...")
        
        # Import here to avoid module loading issues
        from feature_extractor import FeatureExtractor
        from classifier import DeepfakeClassifier
        
        feature_extractor = FeatureExtractor()
        classifier = DeepfakeClassifier()
        
        if os.path.exists("models/classifier.pkl"):
            classifier.load_model("models/classifier.pkl")
            model_loaded = True
            print("✅ Model loaded successfully")
        else:
            print("❌ Model file not found: models/classifier.pkl")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure your ML files are in the src/ folder with correct imports")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    
    yield
    # Shutdown (if needed)
    print("Shutting down...")

app = FastAPI(title="Deepfake Detection API", version="1.0.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    confidence: float
    processing_time: float
    video_info: dict
    features: dict = None
    evidence_frame: str = None
    error: str = None

@app.get("/")
async def root():
    return {
        "message": "Deepfake Detection API", 
        "model_loaded": model_loaded,
        "status": "running"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model_loaded else "unhealthy", 
        "model_loaded": model_loaded
    }

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    temp_path = None
    
    try:
        # Save file
        file_ext = os.path.splitext(file.filename)[1]
        temp_path = os.path.join(temp_dir, f"video{file_ext}")
        
        with open(temp_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Extract features
        import time
        import base64
        import cv2
        from visualization import visualize_blink_detection
        
        start_time = time.time()
        
        # We need to extract features AND get a representative frame
        features = feature_extractor.extract_video_features(temp_path)
        prediction, confidence = classifier.predict(features)
        
        # Capture a representative frame (e.g., first detected blink or middle frame)
        cap = cv2.VideoCapture(temp_path)
        evidence_frame_b64 = None
        
        # Try to find a frame where eyes are relatively closed (low EAR)
        best_vis_frame = None
        max_frames_to_check = 100
        checked = 0
        
        while checked < max_frames_to_check:
            ret, frame = cap.read()
            if not ret: break
            
            # Simple check: try to find a frame with face and landmarks
            faces = feature_extractor.face_detector.detect_faces(frame)
            largest_face = feature_extractor.face_detector.get_largest_face(faces)
            if largest_face is not None:
                x, y, w, h = largest_face
                face_region = frame[y:y+h, x:x+w]
                landmarks = feature_extractor.landmark_detector.detect_landmarks(face_region)
                if landmarks is not None:
                    # Adjust landmarks
                    landmarks[:, 0] += x
                    landmarks[:, 1] += y
                    
                    # Calculate EAR for this specific frame
                    left_eye, right_eye = feature_extractor.landmark_detector.get_eye_landmarks(landmarks)
                    l_ear = feature_extractor.eye_analyzer.calculate_eye_aspect_ratio(left_eye[:6])
                    r_ear = feature_extractor.eye_analyzer.calculate_eye_aspect_ratio(right_eye[:6])
                    
                    # Apply visualization
                    best_vis_frame = visualize_blink_detection(frame, landmarks, l_ear, r_ear)
                    
                    # If this looks like a blink (low EAR), we take it and stop
                    if (l_ear + r_ear) / 2 < 0.28:
                        break
            checked += 1
        cap.release()
        
        if best_vis_frame is not None:
            _, buffer = cv2.imencode('.jpg', best_vis_frame)
            evidence_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        processing_time = time.time() - start_time
        result = "FAKE" if prediction == 1 else "REAL"
        
        return PredictionResponse(
            success=True,
            prediction=result,
            confidence=float(confidence),
            processing_time=processing_time,
            video_info={
                "filename": file.filename,
                "duration": features.get('video_duration', 0),
                "total_frames": features.get('total_frames', 0),
                "fps": features.get('fps', 0)
            },
            features={
                "blink_rate": features.get('avg_blink_rate', 0),
                "avg_blink_duration": features.get('avg_avg_blink_duration', 0),
                "avg_ear": features.get('avg_avg_ear', 0),
                "blink_completeness": features.get('avg_avg_blink_completeness', 0)
            },
            evidence_frame=evidence_frame_b64
        )
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return PredictionResponse(
            success=False,
            prediction="ERROR",
            confidence=0.0,
            processing_time=0.0,
            video_info={},
            error=str(e)
        )
    finally:
        # Cleanup
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Deepfake Detection API...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
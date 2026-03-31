import cv2
import numpy as np

class VideoProcessor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_faces(self, frame):
        """Detect faces in a frame and return cropped face"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            x, y, w, h = faces[0]  # Take the first face
            face = frame[y:y+h, x:x+w]
            return face
        return None
    
    def preprocess_frame(self, frame):
        """Preprocess frame for multimodal model input"""
        # Convert to RGB if needed
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.shape[2] == 3:  # BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model's expected input
        frame = cv2.resize(frame, (128, 128))
        
        # Normalize pixel values
        frame = frame.astype('float32') / 255.0
        
        return frame
    
    def extract_face_features(self, face):
        """Extract features from face for emotion detection"""
        # This can be expanded with more sophisticated feature extraction
        # Currently just returns the preprocessed face
        return self.preprocess_frame(face)
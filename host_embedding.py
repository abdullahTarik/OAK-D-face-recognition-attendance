#!/usr/bin/env python3
"""
Host-side face embedding extraction (fallback when OAK-D embedding not available)
Uses a lightweight embedding model that can run on CPU
"""

import cv2
import numpy as np
import os
import urllib.request
import zipfile
from typing import Optional

# Try to import onnxruntime for running ONNX models
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not available. Install with: pip install onnxruntime")

class HostEmbeddingExtractor:
    """Host-side face embedding extractor"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.session = None
        self.input_size = (112, 112)
        self.embedding_dim = 128
        
        if ONNX_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load embedding model"""
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.session = ort.InferenceSession(self.model_path)
                print(f"Loaded embedding model from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            # Try to use a simple feature extractor as fallback
            print("No embedding model found. Using simple feature extraction.")
            self.session = None
    
    def extract(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract embedding from face image
        Args:
            face_image: BGR face image
        Returns:
            embedding vector (128D) or None
        """
        if face_image is None or face_image.size == 0:
            return None
        
        # Preprocess
        processed = self._preprocess(face_image)
        if processed is None:
            return None
        
        # Extract embedding
        if self.session is not None:
            # Use ONNX model
            try:
                input_name = self.session.get_inputs()[0].name
                output = self.session.run(None, {input_name: processed})
                embedding = output[0].flatten()
                # Normalize embedding
                embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
                return embedding.astype(np.float32)
            except Exception as e:
                print(f"Error running embedding model: {e}")
                return self._simple_feature_extraction(face_image)
        else:
            # Use simple feature extraction
            return self._simple_feature_extraction(face_image)
    
    def _preprocess(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess face image for embedding model"""
        try:
            # Resize to model input size
            resized = cv2.resize(face_image, self.input_size)
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [-1, 1] range (typical for face embedding models)
            normalized = (rgb.astype(np.float32) / 127.5) - 1.0
            
            # Add batch dimension and transpose to NCHW format
            input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]
            
            return input_tensor
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None
    
    def _simple_feature_extraction(self, face_image: np.ndarray) -> np.ndarray:
        """
        Simple feature extraction using histogram and texture features
        This is a fallback when no embedding model is available
        """
        try:
            # Resize to standard size
            resized = cv2.resize(face_image, self.input_size)
            
            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization
            equalized = cv2.equalizeHist(gray)
            
            # Extract features
            features = []
            
            # Histogram features
            hist = cv2.calcHist([equalized], [0], None, [32], [0, 256])
            features.extend(hist.flatten())
            
            # LBP-like texture features (simplified)
            # Divide image into blocks and compute mean/std
            h, w = equalized.shape
            for i in range(0, h, 28):
                for j in range(0, w, 28):
                    block = equalized[i:i+28, j:j+28]
                    if block.size > 0:
                        features.append(np.mean(block))
                        features.append(np.std(block))
            
            # Pad or truncate to embedding_dim
            features = np.array(features[:self.embedding_dim])
            if len(features) < self.embedding_dim:
                features = np.pad(features, (0, self.embedding_dim - len(features)), 'constant')
            
            # Normalize
            features = features / (np.linalg.norm(features) + 1e-10)
            
            return features.astype(np.float32)
        except Exception as e:
            print(f"Error in simple feature extraction: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

def download_mobilefacenet_model(output_dir="models"):
    """Download MobileFaceNet model (placeholder - you'll need actual model URL)"""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "mobilefacenet.onnx")
    
    if os.path.exists(model_path):
        return model_path
    
    print("MobileFaceNet model not found.")
    print("Please download a MobileFaceNet ONNX model and place it in models/ directory")
    print("Or use the simple feature extraction fallback")
    
    return None


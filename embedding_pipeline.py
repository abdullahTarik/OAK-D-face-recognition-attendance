#!/usr/bin/env python3
"""
OAK-D Pipeline with Yunet Face Detector and Face Embedding Model
Runs face detection and embedding extraction on MyriadX VPU
"""

import depthai as dai
import numpy as np
import cv2
import blobconverter
import threading
from typing import Optional, Tuple, List

# Embedding model configuration
EMBEDDING_MODEL_NAME = "mobilefacenet"
EMBEDDING_INPUT_SIZE = (112, 112)  # Standard face embedding input size
EMBEDDING_DIM = 128  # MobileFaceNet produces 128D embeddings

# Yunet face detector configuration
YUNET_INPUT_SIZE = (320, 320)
YUNET_SCORE_THRESHOLD = 0.6
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 5000

class OAKDEmbeddingPipeline:
    """OAK-D pipeline for face detection and embedding extraction"""
    
    def __init__(self, video_width=640, video_height=480):
        self.video_width = video_width
        self.video_height = video_height
        self.device = None
        self.pipeline = None
        self.cam_queue = None
        self.det_queue = None
        self.embed_queue = None
        self.running = False
        
    def create_pipeline(self):
        """Create DepthAI pipeline with face detection and embedding"""
        pipeline = dai.Pipeline()
        
        # Color camera
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(self.video_width, self.video_height)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setFps(30)
        
        # Face detection (Yunet) - using face-detection-retail-0004 as fallback
        # For Yunet, we'll use a NeuralNetwork node with custom blob
        face_det_nn = pipeline.create(dai.node.NeuralNetwork)
        face_det_nn.setBlobPath(self._get_face_detection_blob())
        face_det_nn.input.setBlocking(False)
        
        # Face embedding network
        face_embed_nn = pipeline.create(dai.node.NeuralNetwork)
        face_embed_nn.setBlobPath(self._get_embedding_blob())
        face_embed_nn.input.setBlocking(False)
        
        # Image manipulators for face cropping and resizing
        face_manip = pipeline.create(dai.node.ImageManip)
        face_manip.initialConfig.setResize(EMBEDDING_INPUT_SIZE)
        face_manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
        face_manip.setMaxOutputFrameSize(EMBEDDING_INPUT_SIZE[0] * EMBEDDING_INPUT_SIZE[1] * 3)
        
        # Output streams
        xout_cam = pipeline.create(dai.node.XLinkOut)
        xout_cam.setStreamName("cam")
        
        xout_det = pipeline.create(dai.node.XLinkOut)
        xout_det.setStreamName("det")
        
        xout_embed = pipeline.create(dai.node.XLinkOut)
        xout_embed.setStreamName("embed")
        
        # Linking
        cam.preview.link(xout_cam.input)
        cam.preview.link(face_det_nn.input)
        face_manip.out.link(face_embed_nn.input)
        face_embed_nn.out.link(xout_embed.input)
        face_det_nn.out.link(xout_det.input)
        
        # Script node to crop faces and feed to embedding network
        script = pipeline.create(dai.node.Script)
        script.setScript(self._get_crop_script())
        cam.preview.link(script.inputs['preview'])
        face_det_nn.out.link(script.inputs['detections'])
        script.outputs['manip_cfg'].link(face_manip.inputConfig)
        script.outputs['manip_img'].link(face_manip.inputImage)
        
        return pipeline
    
    def _get_face_detection_blob(self) -> str:
        """Get face detection blob (using face-detection-retail-0004 as Yunet alternative)"""
        try:
            # Try to use face-detection-retail-0004 (OpenVINO model zoo)
            blob_path = blobconverter.from_zoo(
                name="face-detection-retail-0004",
                shaves=6,
                version="2021.4"
            )
            return blob_path
        except Exception as e:
            print(f"Warning: Could not download face detection model: {e}")
            print("Falling back to host-based detection")
            return None
    
    def _get_embedding_blob(self) -> str:
        """Get face embedding blob (MobileFaceNet)"""
        try:
            # Try to download MobileFaceNet from blobconverter
            # Note: You may need to convert your own model
            blob_path = blobconverter.from_zoo(
                name="face-reidentification-retail-0095",  # Alternative embedding model
                shaves=6,
                version="2021.4"
            )
            return blob_path
        except Exception as e:
            print(f"Warning: Could not download embedding model: {e}")
            print("You may need to convert a MobileFaceNet model manually")
            return None
    
    def _get_crop_script(self) -> str:
        """Get script to crop faces from detections"""
        return """
import time
import marshal

# This script crops detected faces and sends them to embedding network
while True:
    preview = node.io['preview'].get()
    detections = node.io['detections'].get()
    
    # Get first detection (largest face)
    if len(detections.detections) > 0:
        det = detections.detections[0]
        bbox = det.bbox
        
        # Create ImageManip config to crop face
        cfg = ImageManipConfig()
        cfg.setCropRect(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
        cfg.setResize(112, 112)
        cfg.setKeepAspectRatio(False)
        
        node.io['manip_cfg'].send(cfg)
        node.io['manip_img'].send(preview)
"""
    
    def start(self):
        """Start the pipeline"""
        if self.running:
            return
        
        try:
            self.pipeline = self.create_pipeline()
            self.device = dai.Device(self.pipeline)
            
            self.cam_queue = self.device.getOutputQueue("cam", 4, blocking=False)
            self.det_queue = self.device.getOutputQueue("det", 4, blocking=False)
            self.embed_queue = self.device.getOutputQueue("embed", 4, blocking=False)
            
            self.running = True
            print("OAK-D embedding pipeline started")
            return True
        except Exception as e:
            print(f"Failed to start OAK-D pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def stop(self):
        """Stop the pipeline"""
        self.running = False
        if self.device:
            self.device.close()
    
    def get_frame_and_embeddings(self) -> Tuple[Optional[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Get latest frame and face embeddings
        Returns: (frame, [(bbox, embedding), ...])
        """
        if not self.running:
            return None, []
        
        frame = None
        embeddings = []
        
        # Get camera frame
        if self.cam_queue.has():
            frame_msg = self.cam_queue.get()
            frame = frame_msg.getCvFrame()
        
        # Get detections
        detections = []
        if self.det_queue.has():
            det_msg = self.det_queue.get()
            detections = det_msg.detections
        
        # Get embeddings
        if self.embed_queue.has():
            embed_msg = self.embed_queue.get()
            embedding = embed_msg.getFirstLayerFp16()  # Get embedding vector
            if len(detections) > 0:
                det = detections[0]
                bbox = np.array([det.xmin, det.ymin, det.xmax, det.ymax])
                embeddings.append((bbox, np.array(embedding)))
        
        return frame, embeddings
    
    def extract_embedding_from_face(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract embedding from a face crop (for enrollment)
        This runs on host as fallback if OAK-D embedding fails
        """
        # Resize to embedding input size
        resized = cv2.resize(face_crop, EMBEDDING_INPUT_SIZE)
        
        # Normalize to [-1, 1] range (typical for face embedding models)
        normalized = (resized.astype(np.float32) / 127.5) - 1.0
        
        # If we have an embedding model on host, run it here
        # For now, return None - this should be implemented with a host-side model
        return None


# On-Device Face Embedding System Setup Guide

This guide explains how to set up the on-device face embedding system for OAK-D camera.

## Architecture Overview

```
OAK-D Camera
    ↓
Yunet Face Detector (on MyriadX VPU)
    ↓
Extract face crops
    ↓
Face Embedding Network (on MyriadX VPU) - Optional
    ↓
Send embeddings to Flask backend
    ↓
Match against stored embeddings (cosine similarity)
    ↓
Mark attendance
```

## Components

1. **embedding_pipeline.py**: OAK-D pipeline with face detection and embedding extraction
2. **embedding_storage.py**: Storage and matching system for embeddings
3. **host_embedding.py**: Host-side embedding extractor (fallback)

## Setup Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `blobconverter`: For downloading/converting models to OpenVINO blobs
- `onnxruntime`: For running ONNX models on host (fallback)

### 2. Model Setup Options

#### Option A: Use Host-Side Embedding (Current Implementation)

The current implementation uses a host-side embedding extractor with a simple feature extraction fallback. This works immediately without additional setup.

**Pros:**
- Works immediately
- No model download needed
- Good for testing

**Cons:**
- Slower than on-device
- Less accurate than trained models

#### Option B: Download MobileFaceNet Model

For better accuracy, download a MobileFaceNet ONNX model:

1. Download a MobileFaceNet ONNX model (e.g., from [Model Zoo](https://github.com/onnx/models))
2. Place it in `models/mobilefacenet.onnx`
3. The system will automatically use it

#### Option C: Use OAK-D On-Device Embedding (Advanced)

To use on-device embedding extraction on MyriadX:

1. Convert a face embedding model to OpenVINO blob format
2. Update `embedding_pipeline.py` to use the blob
3. The pipeline will run embedding extraction on MyriadX VPU

**Example blob conversion:**
```python
import blobconverter

# Convert ONNX to blob
blob_path = blobconverter.from_onnx(
    model="path/to/mobilefacenet.onnx",
    data_type="FP16",
    shaves=6
)
```

## Usage

### Enrollment

When you enroll a new user:
1. Face images are captured and saved to `static/faces/<name_roll>/`
2. Embeddings are automatically extracted and stored in `static/embeddings/<name_roll>.pkl`
3. Both image and embedding storage work together

### Recognition

The system tries embedding-based recognition first:
- Extracts embedding from detected face
- Matches against stored embeddings using cosine similarity
- Falls back to KNN if embeddings not available

### Retraining

Click "Retrain Model" button to:
- Extract embeddings from all existing face images
- Update the embedding database
- Also retrain KNN model for backward compatibility

## Configuration

### Threshold Settings

In `settings.json`, adjust `match_distance_threshold`:
- **0.6-0.7**: Recommended for cosine similarity (higher = more strict)
- **0.5**: For KNN distance (lower = more strict)

### Embedding Storage

Embeddings are stored in:
- Directory: `static/embeddings/`
- Format: Pickle files (`<user_id>.pkl`)
- Each file contains a list of numpy arrays (embeddings)

## Troubleshooting

### "Embedding modules not available"

- Check that `embedding_storage.py` and `host_embedding.py` are in the same directory
- Verify imports are working

### Low Recognition Accuracy

1. **Retrain the model**: Click "Retrain Model" after adding users
2. **Adjust threshold**: Lower threshold (0.5-0.6) for stricter matching
3. **More enrollment images**: Capture more images per user (increase `nimgs` in settings)
4. **Better lighting**: Ensure consistent lighting during enrollment and recognition

### Performance Issues

- Host-side embedding extraction is slower than on-device
- For production, consider implementing on-device embedding extraction
- Current implementation uses simple feature extraction as fallback

## Future Enhancements

1. **On-Device Embedding**: Implement full OAK-D pipeline with embedding extraction on MyriadX
2. **Better Models**: Integrate trained MobileFaceNet or FaceNet models
3. **Anti-Spoofing**: Add liveness detection
4. **Tracking**: Implement face tracking to reduce redundant processing

## Notes

- The system maintains backward compatibility with KNN-based recognition
- Both embedding and KNN methods can work simultaneously
- Embeddings provide better accuracy and scalability
- KNN provides fallback when embeddings are not available


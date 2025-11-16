#!/usr/bin/env python3
"""Quick test script to debug recognition issues"""

import cv2
import numpy as np
from embedding_storage import EmbeddingStorage
from host_embedding import HostEmbeddingExtractor
import os

print("=== Recognition System Test ===\n")

# Initialize systems
print("1. Initializing embedding system...")
storage = EmbeddingStorage()
extractor = HostEmbeddingExtractor()

print(f"   - Registered users: {storage.get_user_count()}")
for user_id in storage.get_all_users():
    embeddings = storage.get_user_embeddings(user_id)
    print(f"   - {user_id}: {len(embeddings)} embeddings")

# Test with a sample face image
print("\n2. Testing recognition...")
test_user = storage.get_all_users()[0] if storage.get_user_count() > 0 else None

if test_user:
    print(f"   - Testing with user: {test_user}")
    user_path = f"static/faces/{test_user}"
    if os.path.exists(user_path):
        test_img_path = os.path.join(user_path, "0.jpg")
        if os.path.exists(test_img_path):
            print(f"   - Loading test image: {test_img_path}")
            test_img = cv2.imread(test_img_path)
            if test_img is not None:
                # Flip to match recognition
                flipped = cv2.flip(test_img, 1)
                print(f"   - Image shape: {flipped.shape}")
                
                # Extract embedding
                print("   - Extracting embedding...")
                embedding = extractor.extract(flipped)
                if embedding is not None:
                    print(f"   - Embedding shape: {embedding.shape}")
                    print(f"   - Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
                    
                    # Test matching
                    print("   - Testing matching...")
                    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
                        user_id, similarity = storage.match_embedding(embedding, threshold)
                        print(f"     Threshold {threshold}: {user_id} (similarity={similarity:.4f})")
                else:
                    print("   - ERROR: Failed to extract embedding!")
            else:
                print(f"   - ERROR: Could not load image!")
        else:
            print(f"   - ERROR: Test image not found!")
    else:
        print(f"   - ERROR: User folder not found!")
else:
    print("   - ERROR: No users registered!")

print("\n=== Test Complete ===")


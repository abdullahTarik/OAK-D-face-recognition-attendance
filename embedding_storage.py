#!/usr/bin/env python3
"""
Embedding storage and matching system
Stores face embeddings for each user and provides matching functionality
"""

import os
import json
import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class EmbeddingStorage:
    """Manages storage and matching of face embeddings"""
    
    def __init__(self, storage_dir="static/embeddings"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.embeddings_db = {}  # {user_id: [embedding1, embedding2, ...]}
        self.load_all_embeddings()
    
    def load_all_embeddings(self):
        """Load all embeddings from storage"""
        self.embeddings_db = {}
        
        if not os.path.exists(self.storage_dir):
            return
        
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.pkl'):
                user_id = filename[:-4]  # Remove .pkl extension
                filepath = os.path.join(self.storage_dir, filename)
                try:
                    with open(filepath, 'rb') as f:
                        embeddings = pickle.load(f)
                    self.embeddings_db[user_id] = embeddings
                    print(f"Loaded {len(embeddings)} embeddings for {user_id}")
                except Exception as e:
                    print(f"Error loading embeddings for {user_id}: {e}")
    
    def save_embeddings(self, user_id: str, embeddings: List[np.ndarray]):
        """Save embeddings for a user"""
        filepath = os.path.join(self.storage_dir, f"{user_id}.pkl")
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(embeddings, f)
            self.embeddings_db[user_id] = embeddings
            print(f"Saved {len(embeddings)} embeddings for {user_id}")
            return True
        except Exception as e:
            print(f"Error saving embeddings for {user_id}: {e}")
            return False
    
    def add_embedding(self, user_id: str, embedding: np.ndarray):
        """Add a single embedding for a user"""
        if user_id not in self.embeddings_db:
            self.embeddings_db[user_id] = []
        
        self.embeddings_db[user_id].append(embedding)
        self.save_embeddings(user_id, self.embeddings_db[user_id])
    
    def get_user_embeddings(self, user_id: str) -> List[np.ndarray]:
        """Get all embeddings for a user"""
        return self.embeddings_db.get(user_id, [])
    
    def delete_user(self, user_id: str):
        """Delete all embeddings for a user"""
        if user_id in self.embeddings_db:
            del self.embeddings_db[user_id]
        
        filepath = os.path.join(self.storage_dir, f"{user_id}.pkl")
        if os.path.exists(filepath):
            os.remove(filepath)
    
    def match_embedding(self, query_embedding: np.ndarray, threshold: float = 0.6) -> Tuple[Optional[str], float]:
        """
        Match a query embedding against all stored embeddings
        Returns: (user_id, similarity_score) or (None, 0.0) if no match
        """
        if len(self.embeddings_db) == 0:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        query_embedding = query_embedding.reshape(1, -1)
        
        for user_id, embeddings in self.embeddings_db.items():
            if len(embeddings) == 0:
                continue
            
            # Convert to numpy array
            user_embeddings = np.array(embeddings)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, user_embeddings)[0]
            max_similarity = float(np.max(similarities))
            
            if max_similarity > best_score:
                best_score = max_similarity
                best_match = user_id
        
        if best_score >= threshold:
            return best_match, best_score
        else:
            return None, best_score
    
    def get_all_users(self) -> List[str]:
        """Get list of all user IDs"""
        return list(self.embeddings_db.keys())
    
    def get_user_count(self) -> int:
        """Get total number of registered users"""
        return len(self.embeddings_db)


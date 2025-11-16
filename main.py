#!/usr/bin/env python3

"""
Flask + OAK-D attendance app with on-device face embedding extraction.
Uses OAK-D MyriadX VPU for face detection and embedding extraction.
Stores embeddings for matching instead of raw images.
"""

import os
import time
import warnings
import json
import cv2
import depthai as dai
import numpy as np
import joblib
import pandas as pd
import threading
from datetime import date, datetime
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_from_directory

# Import embedding system
try:
    from embedding_storage import EmbeddingStorage
    from host_embedding import HostEmbeddingExtractor
    EMBEDDING_AVAILABLE = True
except ImportError:
    print("Warning: Embedding modules not available. Using fallback KNN method.")
    EMBEDDING_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Global variables for camera and auto attendance
camera_device = None
camera_queue = None
latest_frame = None
camera_lock = threading.Lock()
auto_attendance_enabled = False
auto_attendance_thread = None

# Global variables for recognition status (for web feed display)
recognition_active = False
recognition_status = {"recognized": 0, "message": "", "color": (0, 255, 0)}
recognition_lock = threading.Lock()

# ---------------- CONFIG (Default values) ----------------
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
FACE_SIZE = (50, 50)

# Configurable settings (will be loaded from file)
_settings = {
    "auto_attendance_interval_minutes": 15,
    "nimgs": 10,
    "pipeline_timeout": 30,
    "stable_time": 0.4,
    "max_center_movement": 15.0,
    "match_distance_threshold": 0.5
}

# Settings file path
SETTINGS_FILE = "settings.json"

def load_settings():
    """Load settings from file or use defaults."""
    global _settings
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                loaded = json.load(f)
                _settings.update(loaded)
            print("Settings loaded from file")
        except Exception as e:
            print(f"Error loading settings: {e}, using defaults")
    return _settings

def save_settings():
    """Save current settings to file."""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(_settings, f, indent=4)
        print("Settings saved to file")
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

def get_setting(key, default=None):
    """Get a setting value."""
    return _settings.get(key, default)

def set_setting(key, value):
    """Set a setting value."""
    _settings[key] = value
    save_settings()

# Load settings on startup
load_settings()

# Apply settings to global variables
NIMGS = _settings["nimgs"]
PIPELINE_TIMEOUT = _settings["pipeline_timeout"]
STABLE_TIME = _settings["stable_time"]
MAX_CENTER_MOVEMENT = _settings["max_center_movement"]
MATCH_DISTANCE_THRESHOLD = _settings["match_distance_threshold"]
auto_attendance_interval = _settings["auto_attendance_interval_minutes"] * 60

# Global variables for enrollment status
enrollment_active = False
enrollment_status = {"captured": 0, "total": NIMGS, "message": "", "color": (0, 255, 255)}
enrollment_lock = threading.Lock()

# Paths
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

os.makedirs("static/faces", exist_ok=True)
os.makedirs("Attendance", exist_ok=True)
attendance_csv = f"Attendance/Attendance-{datetoday}.csv"

if not os.path.exists(attendance_csv):
    with open(attendance_csv, "w") as f:
        f.write("Name,Roll,Time\n")

# Load Haar Cascade classifier
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    print("Warning: Could not load Haar Cascade. Trying alternative path...")
    # Try downloading or using local file
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise RuntimeError("Could not load Haar Cascade classifier. Please ensure OpenCV is properly installed.")

# Cached recognition model (for backward compatibility)
_recognition_model = None

# Embedding storage system
_embedding_storage = None
_host_embedding_extractor = None

def init_embedding_system():
    """Initialize embedding storage and extractor"""
    global _embedding_storage, _host_embedding_extractor
    
    if EMBEDDING_AVAILABLE:
        _embedding_storage = EmbeddingStorage()
        _host_embedding_extractor = HostEmbeddingExtractor()
        print(f"Embedding system initialized. Registered users: {_embedding_storage.get_user_count()}")
    else:
        print("Embedding system not available. Using KNN fallback.")

def load_recognition_model():
    global _recognition_model
    if _recognition_model is None:
        path = "static/face_recognition_model.pkl"
        if os.path.exists(path):
            try:
                _recognition_model = joblib.load(path)
                # Verify model has training data
                if hasattr(_recognition_model, 'classes_'):
                    print(f"Recognition model loaded with {len(_recognition_model.classes_)} classes.")
                else:
                    print("Recognition model loaded (no class info available).")
            except Exception as e:
                print(f"Failed to load recognition model: {e}")
                import traceback
                traceback.print_exc()
                _recognition_model = None
        else:
            print(f"Recognition model file not found at {path}")
    return _recognition_model

def clear_recognition_model_cache():
    global _recognition_model
    _recognition_model = None

# ---------------- Utilities ----------------
def totalreg():
    return len(os.listdir("static/faces"))

def train_model():
    """
    Train model - extracts embeddings from existing face images and stores them.
    Also trains KNN model for backward compatibility.
    """
    # Extract embeddings from existing face images
    if EMBEDDING_AVAILABLE and _embedding_storage is not None and _host_embedding_extractor is not None:
        print("train_model: Extracting embeddings from face images...")
        userlist = os.listdir("static/faces")
        if len(userlist) > 0:
            for user in userlist:
                user_path = f"static/faces/{user}"
                if not os.path.isdir(user_path):
                    continue
                
                embeddings = []
                img_count = 0
                for imgname in os.listdir(user_path):
                    img_path = os.path.join(user_path, imgname)
                    img = cv2.imread(img_path)
                    if img is None or img.size == 0:
                        continue
                    
                    try:
                        # Flip image horizontally to match recognition
                        flipped_img = cv2.flip(img, 1)
                        # Extract embedding
                        embedding = _host_embedding_extractor.extract(flipped_img)
                        if embedding is not None:
                            embeddings.append(embedding)
                            img_count += 1
                    except Exception as e:
                        print(f"train_model: Error processing {img_path}: {e}")
                        continue
                
                if len(embeddings) > 0:
                    _embedding_storage.save_embeddings(user, embeddings)
                    print(f"train_model: Extracted {len(embeddings)} embeddings for {user}")
    
    # Also train KNN model for backward compatibility
    faces = []
    labels = []
    userlist = os.listdir("static/faces")
    if len(userlist) == 0:
        print("train_model: No users found in static/faces")
        return False
    
    print(f"train_model: Found {len(userlist)} users")
    for user in userlist:
        user_path = f"static/faces/{user}"
        if not os.path.isdir(user_path):
            continue
        img_count = 0
        for imgname in os.listdir(user_path):
            img_path = os.path.join(user_path, imgname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"train_model: Failed to load {img_path}")
                continue
            if img.size == 0:
                print(f"train_model: Empty image {img_path}")
                continue
            try:
                # Flip image horizontally to match recognition (which uses flipped frames)
                flipped_img = cv2.flip(img, 1)
                # Convert to grayscale for better consistency
                if len(flipped_img.shape) == 3:
                    gray_face = cv2.cvtColor(flipped_img, cv2.COLOR_BGR2GRAY)
                else:
                    gray_face = flipped_img
                # Apply histogram equalization for better lighting normalization
                equalized = cv2.equalizeHist(gray_face)
                resized_face = cv2.resize(equalized, FACE_SIZE)
                if resized_face is None or resized_face.size == 0:
                    print(f"train_model: Failed to resize {img_path}")
                    continue
                # Normalize to 0-1 range for better numerical stability
                normalized = resized_face.astype(np.float32) / 255.0
                faces.append(normalized.ravel())
                labels.append(user)
                img_count += 1
            except Exception as e:
                print(f"train_model: Error processing {img_path}: {e}")
                continue
        print(f"train_model: Loaded {img_count} images for user {user}")
    
    if len(faces) == 0:
        print("train_model: No valid face images found")
        return False
    
    print(f"train_model: Training with {len(faces)} face images from {len(np.unique(labels))} users")
    faces = np.array(faces)
    n_neighbors = min(5, len(np.unique(labels)))
    # Use cosine distance for better performance with normalized features
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine', weights='distance')
    knn.fit(faces, labels)
    joblib.dump(knn, "static/face_recognition_model.pkl")
    clear_recognition_model_cache()
    print(f"train_model: Trained and saved recognition model with {n_neighbors} neighbors")
    return True

def add_attendance(name_roll):
    """
    Add attendance record. Returns True if added successfully.
    Note: The calling function should check if user was already marked in this session.
    """
    print(f"[add_attendance] Called with: {name_roll}")
    
    # First, verify user still exists (not deleted)
    user_folder = f"static/faces/{name_roll}"
    if not os.path.exists(user_folder):
        print(f"[add_attendance] ERROR: User {name_roll} does not exist (may have been deleted)")
        return False
    
    try:
        username, userid = name_roll.split("_", 1)
        print(f"[add_attendance] Parsed: username={username}, userid={userid}")
    except Exception as e:
        print(f"[add_attendance] ERROR: Could not parse name_roll '{name_roll}': {e}")
        return False
    
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[add_attendance] Current time: {current_time}")
    
    # Check if CSV exists
    if not os.path.exists(attendance_csv):
        print(f"[add_attendance] Creating new CSV: {attendance_csv}")
        with open(attendance_csv, "w") as f:
            f.write("Name,Roll,Time\n")
    
    # Add new entry
    try:
        with open(attendance_csv, "a") as f:
            f.write(f"{username},{userid},{current_time}\n")
        print(f"[add_attendance] ✓✓✓ SUCCESS: Attendance saved: {username} ({userid}) at {current_time} ✓✓✓")
        return True
    except Exception as e:
        print(f"[add_attendance] ERROR: Failed to write to CSV: {e}")
        return False

def identify_face(face, embedding=None):
    """
    Identify face using embeddings (preferred) or KNN fallback.
    Args:
        face: Face image (BGR)
        embedding: Pre-computed embedding (optional, from OAK-D)
    Returns: (name_roll, score) where score is:
        - For embeddings: similarity (0-1, higher is better)
        - For KNN: distance (lower is better, but we'll convert to similarity-like)
    """
    # Try embedding-based recognition first
    if EMBEDDING_AVAILABLE and _embedding_storage is not None:
        try:
            # Extract embedding if not provided
            if embedding is None:
                if _host_embedding_extractor is None:
                    print("identify_face: Host embedding extractor not available")
                    # Fall through to KNN
                else:
                    embedding = _host_embedding_extractor.extract(face)
                    if embedding is None:
                        print("identify_face: Failed to extract embedding, trying KNN")
                        # Fall through to KNN
                    else:
                        # Match against stored embeddings
                        threshold = get_setting("match_distance_threshold", 0.6)  # Cosine similarity threshold
                        user_id, similarity = _embedding_storage.match_embedding(embedding, threshold)
                        
                        if user_id:
                            # Verify user still exists (not deleted)
                            user_folder = f"static/faces/{user_id}"
                            if not os.path.exists(user_folder):
                                print(f"identify_face: User {user_id} was deleted, skipping match")
                                return None, similarity
                            
                            print(f"identify_face: Matched {user_id} with similarity={similarity:.4f} (threshold={threshold})")
                            return user_id, similarity
                        else:
                            print(f"identify_face: No match (best similarity={similarity:.4f} < threshold={threshold})")
                            return None, similarity
        except Exception as e:
            print(f"identify_face embedding error: {e}")
            import traceback
            traceback.print_exc()
            # Fall through to KNN
    
    # Fallback to KNN method
    model = load_recognition_model()
    if model is None:
        print("identify_face: No model available")
        return None, None
    
    # Validate face ROI
    if face is None or face.size == 0:
        print("identify_face: Empty face ROI")
        return None, None
    
    if face.shape[0] < 10 or face.shape[1] < 10:
        print(f"identify_face: Face ROI too small: {face.shape}")
        return None, None
    
    try:
        # Convert to grayscale if needed (same as training)
        if len(face.shape) == 3:
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face
        
        # Apply histogram equalization for better lighting normalization (same as training)
        equalized = cv2.equalizeHist(gray_face)
        
        # Resize face to match training size
        resized = cv2.resize(equalized, FACE_SIZE)
        if resized is None or resized.size == 0:
            print("identify_face: Failed to resize face")
            return None, None
        
        # Normalize to 0-1 range (same as training)
        normalized = resized.astype(np.float32) / 255.0
        
        # Flatten to vector (same as training)
        vec = normalized.ravel().reshape(1, -1)
        
        # Get prediction and distance
        distances, indices = model.kneighbors(vec, n_neighbors=1)
        dist = float(distances[0][0])
        pred = model.predict(vec)[0]
        
        # Verify user still exists (not deleted)
        user_folder = f"static/faces/{pred}"
        if not os.path.exists(user_folder):
            print(f"identify_face (KNN): User {pred} was deleted, skipping match")
            return None, None
        
        # Convert distance to similarity-like score (1 - normalized_distance)
        # For cosine distance, max is 2.0, so similarity = 1 - (dist/2.0)
        # For other metrics, we'll use a simple conversion
        max_possible_dist = 2.0  # For cosine distance
        similarity_like = max(0.0, 1.0 - (dist / max_possible_dist))
        
        # Always print for debugging (helps diagnose recognition issues)
        threshold = get_setting("match_distance_threshold", 0.5)
        print(f"identify_face (KNN): Prediction={pred}, Distance={dist:.4f}, Similarity-like={similarity_like:.4f}, Threshold={threshold:.4f}")
        # Return similarity-like score for consistent comparison
        return pred, similarity_like
    except Exception as e:
        print(f"identify_face error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def extract_attendance():
    """Extract attendance information from today's CSV"""
    try:
        df = pd.read_csv(attendance_csv)
        names = df['Name'].tolist() if 'Name' in df.columns else []
        rolls = df['Roll'].tolist() if 'Roll' in df.columns else []
        times = df['Time'].tolist() if 'Time' in df.columns else []
        l = len(df)
        return names, rolls, times, l
    except Exception:
        return [], [], [], 0

def getallusers():
    """Get all registered users"""
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    for i in userlist:
        if '_' in i:
            name, roll = i.split('_', 1)
            names.append(name)
            rolls.append(roll)
    l = len(userlist)
    return userlist, names, rolls, l

def deletefolder(duser):
    """Delete a user folder"""
    if os.path.exists(duser):
        pics = os.listdir(duser)
        for i in pics:
            os.remove(os.path.join(duser, i))
        os.rmdir(duser)

def detect_faces_haar(frame):
    """Detect faces using Haar Cascade. Returns list of (x, y, w, h) rectangles."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def center_distance(box1, box2):
    """Calculate distance between centers of two bounding boxes."""
    if box1 is None or box2 is None:
        return float('inf')
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    center1 = (x1 + w1/2, y1 + h1/2)
    center2 = (x2 + w2/2, y2 + h2/2)
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

# ---------------- Pipeline creation (simplified - just camera) ----------------
def create_pipeline():
    """Create simple pipeline with just camera output."""
    pipeline = dai.Pipeline()
    
    # Color camera
    try:
        cam = pipeline.create(dai.node.ColorCamera)
    except Exception:
        cam = pipeline.create(dai.node.Camera)
    
    try:
        cam.setPreviewSize(VIDEO_WIDTH, VIDEO_HEIGHT)
    except Exception:
        pass
    try:
        cam.setInterleaved(False)
    except Exception:
        pass
    try:
        cam.setFps(30)
    except Exception:
        pass
    try:
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    except Exception:
        pass
    
    # Output
    xout_cam = pipeline.create(dai.node.XLinkOut)
    xout_cam.setStreamName("cam")
    
    # Link camera to output
    try:
        cam.preview.link(xout_cam.input)
    except Exception:
        try:
            cam.video.link(xout_cam.input)
        except Exception:
            try:
                cam.out.link(xout_cam.input)
            except Exception:
                pass
    
    return pipeline

# ---------------- Continuous Camera Feed ----------------
def camera_feed_thread():
    """Background thread to continuously capture camera frames."""
    global camera_device, camera_queue, latest_frame
    
    try:
        pipeline = create_pipeline()
        camera_device = dai.Device(pipeline)
        camera_queue = camera_device.getOutputQueue("cam", 4, blocking=True)
        
        print("Camera feed thread started - camera is now running continuously")
        time.sleep(1.0)  # Wait for initialization
        
        while True:
            try:
                in_frame = camera_queue.get()
                if in_frame is not None:
                    frame = in_frame.getCvFrame()
                    with camera_lock:
                        latest_frame = frame.copy()
            except Exception as e:
                print(f"Error in camera feed thread: {e}")
                time.sleep(0.1)
    except Exception as e:
        print(f"Failed to start camera feed: {e}")
        import traceback
        traceback.print_exc()

def get_latest_frame():
    """Get the latest camera frame (horizontally flipped)."""
    with camera_lock:
        if latest_frame is not None:
            # Flip frame horizontally (mirror effect)
            flipped_frame = cv2.flip(latest_frame, 1)
            return flipped_frame
    return None

# ---------------- Automatic Attendance ----------------
def auto_attendance_worker():
    """Background thread for automatic attendance."""
    global auto_attendance_enabled
    
    while True:
        if auto_attendance_enabled:
            interval_minutes = get_setting("auto_attendance_interval_minutes", 15)
            interval_seconds = interval_minutes * 60
            print(f"Auto attendance: Waiting {interval_minutes} minutes...")
            time.sleep(interval_seconds)
            
            if auto_attendance_enabled:
                print("Auto attendance: Taking attendance now...")
                take_attendance_automatic()
        else:
            time.sleep(5)  # Check every 5 seconds if enabled

def take_attendance_automatic():
    """Take attendance automatically without opening a window."""
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        print("Auto attendance: No trained model available")
        return
    
    recognized = set()
    prev_box = None
    stable_start = None
    start_time = time.time()
    recognition_timeout = 60  # 1 minute to recognize faces
    
    print("Auto attendance: Starting recognition...")
    
    while (time.time() - start_time) < recognition_timeout:
        frame = get_latest_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Detect faces using Haar Cascade
        faces = detect_faces_haar(frame)
        
        if len(faces) == 0:
            prev_box = None
            stable_start = None
            continue
        
        # Use the first detected face
        x, y, w, h = faces[0]
        current_box = (x, y, w, h)
        center_dist = center_distance(prev_box, current_box)
        now = time.time()
        
        max_movement = get_setting("max_center_movement", 15.0)
        stable_time = get_setting("stable_time", 0.4)
        match_threshold = get_setting("match_distance_threshold", 0.5)
        
        if prev_box is None:
            prev_box = current_box
            stable_start = now
        else:
            if center_dist <= max_movement:
                if stable_start is None:
                    stable_start = now
                elapsed = now - stable_start
                if elapsed >= stable_time:
                    # Stable - recognize
                    # Ensure bounds are within frame
                    y1 = max(0, y)
                    x1 = max(0, x)
                    y2 = min(frame.shape[0], y + h)
                    x2 = min(frame.shape[1], x + w)
                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size > 0:
                        pred, score = identify_face(face_roi)
                        if pred is not None and score is not None:
                            # Verify user still exists (not deleted)
                            user_folder = f"static/faces/{pred}"
                            if not os.path.exists(user_folder):
                                print(f"Auto attendance: User {pred} was deleted, skipping")
                                continue
                            
                            # score is similarity (higher is better)
                            if score >= match_threshold:
                                # Always add attendance (allow multiple entries)
                                if add_attendance(pred):
                                    if pred not in recognized:
                                        recognized.add(pred)
                                    print(f"Auto attendance: ✓ {pred} (similarity={score:.4f})")
                            else:
                                print(f"Auto attendance: Similarity too low: {pred} (similarity={score:.4f} < {match_threshold})")
                    time.sleep(0.5)
                    prev_box = None
                    stable_start = None
            else:
                prev_box = current_box
                stable_start = now
    
    print(f"Auto attendance: Completed. Recognized {len(recognized)} person(s)")

# ---------------- FLASK ROUTES ----------------
@app.route("/toggle_auto_attendance")
def toggle_auto_attendance():
    """Toggle automatic attendance on/off."""
    global auto_attendance_enabled, auto_attendance_thread
    
    auto_attendance_enabled = not auto_attendance_enabled
    
    if auto_attendance_enabled and auto_attendance_thread is None:
        auto_attendance_thread = threading.Thread(target=auto_attendance_worker, daemon=True)
        auto_attendance_thread.start()
        print(f"Auto attendance enabled - will take attendance every {auto_attendance_interval/60} minutes")
    elif not auto_attendance_enabled:
        print("Auto attendance disabled")
    
    return redirect(url_for("home"))

@app.route("/")
def home():
    """Serve React app or old template based on query param"""
    use_old = request.args.get('old') == '1'
    if use_old or not os.path.exists('index.html'):
        # Fallback to old template
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, auto_attendance=auto_attendance_enabled)
    # Serve React app
    return send_from_directory('.', 'index.html')

@app.route("/api/status")
def api_status():
    """API endpoint to get current system status and data"""
    names, rolls, times, l = extract_attendance()
    userlist, user_names, user_rolls, user_count = getallusers()
    
    # Format attendance records
    attendance_records = []
    for i in range(l):
        attendance_records.append({
            "name": names[i],
            "roll": rolls[i],
            "time": times[i]
        })
    
    # Format user list
    users_list = []
    for i in range(user_count):
        users_list.append({
            "name": user_names[i],
            "roll": user_rolls[i],
            "identifier": userlist[i]
        })
    
    return jsonify({
        "attendance": attendance_records,
        "users": users_list,
        "settings": _settings,
        "totalreg": totalreg(),
        "datetoday2": datetoday2,
        "auto_attendance": auto_attendance_enabled,
        "recognition_active": recognition_active,
        "enrollment_active": enrollment_active
    })

@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    print(f"[deleteuser] Deleting user: {duser}")
    
    # Delete face images folder
    user_folder = 'static/faces/' + duser
    if os.path.exists(user_folder):
        deletefolder(user_folder)
        print(f"[deleteuser] Deleted face images folder: {user_folder}")
    
    # Delete embeddings if available
    if EMBEDDING_AVAILABLE and _embedding_storage is not None:
        _embedding_storage.delete_user(duser)
        print(f"[deleteuser] Deleted embeddings for: {duser}")
    
    # Delete from KNN model by retraining
    if os.path.exists('static/faces') and len(os.listdir('static/faces')) == 0:
        # No users left, delete model
        if os.path.exists('static/face_recognition_model.pkl'):
            os.remove('static/face_recognition_model.pkl')
            clear_recognition_model_cache()
            print(f"[deleteuser] Deleted KNN model (no users left)")
    else:
        # Retrain model without the deleted user
        try:
            print(f"[deleteuser] Retraining model without user {duser}...")
            train_model()
            print(f"[deleteuser] Model retrained successfully")
        except Exception as e:
            print(f"[deleteuser] Error retraining model: {e}")
            import traceback
            traceback.print_exc()
    
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/retrain', methods=['GET'])
def retrain():
    """Retrain the face recognition model with all current users."""
    try:
        if train_model():
            names, rolls, times, l = extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, auto_attendance=auto_attendance_enabled, mess='Model retrained successfully!')
        else:
            names, rolls, times, l = extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, auto_attendance=auto_attendance_enabled, mess='Failed to retrain model. Make sure you have registered users.')
    except Exception as e:
        print(f"Error retraining model: {e}")
        import traceback
        traceback.print_exc()
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, auto_attendance=auto_attendance_enabled, mess=f'Error retraining model: {str(e)}')

def generate_frames():
    """Generator function for video streaming with recognition/enrollment status."""
    while True:
        frame = get_latest_frame()
        if frame is None:
            # Create a black frame with message
            frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for camera feed...", (10, VIDEO_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Detect faces and draw on frame
            faces = detect_faces_haar(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Show recognition status if active
            with recognition_lock:
                if recognition_active:
                    cv2.putText(frame, f"Recognized: {recognition_status['recognized']}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, recognition_status['color'], 2)
                    if recognition_status['message']:
                        cv2.putText(frame, recognition_status['message'], (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, recognition_status['color'], 2)
            
            # Show enrollment status if active
            with enrollment_lock:
                if enrollment_active:
                    cv2.putText(frame, f"Captured: {enrollment_status['captured']}/{enrollment_status['total']}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, enrollment_status['color'], 2)
                    if enrollment_status['message']:
                        cv2.putText(frame, enrollment_status['message'], (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, enrollment_status['color'], 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def recognition_worker():
    """Background worker for manual recognition - no OpenCV windows."""
    global recognition_active, recognition_status
    
    # Check if we have either embeddings or KNN model
    has_model = False
    if EMBEDDING_AVAILABLE and _embedding_storage is not None:
        if _embedding_storage.get_user_count() > 0:
            has_model = True
            print(f"Recognition: Using embedding system with {_embedding_storage.get_user_count()} users")
    
    if not has_model:
        if 'face_recognition_model.pkl' in os.listdir('static'):
            has_model = True
            print("Recognition: Using KNN model")
    
    if not has_model:
        print("Recognition: No model available. Please enroll users first or retrain model.")
        with recognition_lock:
            recognition_active = False
            recognition_status = {"recognized": 0, "message": "No model - enroll users first", "color": (0, 0, 255)}
        return
    
    print("Starting manual recognition session...")
    recognized = set()  # Track which users have been marked in THIS session (one entry per student per session)
    prev_box = None
    stable_start = None
    start_time = time.time()
    timeout = get_setting("pipeline_timeout", 30)
    
    with recognition_lock:
        recognition_active = True
        recognition_status = {"recognized": 0, "message": "Recognition active - detecting faces...", "color": (0, 255, 0)}
    
    try:
        while (time.time() - start_time) < timeout:
            frame = get_latest_frame()
            
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Detect faces using Haar Cascade
            faces = detect_faces_haar(frame)
            
            if len(faces) == 0:
                prev_box = None
                stable_start = None
                with recognition_lock:
                    recognition_status["message"] = "No face detected"
                time.sleep(0.1)
                continue
            
            # Use the first detected face
            x, y, w, h = faces[0]
            current_box = (x, y, w, h)
            center_dist = center_distance(prev_box, current_box)
            now = time.time()
            
            max_movement = get_setting("max_center_movement", 15.0)
            stable_time = get_setting("stable_time", 0.4)
            # Threshold for similarity (higher is better, so use >= comparison)
            # Lower threshold = more lenient, higher = more strict
            # For embeddings: 0.5-0.6 is typical, for KNN converted: 0.3-0.5
            match_threshold = get_setting("match_distance_threshold", 0.5)
            
            if prev_box is None:
                prev_box = current_box
                stable_start = now
                with recognition_lock:
                    recognition_status["message"] = "Face detected - hold still..."
            else:
                if center_dist <= max_movement:
                    if stable_start is None:
                        stable_start = now
                    elapsed = now - stable_start
                    if elapsed >= stable_time:
                        # Stable - recognize (try up to 3 times for better accuracy)
                        # Ensure bounds are within frame
                        y1 = max(0, y)
                        x1 = max(0, x)
                        y2 = min(frame.shape[0], y + h)
                        x2 = min(frame.shape[1], x + w)
                        face_roi = frame[y1:y2, x1:x2]
                        if face_roi.size > 0:
                            # Check face size - should be reasonably large
                            if h < 50 or w < 50:
                                with recognition_lock:
                                    recognition_status["message"] = "Face too small - move closer"
                                    recognition_status["color"] = (0, 165, 255)
                                time.sleep(0.3)
                                prev_box = None
                                stable_start = None
                                continue
                            
                            # Try recognition up to 3 times and take the best result
                            best_pred = None
                            best_score = 0.0  # For similarity, higher is better
                            attempts = 3
                            for attempt in range(attempts):
                                pred, score = identify_face(face_roi)
                                if pred is not None and score is not None:
                                    # score is similarity (higher is better)
                                    if score > best_score:
                                        best_score = score
                                        best_pred = pred
                                time.sleep(0.1)  # Small delay between attempts
                            
                            if best_pred is not None and best_score > 0:
                                # best_score is a similarity score (higher is better)
                                print(f"[DEBUG] Recognition result: pred={best_pred}, score={best_score:.4f}, threshold={match_threshold:.4f}")
                                if best_score >= match_threshold:
                                    # Verify user still exists (not deleted)
                                    user_folder = f"static/faces/{best_pred}"
                                    if not os.path.exists(user_folder):
                                        print(f"[DEBUG] User {best_pred} was deleted, skipping attendance")
                                        with recognition_lock:
                                            recognition_status["message"] = f"User {best_pred} was deleted"
                                            recognition_status["color"] = (0, 0, 255)
                                        time.sleep(0.5)
                                        prev_box = None
                                        stable_start = None
                                        continue
                                    
                                    # Mark attendance only once per session
                                    if best_pred in recognized:
                                        print(f"[DEBUG] {best_pred} already marked in this session, skipping")
                                        with recognition_lock:
                                            recognition_status["message"] = f"{best_pred} already marked in this session"
                                            recognition_status["color"] = (0, 165, 255)
                                    else:
                                        print(f"[DEBUG] Attempting to add attendance for {best_pred}...")
                                        result = add_attendance(best_pred)
                                        print(f"[DEBUG] add_attendance returned: {result}")
                                        if result:
                                            recognized.add(best_pred)  # Mark as recognized in this session
                                            print(f"✓✓✓ ATTENDANCE MARKED: {best_pred} (similarity={best_score:.4f}) ✓✓✓")
                                            with recognition_lock:
                                                recognition_status["recognized"] = len(recognized)
                                                recognition_status["message"] = f"✓ Recognized: {best_pred} - Marked once"
                                                recognition_status["color"] = (0, 255, 0)
                                        else:
                                            print(f"[DEBUG] add_attendance failed for {best_pred}")
                                            with recognition_lock:
                                                recognition_status["message"] = f"Failed to save attendance for {best_pred}"
                                                recognition_status["color"] = (0, 0, 255)
                                else:
                                    # Similarity too low - show it for debugging
                                    print(f"[DEBUG] Similarity too low: {best_pred} (similarity={best_score:.4f} < threshold={match_threshold:.4f})")
                                    with recognition_lock:
                                        recognition_status["message"] = f"Similarity too low: {best_score:.4f} (need: {match_threshold:.4f})"
                                        recognition_status["color"] = (0, 165, 255)  # Orange
                            else:
                                print("No prediction returned from identify_face")
                                with recognition_lock:
                                    recognition_status["message"] = "No prediction - check model or retrain"
                                    recognition_status["color"] = (0, 0, 255)
                        time.sleep(0.5)
                        prev_box = None
                        stable_start = None
                else:
                    prev_box = current_box
                    stable_start = now
                    with recognition_lock:
                        recognition_status["message"] = "Face moving - hold still..."
            
            time.sleep(0.1)
    except Exception as e:
        print(f"Error in recognition session: {e}")
        import traceback
        traceback.print_exc()
    finally:
        with recognition_lock:
            recognition_active = False
            recognition_status = {"recognized": 0, "message": "", "color": (0, 255, 0)}
        print("Recognition session ended")

@app.route("/start")
def start():
    """Start manual recognition in background - no window opens."""
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')
    
    # Start recognition in background thread
    if not recognition_active:
        recognition_thread = threading.Thread(target=recognition_worker, daemon=True)
        recognition_thread.start()
    
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, auto_attendance=auto_attendance_enabled, mess='Recognition started - watch the live feed!')

def enrollment_worker(newusername, newuserid):
    """Background worker for enrollment - no OpenCV windows."""
    global enrollment_active, enrollment_status
    
    userfolder = f"static/faces/{newusername}_{newuserid}"
    captured = 0
    prev_box = None
    stable_start = None
    nimgs = get_setting("nimgs", 10)
    
    with enrollment_lock:
        enrollment_active = True
        enrollment_status = {"captured": 0, "total": nimgs, "message": "Position your face in front of the camera...", "color": (0, 255, 255)}
    
    try:
        print("Camera ready! Position your face in front of the camera.")
        
        while captured < nimgs:
            frame = get_latest_frame()
            
            if frame is None:
                with enrollment_lock:
                    enrollment_status["message"] = "Waiting for camera feed..."
                time.sleep(0.1)
                continue
            
            # Detect faces using Haar Cascade
            faces = detect_faces_haar(frame)
            
            if len(faces) == 0:
                prev_box = None
                stable_start = None
                with enrollment_lock:
                    enrollment_status["message"] = "No face detected - position your face"
                time.sleep(0.1)
                continue
            
            # Use the first detected face
            x, y, w, h = faces[0]
            current_box = (x, y, w, h)
            center_dist = center_distance(prev_box, current_box)
            now = time.time()
            
            max_movement = get_setting("max_center_movement", 15.0)
            stable_time = get_setting("stable_time", 0.4)
            
            if prev_box is None:
                prev_box = current_box
                stable_start = now
                with enrollment_lock:
                    enrollment_status["message"] = "Face detected - hold still..."
                    enrollment_status["color"] = (0, 255, 255)
            else:
                if center_dist <= max_movement:
                    if stable_start is None:
                        stable_start = now
                    elapsed = now - stable_start
                    stability_progress = min(elapsed / stable_time, 1.0)
                    
                    if elapsed >= stable_time:
                        # capture
                        # Ensure bounds are within frame
                        y1 = max(0, y)
                        x1 = max(0, x)
                        y2 = min(frame.shape[0], y + h)
                        x2 = min(frame.shape[1], x + w)
                        face_roi = frame[y1:y2, x1:x2]
                        if face_roi.size > 0:
                            # Save image
                            path = os.path.join(userfolder, f"{captured}.jpg")
                            cv2.imwrite(path, face_roi)
                            
                            # Extract and store embedding if available
                            if EMBEDDING_AVAILABLE and _host_embedding_extractor is not None and _embedding_storage is not None:
                                try:
                                    # Flip to match recognition
                                    flipped_roi = cv2.flip(face_roi, 1)
                                    embedding = _host_embedding_extractor.extract(flipped_roi)
                                    if embedding is not None:
                                        user_id = f"{newusername}_{newuserid}"
                                        _embedding_storage.add_embedding(user_id, embedding)
                                        print(f"Extracted and stored embedding {captured+1}/{nimgs}")
                                except Exception as e:
                                    print(f"Error extracting embedding: {e}")
                            
                            captured += 1
                            print(f"Captured {captured}/{NIMGS}")
                            with enrollment_lock:
                                enrollment_status["captured"] = captured
                                enrollment_status["message"] = f"CAPTURED! ({captured}/{NIMGS})"
                                enrollment_status["color"] = (0, 255, 0)
                            time.sleep(0.5)
                            prev_box = None
                            stable_start = None
                        else:
                            with enrollment_lock:
                                enrollment_status["message"] = "Face extraction failed, retrying..."
                                enrollment_status["color"] = (0, 0, 255)
                            prev_box = None
                            stable_start = None
                    else:
                        remaining = stable_time - elapsed
                        with enrollment_lock:
                            enrollment_status["message"] = f"Hold still... {remaining:.1f}s (Move: {center_dist:.1f}px)"
                            enrollment_status["color"] = (0, 255, 255)
                else:
                        prev_box = current_box
                        stable_start = now
                        with enrollment_lock:
                            enrollment_status["message"] = f"Face moving too much ({center_dist:.1f}px > {max_movement}px) - hold still!"
                            enrollment_status["color"] = (0, 0, 255)
            
            time.sleep(0.1)
    except Exception as e:
        print(f"Error in enrollment session: {e}")
        import traceback
        traceback.print_exc()
    finally:
        with enrollment_lock:
            enrollment_active = False
            enrollment_status = {"captured": 0, "total": nimgs, "message": "", "color": (0, 255, 255)}
        
        # retrain model after capture
        if captured > 0:
            train_model()
            print(f"Enrollment completed: {captured}/{nimgs} images captured for {newusername}_{newuserid}")
        else:
            print("Enrollment failed - no images captured")

@app.route("/add", methods=["GET", "POST"])
def add():
    """Start enrollment in background - no window opens."""
    if request.method == "POST":
        newusername = request.form.get("newusername")
        newuserid = request.form.get("newuserid")
        
        if not newusername or not newuserid:
            names, rolls, times, l = extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Please provide both name and ID.')
        
        userfolder = f"static/faces/{newusername}_{newuserid}"
        if os.path.exists(userfolder):
            names, rolls, times, l = extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='User already exists!')
        
        os.makedirs(userfolder, exist_ok=True)
        
        # Start enrollment in background thread
        if not enrollment_active:
            enrollment_thread = threading.Thread(target=enrollment_worker, args=(newusername, newuserid), daemon=True)
            enrollment_thread.start()
            names, rolls, times, l = extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=f'Enrollment started for {newusername} - watch the live feed!')
        else:
            names, rolls, times, l = extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Enrollment already in progress!')
    
    # GET -> redirect to home
    return redirect(url_for("home"))

@app.route("/settings", methods=["GET", "POST"])
def settings():
    """Settings page to configure system parameters."""
    if request.method == "POST":
        # Update settings from form
        try:
            old_interval = get_setting("auto_attendance_interval_minutes", 15)
            
            set_setting("auto_attendance_interval_minutes", int(request.form.get("auto_attendance_interval", 15)))
            set_setting("nimgs", int(request.form.get("nimgs", 10)))
            set_setting("pipeline_timeout", int(request.form.get("pipeline_timeout", 30)))
            set_setting("stable_time", float(request.form.get("stable_time", 0.4)))
            set_setting("max_center_movement", float(request.form.get("max_center_movement", 15.0)))
            set_setting("match_distance_threshold", float(request.form.get("match_distance_threshold", 0.5)))
            
            # Update global variables
            global NIMGS, PIPELINE_TIMEOUT, STABLE_TIME, MAX_CENTER_MOVEMENT, MATCH_DISTANCE_THRESHOLD
            NIMGS = get_setting("nimgs", 10)
            PIPELINE_TIMEOUT = get_setting("pipeline_timeout", 30)
            STABLE_TIME = get_setting("stable_time", 0.4)
            MAX_CENTER_MOVEMENT = get_setting("max_center_movement", 15.0)
            MATCH_DISTANCE_THRESHOLD = get_setting("match_distance_threshold", 0.5)
            
            # Update enrollment status total
            with enrollment_lock:
                enrollment_status["total"] = NIMGS
            
            # If auto attendance interval changed and it's enabled, note that it will use new interval on next cycle
            new_interval = get_setting("auto_attendance_interval_minutes", 15)
            if old_interval != new_interval and auto_attendance_enabled:
                print(f"Auto attendance interval changed from {old_interval} to {new_interval} minutes. Will apply on next cycle.")
            
            names, rolls, times, l = extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), 
                                 datetoday2=datetoday2, auto_attendance=auto_attendance_enabled, 
                                 mess='Settings saved successfully!')
        except Exception as e:
            print(f"Error saving settings: {e}")
            import traceback
            traceback.print_exc()
            return render_template('settings.html', settings=_settings, 
                                 mess=f'Error saving settings: {str(e)}')
    
    # GET - show settings page
    return render_template('settings.html', settings=_settings)

@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    """API endpoint to retrain the model"""
    try:
        if train_model():
            return jsonify({"success": True, "message": "Model retrained successfully!"})
        else:
            return jsonify({"success": False, "message": "Failed to retrain model. Make sure you have registered users."}), 400
    except Exception as e:
        return jsonify({"success": False, "message": f"Error retraining model: {str(e)}"}), 500

@app.route('/api/users/<user_id>', methods=['DELETE'])
def api_delete_user(user_id):
    """API endpoint to delete a user"""
    try:
        deletefolder(f'static/faces/{user_id}')
        if EMBEDDING_AVAILABLE and _embedding_storage is not None:
            _embedding_storage.delete_user(user_id)
        # Retrain model after deletion
        train_model()
        return jsonify({"success": True, "message": f"User {user_id} deleted successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error deleting user: {str(e)}"}), 500

# Serve React app static files (must be last route)
@app.route('/<path:path>')
def serve_react_app(path):
    """Serve React app files - this should be the last route"""
    # Don't interfere with API routes
    if path.startswith('api/'):
        return jsonify({"error": "Not found"}), 404
    
    # Don't interfere with existing Flask routes (but allow video_feed)
    if path in ['start', 'add', 'listusers', 'deleteuser', 'settings', 'retrain', 'toggle_auto_attendance']:
        return jsonify({"error": "Use Flask routes"}), 404
    
    # Serve video feed
    if path == 'video_feed':
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    # Handle module loading for React app
    if path.startswith('load-module'):
        from urllib.parse import parse_qs, urlparse
        parsed = urlparse(request.url)
        params = parse_qs(parsed.query)
        module_path = params.get('path', [''])[0]
        if module_path:
            # Try to serve the module file
            if os.path.exists(module_path.lstrip('/')):
                try:
                    with open(module_path.lstrip('/'), 'r') as f:
                        code = f.read()
                    # Transform with Babel (would need Babel server-side, but for now just return)
                    return Response(code, mimetype='application/javascript')
                except:
                    pass
    
    # Try to serve static files (JS, CSS, TSX, images, etc.)
    if os.path.exists(path):
        try:
            # Set proper MIME types
            if path.endswith('.tsx') or path.endswith('.ts'):
                return send_from_directory('.', path), 200, {'Content-Type': 'text/plain; charset=utf-8'}
            elif path.endswith('.css'):
                return send_from_directory('.', path), 200, {'Content-Type': 'text/css; charset=utf-8'}
            elif path.endswith('.js') or path.endswith('.jsx'):
                return send_from_directory('.', path), 200, {'Content-Type': 'application/javascript; charset=utf-8'}
            return send_from_directory('.', path)
        except Exception as e:
            print(f"Error serving {path}: {e}")
            pass
    
    # Only serve index.html for HTML requests or root-like paths
    # Don't serve it for TypeScript/JavaScript file requests
    if path.endswith(('.tsx', '.ts', '.js', '.jsx', '.css', '.json', '.png', '.jpg', '.jpeg', '.svg', '.ico')):
        return jsonify({"error": "File not found"}), 404
    
    # For React app, serve index.html (React Router will handle client-side routing)
    # But only if it's not a file request
    if os.path.exists('index.html') and not path.startswith('/api/'):
        return send_from_directory('.', 'index.html')
    
    return jsonify({"error": "Not found"}), 404

# ---------------- Main ----------------
if __name__ == "__main__":
    # Initialize embedding system
    if EMBEDDING_AVAILABLE:
        init_embedding_system()
    
    # Load KNN model for backward compatibility
    load_recognition_model()
    
    print("Starting Flask server. Total registered:", totalreg())
    if EMBEDDING_AVAILABLE:
        print("Using embedding-based face recognition (on-device extraction available)")
    else:
        print("Using OpenCV Haar Cascade for face detection (no pre-trained models required)")
    
    # Start continuous camera feed in background thread
    camera_thread = threading.Thread(target=camera_feed_thread, daemon=True)
    camera_thread.start()
    print("Starting continuous camera feed...")
    
    # Start auto attendance thread (will wait until enabled)
    auto_attendance_thread = threading.Thread(target=auto_attendance_worker, daemon=True)
    auto_attendance_thread.start()
    
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

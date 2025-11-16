<div align="center">

# OAK-D-FACE-RECOGNITION-ATTENDANCE

<br/>

_Seamless Recognition, Effortless Attendance, Limitless Possibilities_


<a href="https://github.com/abdullahTarik/OAK-D-face-recognition-attendance">
    <img src="https://img.shields.io/github/last-commit/abdullahTarik/OAK-D-face-recognition-attendance?label=last%20commit" />
</a>
<a href="https://github.com/abdullahTarik/OAK-D-face-recognition-attendance">
    <img src="https://img.shields.io/github/languages/top/abdullahTarik/OAK-D-face-recognition-attendance?label=python" />
</a>
<a href="https://github.com/abdullahTarik/OAK-D-face-recognition-attendance">
    <img src="https://img.shields.io/github/languages/count/abdullahTarik/OAK-D-face-recognition-attendance?label=languages" />
</a>

<br/>
<br/>

_Built with the tools and technologies:_


<img src="https://img.shields.io/badge/Flask-black?logo=flask&logoColor=white" />
<img src="https://img.shields.io/badge/JSON-orange?logo=json&logoColor=white" />
<img src="https://img.shields.io/badge/Markdown-000000?logo=markdown&logoColor=white" />
<img src="https://img.shields.io/badge/npm-CB3837?logo=npm&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-FF6F00?logo=scikitlearn&logoColor=white" />
<img src="https://img.shields.io/badge/React-61DAFB?logo=react&logoColor=black" />

<br/>

<img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/XML-0060A0?logo=xml&logoColor=white" />
<img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white" />
<img src="https://img.shields.io/badge/Vite-646cff?logo=vite&logoColor=white" />
<img src="https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white" />

<br/><br/>

</div>

## Overview

A modern, real-time face recognition attendance system built with OAK-D camera, Flask backend, and React frontend. The system uses advanced face embedding extraction for accurate recognition and provides a beautiful, responsive web interface for managing attendance and users.

![Face Recognition System](imgs/example.gif)

Image is taken from [here](https://www.pexels.com/photo/multi-cultural-people-3184419/).

## 🎯 Features

- **Real-time Face Recognition**: Uses OAK-D camera with on-device face detection and embedding extraction
- **Dual Recognition Methods**: 
  - Advanced embedding-based recognition (cosine similarity)
  - Fallback KNN-based recognition for backward compatibility
- **Modern Web UI**: Beautiful React-based interface with glassmorphism design
- **Automatic Attendance**: Configurable automatic attendance marking at intervals
- **Manual Attendance**: Real-time manual attendance taking with live video feed
- **User Management**: Easy enrollment, deletion, and management of users
- **Attendance Records**: Daily CSV attendance logs with timestamps
- **Configurable Settings**: Adjustable recognition thresholds, timeouts, and intervals
- **Multi-session Support**: Mark attendance once per session, multiple entries per day

## 🏗️ Architecture

```
OAK-D Camera (MyriadX VPU)
    ↓
Face Detection (Haar Cascade / Yunet)
    ↓
Face Embedding Extraction (Host-side / On-device)
    ↓
Flask Backend (Python)
    ↓
Embedding Matching (Cosine Similarity)
    ↓
React Frontend (TypeScript/TSX)
    ↓
Attendance Recording (CSV)
```

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**
- **Flask**: Web framework and API server
- **OpenCV**: Face detection using Haar Cascade
- **DepthAI**: OAK-D camera integration
- **scikit-learn**: KNN classifier for fallback recognition
- **NumPy**: Numerical operations
- **Pandas**: Data processing and CSV management
- **ONNX Runtime**: Host-side embedding extraction
- **Joblib**: Model persistence

### Frontend
- **React**: UI framework
- **TypeScript/TSX**: Type-safe component development
- **Tailwind CSS**: Modern styling (via CDN)
- **Babel Standalone**: In-browser TypeScript compilation

### Face Recognition
- **Embedding-based**: Uses face embeddings with cosine similarity matching
- **KNN Fallback**: Traditional KNN classifier for backward compatibility
- **Image Preprocessing**: Grayscale conversion, histogram equalization, normalization

## 📋 Prerequisites

- **OAK-D Camera** (or compatible DepthAI device)
- **Python 3.8+**
- **Modern web browser** (Chrome, Firefox, Edge)
- **Linux/Windows/macOS** (tested on Linux)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/abdullahTarik/face-recognition-attendance
cd face-recognition-attendance
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `opencv-python>=4.8.0` - Computer vision library
- `numpy>=1.24.0` - Numerical computing
- `depthai==2.20.2.0` - OAK-D camera SDK
- `flask>=2.3.0` - Web framework
- `pandas>=2.0.0` - Data processing
- `scikit-learn>=1.3.0` - Machine learning
- `joblib>=1.3.0` - Model persistence
- `blobconverter>=1.4.0` - Model conversion
- `onnxruntime>=1.15.0` - ONNX model runtime

### 3. Connect OAK-D Camera

Ensure your OAK-D camera is connected via USB and recognized by the system.

### 4. (Optional) Download Face Embedding Model

For improved accuracy, download a MobileFaceNet ONNX model:
1. Download from [ONNX Model Zoo](https://github.com/onnx/models)
2. Place in `models/mobilefacenet.onnx`
3. The system will automatically use it

## 💻 Usage

### Start the Server

```bash
python3 main.py
```

The server will start on `http://0.0.0.0:5000` (accessible from any device on your network).

### Access the Web Interface

Open your browser and navigate to:
- **Local**: `http://localhost:5000`
- **Network**: `http://YOUR_IP:5000` (replace YOUR_IP with your machine's IP address)

### Basic Workflow

1. **Enroll Users**:
   - Click "Add User" in the web interface
   - Enter name and roll number (format: `name_roll`, e.g., `john_123`)
   - Position face in front of camera
   - System captures 10 images automatically
   - Model is trained automatically

2. **Take Attendance**:
   - **Manual**: Click "Take Attendance" and position face in front of camera
   - **Automatic**: Enable auto-attendance in settings (marks attendance every 15 minutes by default)

3. **View Attendance**:
   - Check the attendance table on the main page
   - Daily CSV files are saved in `Attendance/` directory

4. **Manage Users**:
   - View all registered users
   - Delete users (removes face data and embeddings)
   - Retrain model after changes

## 📁 Project Structure

```
ML/
├── main.py                 # Main Flask application
├── embedding_pipeline.py   # OAK-D pipeline for face detection
├── embedding_storage.py    # Embedding storage and matching
├── host_embedding.py       # Host-side embedding extraction
├── requirements.txt       # Python dependencies
├── settings.json          # Configuration file
│
├── static/
│   ├── faces/             # User face images (name_roll/)
│   ├── embeddings/        # Stored face embeddings (.pkl)
│   └── images/            # Static images
│
├── Attendance/            # Daily attendance CSV files
│
├── components/            # React components
│   ├── AttendanceTable.tsx
│   ├── ControlPanel.tsx
│   ├── Header.tsx
│   ├── UserManagement.tsx
│   ├── VideoFeed.tsx
│   └── common/            # Reusable components
│
├── hooks/                 # React hooks
│   └── useAppData.ts
│
├── templates/             # Legacy HTML templates (fallback)
│
├── utils/                 # Utility functions
│
├── index.html             # Main HTML entry point
├── index.tsx              # React app entry point
├── types.ts               # TypeScript type definitions
└── README.md              # This file
```

## ⚙️ Configuration

Settings can be configured via the web interface or by editing `settings.json`:

```json
{
  "auto_attendance_interval_minutes": 15,
  "nimgs": 10,
  "pipeline_timeout": 30,
  "stable_time": 0.4,
  "max_center_movement": 15.0,
  "match_distance_threshold": 0.5
}
```

### Settings Explained

- **auto_attendance_interval_minutes**: Minutes between automatic attendance checks
- **nimgs**: Number of images to capture during enrollment
- **pipeline_timeout**: Timeout for recognition pipeline (seconds)
- **stable_time**: Time face must be stable before recognition (seconds)
- **max_center_movement**: Maximum face movement allowed (pixels)
- **match_distance_threshold**: Similarity threshold for recognition (0.0-1.0, higher = stricter)

## 🔌 API Endpoints

### Web Interface
- `GET /` - Main React application
- `GET /video_feed` - Live camera feed (MJPEG stream)

### API Endpoints
- `GET /api/status` - Get system status, attendance, users, and settings
- `POST /api/retrain` - Retrain the recognition model
- `DELETE /api/users/<user_id>` - Delete a user and retrain model

### Legacy Endpoints (for backward compatibility)
- `GET /listusers` - List all registered users
- `POST /add` - Add new user (enrollment)
- `POST /deleteuser` - Delete user
- `GET /settings` - View settings page
- `POST /settings` - Update settings
- `POST /start` - Start manual attendance recognition
- `POST /toggle_auto_attendance` - Toggle automatic attendance

## 🎨 Frontend Development

The frontend uses React with TypeScript/TSX, compiled in-browser using Babel Standalone.

### Key Components

- **App.tsx**: Main application component
- **VideoFeed.tsx**: Live camera feed display
- **ControlPanel.tsx**: Control buttons (enroll, attendance, etc.)
- **AttendanceTable.tsx**: Attendance records display
- **UserManagement.tsx**: User list and management
- **SettingsForm.tsx**: Configuration interface

### Development Notes

- TypeScript files are compiled on-the-fly in the browser
- No build step required for development
- Uses Tailwind CSS via CDN (not recommended for production)
- Module loader handles relative imports automatically

## 🔧 Troubleshooting

### Camera Not Detected
- Ensure OAK-D camera is connected via USB
- Check USB cable and port
- Verify DepthAI installation: `python3 -c "import depthai; print(depthai.__version__)"`

### Recognition Not Working
- Ensure users are enrolled (at least 10 images per user)
- Check recognition threshold in settings (try lowering it)
- Verify face is well-lit and clearly visible
- Retrain model: Click "Retrain Model" in settings

### Web Interface Not Loading
- Check browser console for errors (F12)
- Ensure Flask server is running
- Try accessing `http://localhost:5000?old=1` for legacy interface
- Clear browser cache and hard refresh (Ctrl+Shift+R)

### Module Import Errors
- Ensure all Python dependencies are installed
- Check Python version: `python3 --version` (should be 3.8+)
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

### Low Recognition Accuracy
- Increase number of enrollment images
- Improve lighting conditions
- Adjust `match_distance_threshold` in settings
- Ensure face is centered and stable during enrollment
- Use MobileFaceNet model for better embeddings (see EMBEDDING_SETUP.md)

## 📝 Attendance Format

Attendance is saved as CSV files in `Attendance/` directory:
- Filename: `Attendance-MM_DD_YY.csv`
- Format: `Name,Roll,Time`
- Example: `John,123,14:30:25`

## 🔐 Security Notes

- The system runs on `0.0.0.0:5000` by default (accessible from network)
- For production, use a reverse proxy (nginx) with HTTPS
- Consider adding authentication for production deployments
- Face images are stored locally - ensure proper access controls

## 📚 Additional Documentation

- **EMBEDDING_SETUP.md**: Detailed guide for embedding system setup
- **requirements.txt**: Complete dependency list with versions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

See individual file licenses in the repository.

## 🙏 Acknowledgments

- **DepthAI** for OAK-D camera SDK
- **OpenCV** for face detection algorithms
- **React** team for the excellent UI framework
- **scikit-learn** for machine learning tools

## 📧 Support

For issues and questions:
1. Check the Troubleshooting section above
2. Review EMBEDDING_SETUP.md for embedding-related issues
3. Open an issue on the repository

---

**Built with ❤️ using OAK-D, Flask, and React**

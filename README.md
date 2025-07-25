# Real-Time Eye Blink Detection using OpenCV & Dlib

This project detects eye blinks in real-time using a webcam. It leverages facial landmark detection with Dlib to track eye movements and compute Eye Aspect Ratio (EAR) to detect blinks.

### 🔍 How It Works
- Dlib detects 68 facial landmarks on the face.
- The eye region landmarks are used to calculate EAR.
- If EAR drops below a defined threshold for consecutive frames, it is counted as a blink.
- Blink count is updated in real time and saved as a CSV file upon exit.

### 🧠 Technologies Used
- Python
- OpenCV
- Dlib
- SciPy
- NumPy
- Pandas

### 💾 Output
- Real-time video stream with blink count overlay.
- `data.csv` file storing total blink count after the session ends.

---

> ⚠️ Note: You must download `shape_predictor_68_face_landmarks.dat` from the [official Dlib model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the root directory before running the script.

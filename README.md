AI Security & Face Recognition System
A high-performance surveillance solution that integrates real-time motion detection, deep-learning-based face recognition, and automated security logging. This system is designed to monitor a video feed (webcam or RTSP) and autonomously identify individuals or capture alerts for strangers.

System Overview
The project consists of two core applications:

The Recognition Engine (main.py): A multi-threaded monitoring system that handles live analysis and data logging.

The Enrollment Portal (enrollment.py): A modern GUI for onboarding new users and managing biometric data.

Architectural Decisions
1. Multi-Threaded Performance
To ensure the video feed remains smooth (low latency), the system separates tasks into independent threads:

Frame Grabber Thread: Continuously pulls new frames from the camera to prevent buffer lag.

Face Analyzer Thread: Handles the heavy computation (Motion detection and AI recognition).

Database Writer Thread: Uses a Queue system to handle SQL writes in the background so the AI never has to wait for the hard drive.

2. Intelligent Motion Triggering
Unlike basic systems that run AI on every single frame, this system uses "Background Subtraction":

The system calculates the "delta" between the current frame and a weighted average of previous frames.

AI analysis only activates when significant motion is detected, drastically reducing CPU usage.

3. Deep Learning Recognition
Algorithm: Uses the face_recognition library, which is built on dlib’s state-of-the-art HOG (Histogram of Oriented Gradients) model.

Optimization: Frames are downscaled to 25% of their size for the recognition pass, then scaled back up for display, allowing for real-time speeds without sacrificing too much accuracy.

4. Professional Data Management
SQLite Logging: Every detection is stored in a structured SQL database with a name, timestamp, and status.

Signature Caching: Facial encodings are "pickled" into a cache file. This allows the system to load thousands of known faces instantly on startup without re-processing images.

Stranger Alerts: If an unknown face is detected, the system captures a high-resolution snapshot and logs a "Security Alert" to the database.

Technical Stack
OpenCV: Used for video stream manipulation and image processing.

Dlib / Face_Recognition: Powers the facial feature extraction and distance matching.

CustomTkinter: A modern wrapper for Tkinter used to build a dark-themed, professional enrollment interface.

SQLite3: Provides a lightweight, reliable database for security logs.

Python-Dotenv: Manages sensitive configuration like RTSP URLs and directory paths.

Setup and Usage
Configuration
Create a .env file in the root folder.

Set RTSP_URL=0 for a local webcam or enter your IP camera link.

Define your KNOWN_FACES_DIR for storing enrollment photos.

Enrollment
Run enrollment.py to add new authorized users:

Enter the user's name.

Capture 5 distinct photos using the "Snap Photo" button.

The app automatically clears the old AI cache so the new user is recognized immediately upon the next system start.

Recognition
Run main.py to start the security monitor:

Known users will be outlined in GREEN.

Strangers will be outlined in RED and their photo will be saved to the captured_strangers folder.

Logs can be reviewed in the security_log.db file.

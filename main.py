import cv2
import face_recognition
import threading
import numpy as np
import os
import time
import logging
import pickle
import sqlite3
import queue
from datetime import datetime
from dotenv import load_dotenv

# Load configuration
load_dotenv()

# Setup professional logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s [%(levelname)s] %(threadName)s: %(message)s',
    handlers=[logging.FileHandler("security_system.log"), logging.StreamHandler()]
)

class DetectionDatabase:
    """Handles SQL interactions in a thread-safe manner using a Queue."""
    def __init__(self, db_name="security_log.db"):
        self.db_name = db_name
        self.log_queue = queue.Queue()
        self._create_table()
        
        # Start background writer thread
        self.writer_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.writer_thread.start()

    def _create_table(self):
        try:
            with sqlite3.connect(self.db_name) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        status TEXT NOT NULL
                    )
                ''')
        except sqlite3.Error as e:
            logging.error(f"Database initialization failed: {e}")

    def log_entry(self, name: str, status: str = "Recognized"):
        """Add log request to the queue (Non-blocking)."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_queue.put((name, timestamp, status))

    def _process_queue(self):
        """Dedicated thread to handle SQLite writes."""
        while True:
            item = self.log_queue.get()
            if item is None: break
            name, timestamp, status = item
            try:
                with sqlite3.connect(self.db_name) as conn:
                    conn.execute(
                        "INSERT INTO detections (name, timestamp, status) VALUES (?, ?, ?)",
                        (name, timestamp, status)
                    )
            except sqlite3.Error as e:
                logging.error(f"Async database write failed: {e}")
            finally:
                self.log_queue.task_done()

class FaceRecognitionSystem:
    def __init__(self):
        # 1. Configuration & Validation
        self.rtsp_url = os.getenv("RTSP_URL")
        self.faces_dir = os.getenv("KNOWN_FACES_DIR", "known_faces")
        self.cache_file = "face_signatures.pkl"
        self.unknown_dir = "captured_strangers"
        
        if not self.rtsp_url:
            logging.error("RTSP_URL not found in .env file.")
            raise ValueError("Configuration Error: Missing RTSP_URL")

        # Thresholds
        self.log_cooldown = 30
        self.unknown_capture_cooldown = 300 
        self.motion_threshold = 1000 
        
        # 2. State & Threading
        self.shutdown_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.results_lock = threading.Lock()
        
        self.latest_frame = None
        self.latest_results = ([], []) 
        self.last_seen = {}
        self.last_unknown_time = 0
        self.last_motion_time = 0
        
        # 3. Model & DB
        self.db = DetectionDatabase()
        self.known_encodings = []
        self.known_names = []
        
        os.makedirs(self.unknown_dir, exist_ok=True)
        self._initialize_signatures()

    def _initialize_signatures(self):
        """Loads encodings from cache or builds from scratch."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_encodings = data["encodings"]
                    self.known_names = data["names"]
                logging.info(f"Loaded {len(self.known_names)} signatures from cache.")
                return
            except Exception as e:
                logging.warning(f"Cache corrupt ({e}). Rebuilding...")

        self._rebuild_database()

    def _rebuild_database(self):
        if not os.path.exists(self.faces_dir):
            logging.error(f"Faces directory '{self.faces_dir}' not found!")
            return

        logging.info(f"Scanning {self.faces_dir}...")
        new_encodings, new_names = [], []

        for person_name in os.listdir(self.faces_dir):
            person_dir = os.path.join(self.faces_dir, person_name)
            if not os.path.isdir(person_dir): continue

            for filename in os.listdir(person_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    try:
                        img = face_recognition.load_image_file(os.path.join(person_dir, filename))
                        encs = face_recognition.face_encodings(img)
                        if encs:
                            new_encodings.append(encs[0])
                            new_names.append(person_name)
                    except Exception as e:
                        logging.error(f"Error processing {filename}: {e}")
        
        self.known_encodings, self.known_names = new_encodings, new_names
        with open(self.cache_file, 'wb') as f:
            pickle.dump({"encodings": self.known_encodings, "names": self.known_names}, f)
        logging.info("Database cache successfully built.")

    def _save_unknown_face(self, frame):
        now = time.time()
        if now - self.last_unknown_time > self.unknown_capture_cooldown:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(self.unknown_dir, f"stranger_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            self.db.log_entry("Unknown Stranger", status="Captured")
            logging.warning(f"SECURITY ALERT: Unknown face saved to {filename}")
            self.last_unknown_time = now

    def frame_grabber(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        while not self.shutdown_event.is_set():
            ret, frame = cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                logging.warning("Stream disconnected. Retrying in 5s...")
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(self.rtsp_url)
        cap.release()

    def face_analyzer(self):
        avg_frame = None
        while not self.shutdown_event.is_set():
            local_frame = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    local_frame = self.latest_frame.copy()

            if local_frame is None:
                time.sleep(0.1)
                continue

            gray = cv2.cvtColor(local_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if avg_frame is None:
                avg_frame = gray.copy().astype("float")
                continue

            cv2.accumulateWeighted(gray, avg_frame, 0.5)
            frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg_frame))
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if any(cv2.contourArea(c) > self.motion_threshold for c in cnts):
                self.last_motion_time = time.time()
                small = cv2.resize(local_frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                
                locs = face_recognition.face_locations(rgb_small, model="hog")
                encs = face_recognition.face_encodings(rgb_small, locs)
                
                current_names = []
                found_unknown = False

                for enc in encs:
                    name = "Unknown"
                    if self.known_encodings:
                        dists = face_recognition.face_distance(self.known_encodings, enc)
                        best_idx = np.argmin(dists)
                        if dists[best_idx] < 0.5:
                            name = self.known_names[best_idx]
                    
                    current_names.append(name)

                    if name != "Unknown":
                        if name not in self.last_seen or (time.time() - self.last_seen[name]) > self.log_cooldown:
                            self.db.log_entry(name) 
                            logging.info(f"MATCH: {name}")
                            self.last_seen[name] = time.time()
                    else:
                        found_unknown = True
                
                if found_unknown:
                    self._save_unknown_face(local_frame)

                with self.results_lock:
                    self.latest_results = (locs, current_names)
            else:
                if time.time() - self.last_motion_time > 3:
                    with self.results_lock:
                        self.latest_results = ([], [])
            
            time.sleep(0.01)

    def run(self):
        threading.Thread(target=self.frame_grabber, daemon=True).start()
        threading.Thread(target=self.face_analyzer, daemon=True).start()

        logging.info("System operational. Press 'q' to exit.")
        try:
            while True:
                with self.frame_lock:
                    if self.latest_frame is None: continue
                    display_frame = self.latest_frame.copy()

                with self.results_lock:
                    locs, names = self.latest_results
                
                for (top, right, bottom, left), name in zip(locs, names):
                    top, right, bottom, left = top*4, right*4, bottom*4, left*4
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(display_frame, (left, top - 30), (right, top), color, cv2.FILLED)
                    cv2.putText(display_frame, name, (left + 6, top - 6), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow("Security AI Portfolio Project", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.shutdown_event.set()
            cv2.destroyAllWindows()
            logging.info("System shutdown complete.")

if __name__ == "__main__":
    try:
        system = FaceRecognitionSystem()
        system.run()
    except Exception as e:
        logging.critical(f"System failed to start: {e}")
import cv2
import os
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import messagebox

# --- SETTINGS ---
FACES_PATH = r"C:\Users\NoName\Desktop\camera project\faces"
# Ensure this points to where your security_system.py looks for the cache
PICKLE_PATH = "face_signatures.pkl" 
NUM_PHOTOS_NEEDED = 5

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ManualEnrollmentApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Pro Face Enrollment")
        self.geometry("1100x750")

        # Camera & Logic State
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.count = 0
        self.current_person_name = ""

        # --- UI LAYOUT ---
        self.header = ctk.CTkLabel(self, text="MANUAL FACE CAPTURE", font=ctk.CTkFont(size=24, weight="bold"))
        self.header.pack(pady=15)

        # Main Container (Left: Camera, Right: Preview)
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(expand=True, fill="both", padx=20)

        # Left Column: Video Feed
        self.left_col = ctk.CTkFrame(self.main_container, corner_radius=15)
        self.left_col.pack(side="left", padx=10, expand=True)

        self.video_label = ctk.CTkLabel(self.left_col, text="")
        self.video_label.pack(padx=10, pady=10)

        # Right Column: Preview
        self.right_col = ctk.CTkFrame(self.main_container, width=250, corner_radius=15)
        self.right_col.pack(side="right", padx=10, fill="y")
        
        self.preview_title = ctk.CTkLabel(self.right_col, text="LAST CAPTURE", font=ctk.CTkFont(size=14, weight="bold"))
        self.preview_title.pack(pady=10)
        
        self.preview_label = ctk.CTkLabel(self.right_col, text="No image yet", width=200, height=150)
        self.preview_label.pack(padx=20, pady=20)

        # Counter and Progress (at the bottom)
        self.counter_label = ctk.CTkLabel(self, text="Photos Captured: 0 / 5", font=ctk.CTkFont(size=16))
        self.counter_label.pack(pady=5)
        
        self.progress = ctk.CTkProgressBar(self, width=500)
        self.progress.set(0)
        self.progress.pack(pady=10)

        # Bottom Controls
        self.controls = ctk.CTkFrame(self, fg_color="transparent")
        self.controls.pack(pady=20)

        self.name_entry = ctk.CTkEntry(self.controls, placeholder_text="Enter Name", width=250)
        self.name_entry.grid(row=0, column=0, padx=10)

        self.btn_snap = ctk.CTkButton(self.controls, text="📸 SNAP PHOTO", 
                                      command=self.take_photo, 
                                      fg_color="#3498db", 
                                      hover_color="#2980b9",
                                      height=40,
                                      font=ctk.CTkFont(weight="bold"))
        self.btn_snap.grid(row=0, column=1, padx=10)

        self.btn_reset = ctk.CTkButton(self.controls, text="Reset", width=80, 
                                       fg_color="#e74c3c", hover_color="#c0392b",
                                       command=self.reset_session)
        self.btn_reset.grid(row=0, column=2, padx=10)

        self.update_video()

    def take_photo(self):
        name = self.name_entry.get().strip().replace(" ", "_")
        
        if not name:
            messagebox.showwarning("Input Required", "Please enter a name before taking photos.")
            return

        self.current_person_name = name
        save_path = os.path.join(FACES_PATH, name)
        if not os.path.exists(save_path): 
            os.makedirs(save_path)

        ret, frame = self.cap.read()
        if ret:
            # Save a clean mirrored frame
            frame = cv2.flip(frame, 1)
            img_name = f"{name}_{self.count}.jpg"
            file_path = os.path.join(save_path, img_name)
            cv2.imwrite(file_path, frame)
            
            # Update Preview Window
            self.update_preview(frame)
            
            # Update Counters
            self.count += 1
            self.progress.set(self.count / NUM_PHOTOS_NEEDED)
            self.counter_label.configure(text=f"Photos Captured: {self.count} / {NUM_PHOTOS_NEEDED}")

            # Visual Flash effect
            self.left_col.configure(border_width=2, border_color="#2ecc71")
            self.after(150, lambda: self.left_col.configure(border_width=0))

            # --- CHECK COMPLETION AND DELETE PICKLE ---
            if self.count >= NUM_PHOTOS_NEEDED:
                status_msg = ""
                if os.path.exists(PICKLE_PATH):
                    try:
                        os.remove(PICKLE_PATH)
                        status_msg = "\n\nCache deleted. New faces will be loaded on next startup."
                    except Exception as e:
                        status_msg = f"\n\nWarning: Could not delete cache file: {e}"
                else:
                    status_msg = "\n\nCache file not found (nothing to delete)."

                messagebox.showinfo("Complete", f"Successfully captured {NUM_PHOTOS_NEEDED} photos!{status_msg}")
                self.btn_snap.configure(state="disabled")

    def update_preview(self, frame):
        # Resize frame for the small preview box
        prev_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prev_img = prev_img.resize((200, 150))
        prev_img_tk = ImageTk.PhotoImage(image=prev_img)
        self.preview_label.configure(image=prev_img_tk, text="")
        self.preview_label.image = prev_img_tk

    def reset_session(self):
        self.count = 0
        self.progress.set(0)
        self.counter_label.configure(text="Photos Captured: 0 / 5")
        self.btn_snap.configure(state="normal")
        self.name_entry.delete(0, 'end')
        self.preview_label.configure(image="", text="No image yet")

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            # Draw detection boxes for visual feedback
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (46, 204, 113), 2)

            # Update main camera feed
            img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.after(10, self.update_video)

if __name__ == "__main__":
    app = ManualEnrollmentApp()
    app.mainloop()
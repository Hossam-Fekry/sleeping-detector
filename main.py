import cv2
import math
import customtkinter
from PIL import Image, ImageTk
import mediapipe as mp

# ========== Constants ==========
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.23
CLOSED_FRAMES = 15

# ========== Global Variables ==========
cap = None
closed_eyes_counter = 0
running = False

# ========== Mediapipe Setup ==========
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# ========== UI Setup ==========
customtkinter.set_appearance_mode("light")
customtkinter.set_default_color_theme("blue")

root = customtkinter.CTk()
root.title("Sleeping Detector")
root.geometry("525x325")
root.resizable(False, False)

title_label = customtkinter.CTkLabel(root, text="Sleeping detector", font=("Arial", 24, "bold"))
title_label.pack(pady=10)

video_label = customtkinter.CTkLabel(root, text="")
video_label.place(y=65, x = 200)

status_label = customtkinter.CTkLabel(root, text="Is sleeping: No", font=("Arial", 18, "bold"))
status_label.pack(pady=20, side = customtkinter.BOTTOM)

# ========== EAR Calculation ==========
def calculate_EAR(eye):
    v1 = math.dist(eye[1], eye[5])
    v2 = math.dist(eye[2], eye[4])
    h = math.dist(eye[0], eye[3])
    return (v1 + v2) / (2.0 * h)

# ========== Frame Detection Loop ==========
def detect():
    global closed_eyes_counter

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (300, 250))
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)
    status_text = "No"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
            )

            landmarks = face_landmarks.landmark
            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]
            left_ear = calculate_EAR(left_eye)
            right_ear = calculate_EAR(right_eye)
            avg_ear = (left_ear + right_ear) / 2

            if avg_ear < EAR_THRESHOLD:
                closed_eyes_counter += 1
            else:
                closed_eyes_counter = 0

            if closed_eyes_counter >= CLOSED_FRAMES:
                status_text = "Yes"

    status_label.configure(text=f"Is sleeping: {status_text}")

    # Convert BGR to RGB and show in Tkinter
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, detect)

# ========== Start and Exit Functions ==========
def start_detecting():
    global cap, running
    cap = cv2.VideoCapture(0)
    running = True
    detect()

def exit_app():
    global cap, running
    running = False
    if cap:
        cap.release()
    root.destroy()

# ========== Buttons ==========
start_button = customtkinter.CTkButton(root, text="Start Detecting", fg_color="#24E524", command=start_detecting,  hover_color="#10B510")
# start_button.pack(pady=5)
start_button.place(x = 15, y = 125)

exit_button = customtkinter.CTkButton(root, text="Exit", fg_color="#FF5F33", command=exit_app, hover_color="#E54E24")
# exit_button.pack(pady=5)
exit_button.place(x = 15, y = 170)


# ========== Run App ==========
root.mainloop()

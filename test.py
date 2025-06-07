import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Drawing settings (optional)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Eye indices (based on Face Mesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# EAR calculation function
def calculate_EAR(eye_landmarks):
    vertical1 = math.dist(eye_landmarks[1], eye_landmarks[5])
    vertical2 = math.dist(eye_landmarks[2], eye_landmarks[4])
    horizontal = math.dist(eye_landmarks[0], eye_landmarks[3])
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

# EAR threshold and counter
EAR_THRESHOLD = 0.23
CLOSED_EYES_FRAME_LIMIT = 15
closed_eyes_counter = 0

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert color
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get face mesh results
    result = face_mesh.process(rgb_frame)
    h, w, _ = frame.shape

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Get landmarks
            landmarks = face_landmarks.landmark

            # Extract eye coordinates
            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]

            # Draw eye contours (optional)
            for pt in left_eye:
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)
            for pt in right_eye:
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)

            # Calculate EAR
            left_ear = calculate_EAR(left_eye)
            right_ear = calculate_EAR(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Show EAR on screen
            cv2.putText(frame, f'EAR: {avg_ear:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Sleep detection
            if avg_ear < EAR_THRESHOLD:
                closed_eyes_counter += 1
            else:
                closed_eyes_counter = 0

            if closed_eyes_counter >= CLOSED_EYES_FRAME_LIMIT:
                cv2.putText(frame, "😴 USER IS SLEEPING OR DROWSY!", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    cv2.imshow("Sleeping Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

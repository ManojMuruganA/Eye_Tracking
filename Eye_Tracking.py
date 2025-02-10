import cv2
import dlib
import time
import numpy as np

# Load the detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\manojmurugan\Downloads\shape_predictor_68_face_landmarks.dat")

# Define a function to detect and return the coordinates of the eyes
def get_eye_coordinates(landmarks):
    # Right eye
    right_eye_points = [landmarks.part(i) for i in range(36, 42)]
    right_eye_coords = np.array([(p.x, p.y) for p in right_eye_points])

    # Left eye
    left_eye_points = [landmarks.part(i) for i in range(42, 48)]
    left_eye_coords = np.array([(p.x, p.y) for p in left_eye_points])

    return right_eye_coords, left_eye_coords

# Define a function to calculate the aspect ratio of the eyes
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define the main function to start monitoring
def monitor_eye_movement():
    cap = cv2.VideoCapture(0)
    suspicious_count = 0
    eye_ar_threshold = 0.2
    frame_check = 20
    consecutive_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            landmarks = predictor(gray, face)
            right_eye_coords, left_eye_coords = get_eye_coordinates(landmarks)

            right_ear = eye_aspect_ratio(right_eye_coords)
            left_ear = eye_aspect_ratio(left_eye_coords)
            
            # Print EAR values for both eyes
            print(f"Right EAR: {right_ear:.2f} | Left EAR: {left_ear:.2f}")

            # Check if either eye is below the threshold
            if right_ear < eye_ar_threshold or left_ear < eye_ar_threshold:
                consecutive_frames += 1
                if consecutive_frames >= frame_check:
                    suspicious_count += 1
                    print(f"Suspicious activity detected! Count: {suspicious_count}")
                    consecutive_frames = 0
            else:
                consecutive_frames = 0

            # Draw circles around the detected eye landmarks for visualization
            for point in right_eye_coords:
                cv2.circle(frame, tuple(point), 2, (255, 0, 0), -1)
            for point in left_eye_coords:
                cv2.circle(frame, tuple(point), 2, (255, 0, 0), -1)

        # Show the frame with detected landmarks
        cv2.imshow("Eye Movement Monitor", frame)
        
        # Check if 'Esc' is pressed to exit
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor_eye_movement()

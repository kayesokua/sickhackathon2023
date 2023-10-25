import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)  # 0 represents the default camera (change if needed)

with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        results = holistic.process(frame)

        if results.face_landmarks:

            # Checks if the Y position of the upper lip (landmark 12) is higher than the Y position of the lower lip (landmark 11)
            # To be improved

            smile = results.face_landmarks.landmark[12].y > results.face_landmarks.landmark[11].y
            label = "Smiling" if smile else "Non-Smiling"

            cv2.putText(
                frame,
                label,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if smile else (0, 0, 255),
                2,
            )

        cv2.imshow("Smile Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
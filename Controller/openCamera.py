import cv2
import mediapipe as mp


class OpenCamera:
    # Gives access to a device camera
    capture = cv2.VideoCapture(0)  # 0 means 1 camera on device
    stopFlag: bool = True

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    def __init__(self):
        self.record()

    def record(self):
        while self.stopFlag:
            # Reads frames from a camera
            ret, frame = self.capture.read()

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            if not ret:
                break

            # Checks if there are hands in the frame
            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    for id, lm in enumerate(hand_landmark.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if id == 20:
                            cv2.circle(frame, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                        self.mp_draw.draw_landmarks(frame, hand_landmark, self.mp_hands.HAND_CONNECTIONS)

            # Shows a frame on the screen
            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Closes the camera ad destroys the window
        self.capture.release()
        cv2.destroyAllWindows()

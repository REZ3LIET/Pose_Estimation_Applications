import time
import math
import cv2 as cv
import numpy as np

from Hand_Tacking import hand_tracking_module as htm

class FingerCounter:
    def __init__(self):
        pass

    def thumb_check(self, landmarks):
        lm_0 = landmarks[0][1]
        lm_1 = landmarks[2][1]
        thumb = landmarks[4][1]

        # Assuming thumb is open on left
        if lm_1 <= thumb <= lm_0 or lm_1 <= lm_0 <= thumb:
            return False
        # Assuming thumb is open on right
        elif lm_0 <= thumb <= lm_1 or thumb <= lm_0 <= lm_1:
            return False
        return True

    def finger_counter(self, landmarks, count=True):
        finger_tips = [8, 12, 16, 20]
        standing_fingers = list(map(lambda x: landmarks[x][2] < landmarks[x-1][2], finger_tips))
        standing_fingers.insert(0, self.thumb_check(landmarks))
        if count:
            return sum(standing_fingers)
        return standing_fingers
    
def disp_number(number, size, img):
    height, width = img.shape[:2]
    start_cords = 10, 150
    end_cords = start_cords[0] + size, start_cords[1] + size
    text_cords = start_cords[0] + size//3, end_cords[1] - size//4

    cv.rectangle(img, start_cords, end_cords, (255, 0, 0), -1)
    cv.putText(img, str(number), text_cords, cv.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 2)
    return img

def main():
    detector = htm.HandPoseDetect(max_hands=1, detect_conf=0.8, track_conf=0.8)
    fc = FingerCounter()
    cap = cv.VideoCapture(0)
    prev_time = time.time()
    cur_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video Over")
            break

        img = frame.copy()
        finger_count = 0

        landmarks, output = detector.detect_landmarks(img)
        if landmarks:
            hand_info = detector.get_info(landmarks, img.shape[:2])
            finger_count = fc.finger_counter(hand_info)
            # print(finger_count)

        output = cv.flip(output, 1)
        cur_time = time.time()
        fps = 1/(cur_time - prev_time)
        prev_time = cur_time
        cv.putText(output, f'FPS: {str(int(fps))}', (10, 70), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 50, 170), 2)
        output = disp_number(finger_count, 150, output)

        cv.imshow("Detection", output)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
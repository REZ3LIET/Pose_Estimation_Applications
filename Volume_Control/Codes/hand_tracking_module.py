import os
import time
import argparse
import cv2 as cv
from mediapipe import solutions as mp

class HandPoseDetect:
    def __init__(self, static_image=False, max_hands=2, complexity=1, detect_conf=0.5, track_conf=0.5):
        self.mp_hands = mp.hands
        self.mp_draw = mp.drawing_utils
        self.hands = self.mp_hands.Hands(static_image, max_hands, complexity, detect_conf, track_conf)

    def detect_landmarks(self, img, disp=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        detected_landmarks = results.multi_hand_landmarks

        if detected_landmarks:
            if disp:
                for h_landmark in detected_landmarks:
                    self.mp_draw.draw_landmarks(img, h_landmark, self.mp_hands.HAND_CONNECTIONS)
        return detected_landmarks, img

    def get_info(self, detected_landmarks, img_dims, hand_no=1):
        lm_list = []
        if not detected_landmarks:
            return lm_list

        if hand_no > 2:
            print('[WARNING] Provided hand number is greater than max number 2')
            print('[WARNING] Calculating information for hand 2')
            hand_no = 2
        elif hand_no < 1:
            print('[WARNING] Provided hand number is less than min number 1')
            print('[WARNING] Calculating information for hand 1')

        if len(detected_landmarks) < 2:
            hand_no = 0
        else:
            hand_no -= 1

        height, width = img_dims
        for id, h_landmarks in enumerate(detected_landmarks[hand_no].landmark):
            cord_x, cord_y = int(h_landmarks.x * width), int(h_landmarks.y * height)
            lm_list.append([id, cord_x, cord_y])

        return lm_list

def main(path, is_image=True):
    if is_image:
        detector = HandPoseDetect(static_image=True)
        ori_img = cv.imread(path)

        img = ori_img.copy()
        landmarks, output_img = detector.detect_landmarks(img)
        info_landmarks = detector.get_info(landmarks, 3, img.shape[:2])
        # print(info_landmarks)

        cv.imshow("Landmarks", output_img)
        cv.waitKey(0)

    else:
        detector = HandPoseDetect()
        cap = cv.VideoCapture(path)
        prev_time = time.time()
        cur_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video Over")
                break

            img = frame.copy()
            landmarks, output_img = detector.detect_landmarks(img)
            info_landmarks = detector.get_info(landmarks, 3, img.shape[:2])
            # print(info_landmarks)

            cur_time = time.time()
            fps = 1/(cur_time - prev_time)
            prev_time = cur_time
            cv.putText(output_img, f'FPS: {str(int(fps))}', (10, 70), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 50, 170), 2)

            cv.imshow("Detection", output_img)
            cv.imshow("Original", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Type of media and path to it")
    parser.add_argument("path", help="Path to media from current working directory")
    parser.add_argument("--image", action="store_true", help="If media in an image")

    args = parser.parse_args()
    is_image = args.image
    media_path = args.path

    if os.path.exists(os.path.join(os.getcwd(), media_path)):
        main(media_path, is_image)
    else:
        print("Invalid Path")

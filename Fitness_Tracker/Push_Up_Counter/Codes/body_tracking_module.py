import os
import time
import argparse
import cv2 as cv
import mediapipe as mp

class BodyPoseDetect:
    def __init__(self, static_image=False, complexity=1, smooth_lm=True, segmentation=False, smooth_sm=True, detect_conf=0.5, track_conf=0.5):
        self.mp_body = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.body = self.mp_body.Pose(static_image, complexity, smooth_lm, segmentation, smooth_sm, detect_conf, track_conf)

    def detect_landmarks(self, img, disp=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.body.process(img_rgb)
        detected_landmarks = results.pose_landmarks

        if detected_landmarks:
            if disp:
                self.mp_draw.draw_landmarks(img, detected_landmarks, self.mp_body.POSE_CONNECTIONS)
        return detected_landmarks, img
    
    def get_info(self, detected_landmarks, img_dims):
        lm_list = []
        if not detected_landmarks:
            return lm_list

        height, width = img_dims
        for id, b_landmark in enumerate(detected_landmarks.landmark):
            cord_x, cord_y = int(b_landmark.x * width), int(b_landmark.y * height)
            lm_list.append([id, cord_x, cord_y])
        
        return lm_list

def main(path, is_image=True):
    if is_image:
        detector = BodyPoseDetect(static_image=True)
        ori_img = cv.imread(path)
        
        img = ori_img.copy()
        landmarks, output_img = detector.detect_landmarks(img)
        info_landmarks = detector.get_info(landmarks, img)
        # print(info_landmarks[3])

        cv.imshow("Original", ori_img)
        cv.imshow("Detection", output_img)
        cv.waitKey(0)
    
    else:
        detector = BodyPoseDetect()
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
            info_landmarks = detector.get_info(landmarks, img)
            # print(info_landmarks[3])

            cur_time = time.time()
            fps = 1/(cur_time - prev_time)
            prev_time = cur_time
            cv.putText(output_img, f'FPS: {str(int(fps))}', (10, 70), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 50, 170), 2)

            cv.imshow("Original", frame)
            cv.imshow("Detection", output_img)
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

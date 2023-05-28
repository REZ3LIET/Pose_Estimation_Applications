import time
import cv2 as cv
import numpy as np

from utils import get_fps
from body_tracking_module import BodyPoseDetect

class PushUpChecker:
    def __init__(self):
        self.landmarks = None

    def get_angle(self, pt_1, pt_2, pt_3):
        '''pt_2 is mid point'''
        line_1 = np.array(pt_1) - np.array(pt_2)
        line_2 = np.array(pt_3) - np.array(pt_2)

        vector_normal = np.linalg.norm(line_1) * np.linalg.norm(line_2)
        cosine_angle = np.dot(line_1, line_2) / vector_normal
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def check_form(self, left_side_pts, right_side_pts, img=None, draw=False):
        left_pts = list(map(lambda x: self.landmarks[x][1:], left_side_pts))
        right_pts = list(map(lambda x: self.landmarks[x][1:], right_side_pts))

        left_body_angle = self.get_angle(*left_pts)
        right_body_angle = self.get_angle(*right_pts)

        if draw:
            assert img is not None, ("Image required to draw points")
            self._draw_points([left_pts, right_pts], img)

        return [left_body_angle, right_body_angle]
    
    def _draw_points(self, points, img):
        assert len(points) == 2, ("Points provided are incorrect")
        left_pts = points[0]
        right_pts = points[1]

        for pts in [left_pts + right_pts][0]:
            cv.circle(img, pts, 5, (0, 255, 255), -1)

        cv.line(img, left_pts[0], left_pts[1], (255, 255, 255), 3)
        cv.line(img, left_pts[1], left_pts[2], (255, 255, 255), 3)

        cv.line(img, right_pts[0], right_pts[1], (255, 255, 255), 3)
        cv.line(img, right_pts[1], right_pts[2], (255, 255, 255), 3)

        return img

def count_push_ups(body_form, arm_angles, count_flag):
    form = None
    push_up_count = 0

    if body_form[0] > 140 and body_form[1] > 140:
        form = True
    else:
        form = False

    if form:
        if arm_angles[0] > 150 and arm_angles[1] > 150:
            if count_flag:
                push_up_count = 1
                count_flag = False

        elif arm_angles[0] < 90 and arm_angles[1] < 90:
            count_flag = True
    
    return form, push_up_count, count_flag

def main():
    detector = BodyPoseDetect()
    counter = PushUpChecker()

    cap = cv.VideoCapture("Push_Up_Counter\Data\pushup_1.mp4")
    curr_time = time.time()

    left_body = [11, 23, 25]    # Shoulder, Hip, Knee left side
    right_body = [12, 24, 26]    # Shoulder, Hip, Knee right side

    left_arm = [11, 13, 15]    # Shoulder, Elbow, Wrist left side
    right_arm = [12, 14, 16]    # Shoulder, Elbow, Wrist right side

    count_flag = False
    total_push_ups = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video Over")
            break

        img = frame.copy()
        results, output = detector.detect_landmarks(img, False)
        landmarks = detector.get_info(results, img.shape[:2])

        if landmarks:
            counter.landmarks = landmarks
            body_form = counter.check_form(left_body, right_body)
            arm_angles = counter.check_form(left_arm, right_arm, img=output, draw=True)
            form, push_up_count, count_flag = count_push_ups(body_form, arm_angles, count_flag)
            if form:
                form_msg = "Form: Correct"
                color = (0, 255, 0)
            else:
                form_msg = "Form: Incorrect"
                color = (0, 0, 255)
            total_push_ups += push_up_count
            cv.rectangle(output, (output.shape[1] - 30, 1), (output.shape[1] - 275, 60), (100, 100, 100), -1)
            cv.putText(output, form_msg, (output.shape[1] - 250, 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            cv.putText(output, f'Count: {str(total_push_ups)}', (img.shape[1] - 250, 50), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)

            fps = int(get_fps(curr_time))
            curr_time = time.time()
            cv.putText(output, f'FPS: {str(fps)}', (10, 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

        cv.imshow("RESULT", output)

        if cv.waitKey(15) & 0xFF == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
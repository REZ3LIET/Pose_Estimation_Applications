import time
import math
import cv2 as cv
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

from Hand_Tacking import hand_tracking_module as htm

def volume_bar(img, volume_percent):
    height, width, _ = img.shape
    width_percent = (width * 10)//100
    height_percent = (height * 10)//100
    bar_ht = np.interp(volume_percent, [0, 100], [height - height_percent, height_percent]).astype(int)
    # print(bar_ht, volume_percent)

    rect_min = width_percent, height_percent + bar_ht - 50
    rect_max = width - width_percent, height - height_percent

    if volume_percent <= 5:
        color = (0, 0, 255)
    elif volume_percent >= 95:
        color = (0, 255, 0)
    else:
        color = (0, 255, 255)

    cv.rectangle(img, rect_min, rect_max, color, 20)
    cv.putText(img, f'{str(volume_percent)}%', (rect_max[0] - 35, rect_min[1] - 15), cv.FONT_HERSHEY_PLAIN, 2, color, 2)
    return img

def track_landmark(lm_1, lm_2, img):
    """
    Tracks the location of fingertips provided
    Parameters
    ----------
    lm_1: tuple
        Location of 1st landmark
    lm_2: tuple
        Location of 2nd landmark
    img: array
        Image to draw on
    """
    lm_1 = np.array(lm_1)
    lm_2 = np.array(lm_2)
    mid_pt = (lm_1 + lm_2) // 2
    length = int(math.dist(lm_1, lm_2))

    cv.circle(img, lm_1, 9, (255, 0, 0), -1)
    cv.circle(img, lm_2, 9, (255, 0, 0), -1)
    cv.line(img, lm_1, lm_2, (255, 0, 0), 3)
    if length <= 20:
        color = (0, 0, 255)
    elif length >= 200:
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)

    cv.circle(img, mid_pt, 13, color, -1)

    return length, img

def main():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
    volume_range = volume.GetVolumeRange()

    detector = htm.HandPoseDetect(max_hands=1, detect_conf=0.8, track_conf=0.8)
    cap = cv.VideoCapture(0)
    prev_time = time.time()
    cur_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video Over")
            break

        img = frame.copy()
        landmarks, output = detector.detect_landmarks(img)
        if landmarks:
            hand_info = detector.get_info(landmarks, img.shape[:2])
            length, output = track_landmark(hand_info[4][1:], hand_info[8][1:], output)
            volume_level = np.interp(length, [20, 200], volume_range[:2]).astype(int)
            volume.SetMasterVolumeLevel(volume_level, None)
            # print(length, volume_level)


        output = cv.flip(output, 1)
        volume_percent = np.interp(volume.GetMasterVolumeLevel(), volume_range[:2], [0, 100]).astype(int)
        output = volume_bar(output, volume_percent)

        cur_time = time.time()
        fps = 1/(cur_time - prev_time)
        prev_time = cur_time
        cv.putText(output, f'FPS: {str(int(fps))}', (10, 70), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 50, 170), 2)

        cv.imshow("Detection", output)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
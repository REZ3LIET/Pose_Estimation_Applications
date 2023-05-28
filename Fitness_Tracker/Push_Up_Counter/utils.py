import time

def get_fps(prev_time):
    fps = 1/(time.time() - prev_time)
    return fps
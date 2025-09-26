from multiprocessing import Queue, shared_memory
import numpy as np
import cv2 as cv
from inference import IMAGE_SHAPE
from constants import SYNC_FPS, GLOBAL_MESSAGE_LENGTH, CAMERA_URL

def capture(shms: tuple[str, ...], q_out: Queue):
    shm_global_msg, shm_frame_in = shms

    shm_frame_in = shared_memory.SharedMemory(name=shm_frame_in)
    shm_global_msg = shared_memory.SharedMemory(name=shm_global_msg)

    frame = np.ndarray(IMAGE_SHAPE, dtype=np.uint8, buffer=shm_frame_in.buf)
    global_message = np.ndarray((GLOBAL_MESSAGE_LENGTH), dtype=np.uint8, buffer=shm_global_msg.buf)

    cap = cv.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera")
        global_message[0] = 0
        return

    try:
        while global_message[0]:
            _ret, frame_temp = cap.read()
            cv.resize(frame_temp, (640, 640), frame)
            if SYNC_FPS:
                q_out.put(0)
    except Exception:
        print(Exception)
        global_message[0] = 0


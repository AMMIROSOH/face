from multiprocessing import Queue, shared_memory
import numpy as np
import cv2 as cv
from inference import IMAGE_SHAPE
from constants import SYNC_FPS, GLOBAL_MESSAGE_LENGTH, CAMERA_URL, IMAGE_SHAPE_GUI

def capture(shms: tuple[str, ...], q_out: Queue):
    shm_global_msg, shm_frame_in, shm_cap_gui_frame = shms

    shm_frame_in = shared_memory.SharedMemory(name=shm_frame_in)
    gui_frame_in = np.ndarray(IMAGE_SHAPE_GUI, dtype=np.uint8, buffer=shm_cap_gui_frame.buf)
    shm_global_msg = shared_memory.SharedMemory(name=shm_global_msg)

    frame = np.ndarray(IMAGE_SHAPE, dtype=np.uint8, buffer=shm_frame_in.buf)
    global_message = np.ndarray((GLOBAL_MESSAGE_LENGTH), dtype=np.uint8, buffer=shm_global_msg.buf)

    cap = cv.VideoCapture("test.mov") # CAMERA_URL
    if not cap.isOpened():
        print(f"ERROR: Cannot find camera or movie")
        global_message[0] = 0
        return

    try:
        while global_message[0]:
            ret, frame_temp = cap.read()
            if(not ret):
               global_message[0] = 0
               break
            cv.resize(frame_temp, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]), frame)
            cv.resize(frame_temp, (IMAGE_SHAPE_GUI[1], IMAGE_SHAPE_GUI[0]), gui_frame_in)
            if SYNC_FPS:
                q_out.put(0, False)
    except Exception as e:
        print(e)
        global_message[0] = 0


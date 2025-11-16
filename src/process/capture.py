from multiprocessing import Queue, shared_memory
import numpy as np
import cv2 as cv
import time
from inference import FRAME_SHAPE
from constants import GLOBAL_MESSAGE_LENGTH, VIDEO_SOURCE, FRAME_SHAPE_GUI

def capture(shms: tuple[str, ...], q_out: Queue):
    shm_global_msg, shm_frame_in, shm_cap_gui_frame = shms

    shm_frame_in = shared_memory.SharedMemory(name=shm_frame_in)
    gui_frame_in = np.ndarray(FRAME_SHAPE_GUI, dtype=np.uint8, buffer=shm_cap_gui_frame.buf)
    shm_global_msg = shared_memory.SharedMemory(name=shm_global_msg)

    frame = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm_frame_in.buf)
    global_message = np.ndarray((GLOBAL_MESSAGE_LENGTH), dtype=np.uint8, buffer=shm_global_msg.buf)

    cap = cv.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"ERROR: Cannot find camera or movie")
        global_message[0] = 0
        return
    
    desired_fps = 60 
    frame_time = 1 / desired_fps 

    try:
        while global_message[0]:
            start_time = time.time()
            ret, frame_temp = cap.read()
            if(not ret):
               global_message[0] = 0
               q_out.put(0, False)
               cap.release()
               break
            cv.resize(frame_temp, (FRAME_SHAPE[1], FRAME_SHAPE[0]), frame)
            cv.resize(frame_temp, (FRAME_SHAPE_GUI[1], FRAME_SHAPE_GUI[0]), gui_frame_in)
            q_out.put(0, False)

            # limiting max fps
            elapsed_time = time.time() - start_time
            time_to_wait = frame_time - elapsed_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)
    except Exception as e:
        print(e)
        global_message[0] = 0
        q_out.put(0, False)


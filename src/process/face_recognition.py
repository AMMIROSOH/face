from multiprocessing import Queue, shared_memory
import numpy as np
import torch
from inference import IMAGE_SHAPE
from inference import Inference, LOC_LENGTH, CONF_LENGTH, LANDS_LENGTH, IMAGE_SHAPE
from constants import SYNC_FPS, GLOBAL_MESSAGE_LENGTH, MAX_PEAPLE

def face_recognition(shms: tuple[str, ...], q_in: Queue, q_out: Queue):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info_shape = (int((LOC_LENGTH + CONF_LENGTH + LANDS_LENGTH)/16800)*MAX_PEAPLE, )

    shm_global_msg, shm_frame_in, shm_info_in, shm_frame_out, shm_info_out, shm_vec_out = shms
    shm_frame_in = shared_memory.SharedMemory(name=shm_frame_in)
    shm_info_in = shared_memory.SharedMemory(name=shm_info_in)
    shm_frame_out = shared_memory.SharedMemory(name=shm_frame_out)
    shm_info_out = shared_memory.SharedMemory(name=shm_info_out)
    shm_global_msg = shared_memory.SharedMemory(name=shm_global_msg)

    frame_in = np.ndarray(IMAGE_SHAPE, dtype=np.uint8, buffer=shm_frame_in.buf)
    info_in = np.ndarray(info_shape, dtype=np.float32, buffer=shm_info_in.buf)
    frame_out = np.ndarray(IMAGE_SHAPE, dtype=np.uint8, buffer=shm_frame_out.buf)
    info_out = np.ndarray(info_shape, dtype=np.float32, buffer=shm_info_out.buf)
    # vec_out = np.ndarray(512*MAX_PEAPLE, dtype=np.float32, buffer=shm_vec_out.buf)
    global_message = np.ndarray((GLOBAL_MESSAGE_LENGTH), dtype=np.uint8, buffer=shm_global_msg.buf)

    while global_message[0]:
        count = 0
        if SYNC_FPS:
            # this will make SYNC_FPS not working
            count: int = q_in.get()
        for i in range(count):
            pass
        
        frame_out[:] = frame_in
        info_out[:] = info_in
        q_out.put(count)


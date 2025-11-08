from multiprocessing import Queue, shared_memory
import numpy as np
from inference import Inference, LOC_LENGTH, CONF_LENGTH, LANDS_LENGTH, DETECTION_LENGTH, FRAME_SHAPE
from constants import GLOBAL_MESSAGE_LENGTH

def detection(shms: tuple[str, ...], q_in: Queue, q_out: Queue):
    retinaModel = Inference(model="RetinaFace-R50_fp16")
    shm_global_msg, shm_frame_in, shm_frame_out, shm_info_out = shms

    shm_global_msg = shared_memory.SharedMemory(name=shm_global_msg)
    shm_frame_in = shared_memory.SharedMemory(name=shm_frame_in)
    shm_frame_out = shared_memory.SharedMemory(name=shm_frame_out)
    shm_info_out = shared_memory.SharedMemory(name=shm_info_out)

    global_message = np.ndarray((GLOBAL_MESSAGE_LENGTH), dtype=np.uint8, buffer=shm_global_msg.buf)
    frame_in = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm_frame_in.buf)
    frame_out = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm_frame_out.buf)
    info_out = np.ndarray(((LOC_LENGTH + CONF_LENGTH + LANDS_LENGTH) * DETECTION_LENGTH), dtype=np.float32, buffer=shm_info_out.buf)

    while global_message[0]:
        q_in.get()

        frame = np.array(frame_in, dtype=np.int32)
        # frame -= (104, 117, 123)
        frame = frame.transpose(2, 0, 1)

        loc, conf, landms = retinaModel.infer(frame)
        frame_out[:] = frame_in
        info_out[:] = np.concatenate((loc[0], landms[0], conf[0]))

        q_out.put(0)

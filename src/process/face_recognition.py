from multiprocessing import Queue, shared_memory
import numpy as np
import torch
import cv2 as cv
from utils.qdrant import search_vec
from inference import Inference, LOC_LENGTH, CONF_LENGTH, LANDS_LENGTH, IMAGE_SHAPE
from constants import SYNC_FPS, GLOBAL_MESSAGE_LENGTH, MAX_PEOPLE

def normalize(img):
    # img: (C, H, W)
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
    return (img - mean) / std

def face_recognition(shms: tuple[str, ...], q_in: Queue, q_out: Queue):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arcModel = Inference(model="arcface-r100-glint360k_fp16")
    info_shape = (int((LOC_LENGTH + CONF_LENGTH + LANDS_LENGTH)/16800)*MAX_PEOPLE, )

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

    vectors = np.zeros((MAX_PEOPLE, 512), np.float32)

    while global_message[0]:
        count = 0
        if SYNC_FPS:
            # todo: this will make SYNC_FPS not working
            count: int = q_in.get()
        if(count>0):
            loc, _conf, _lands = np.split(info_in[0:count*15], [4 * count, 4 * count + 1 * count])
            loc = loc.reshape(count, 4).astype(int)
            face_temp = np.zeros((112, 112, 3), np.uint8)
            for i in range(count):
                x1, y1, x2, y2 = loc[i]
                # todo: check for axis to doesnt go out of bounds
                cv.resize(frame_in[y1:y2, x1:x2], (112, 112), dst=face_temp)
                cv.cvtColor(face_temp, cv.COLOR_BGR2RGB, dst=face_temp)
                face = np.transpose(face_temp / 127.5 - 1.0, (2,0,1)).astype(np.float32)              
                # cv.imwrite("asd.jpg", frame_in[y1:y2, x1:x2]) # EASTER EGG
                # batch inference here
                vectors[i][:] = arcModel.infer(face)[0][0]
                results = search_vec(vectors[i].tolist())
                print(results[0].score, results[0].payload.get("name", "unkown"))
                if(len(results)>0):
                    if(results[0].score > 0.6):
                        name = results[0].payload.get("name", "unkown")
                        print(name, " is here with score: ", results[0].score)
                pass
            pass
        
        frame_out[:] = frame_in
        info_out[:] = info_in
        q_out.put(count)


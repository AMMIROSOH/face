from multiprocessing import Queue, shared_memory
from bytetrack.byte_tracker import BYTETracker
import numpy as np
import time
import cv2 as cv
from bytetrack.matching import iou_distance
from utils.qdrant import search_vec
from inference import Inference, LOC_LENGTH, CONF_LENGTH, LANDS_LENGTH, IMAGE_SHAPE
from constants import SYNC_FPS, GLOBAL_MESSAGE_LENGTH, MAX_PEOPLE

def face_recognition(shms: tuple[str, ...], q_in: Queue, q_out: Queue):
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
    global_message = np.ndarray((GLOBAL_MESSAGE_LENGTH), dtype=np.uint8, buffer=shm_global_msg.buf)


    gallery_count = 0
    gallery_met = []
    gallery_vectors = np.zeros((MAX_PEOPLE*2, 512), dtype=np.float32)
    while global_message[0]:
        count = 0
        if SYNC_FPS:
            # todo: this will make SYNC_FPS not working
            count: int = q_in.get()
        box_owners = [] 
        if(count>0):
            loc, _conf, _lands = np.split(info_in[0:count*15], [4 * count, 4 * count + 1 * count])
            loc = loc.reshape(count, 4).astype(int)
            face_temp = np.zeros((112, 112, 3), np.uint8)
            for i in range(count):
                x1, y1, x2, y2 = loc[i]
                cv.resize(frame_in[y1:y2, x1:x2], (112, 112), dst=face_temp)
                cv.cvtColor(face_temp, cv.COLOR_BGR2RGB, dst=face_temp)
                face = np.transpose(face_temp / 127.5 - 1.0, (2,0,1)).astype(np.float32)              
                # cv.imwrite("asd.jpg", frame_in[y1:y2, x1:x2]) # EASTER EGG
                vector = arcModel.infer(face)[0][0]
                norm = np.linalg.norm(vector)
                vector = vector/norm

                cosine_sim = gallery_vectors @ vector
                max_sim = np.max(cosine_sim)
                max_index = np.argmax(cosine_sim)

                if(max_sim<0.8 and gallery_count < MAX_PEOPLE*2):
                    results = search_vec(vector.tolist())
                    if(len(results) > 0 and results[0].score > 0.6):
                        hit = results[0]
                        index_gallery = next((i for i, row in enumerate(gallery_met) if row[0] == hit.id), None)
                        if(index_gallery != None):
                            gallery_vectors[index_gallery] = vector.copy()
                            box_owners.append(gallery_met[max_index][1])
                            #print(name, " it was same person: ", hit.score)
                        else:
                            name = hit.payload.get("name", "unkown")
                            gallery_vectors[gallery_count] = vector.copy()
                            gallery_count += 1
                            gallery_met.append([hit.id, name, hit.score, time.time()])

                            box_owners.append(name)
                            #print(name, " Welcome, score: ", hit.score)
                    else:
                        box_owners.append("unkown")
                else:
                    box_owners.append(gallery_met[max_index][1])
                    #print(gallery_met[max_index][1], " is still here with score: ", max_sim)
                #print("len(gallery_met)", len(gallery_met))
        
        frame_out[:] = frame_in
        info_out[:] = info_in
        q_out.put((count, box_owners))

def face_track(shms: tuple[str, ...], q_in: Queue, q_out: Queue):
    tracker = BYTETracker(args={"track_thresh": 0.5, "match_thresh": 0.7, "track_buffer": 30, "mot20": False })
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
    global_message = np.ndarray((GLOBAL_MESSAGE_LENGTH), dtype=np.uint8, buffer=shm_global_msg.buf)

    while global_message[0]:
        count = 0
        if SYNC_FPS:
            # todo: this will make SYNC_FPS not working
            count: int = q_in.get()
        box_owners = ["unkown"] * count 
        if(count>0):
            loc, conf, _ = np.split(info_in[0:count*15], [4 * count, 4 * count + 1 * count])
            loc = loc.reshape(count, 4).astype(int)
            conf = conf.reshape(count, 1).astype(float)
            track_loc = loc.copy()
            track_loc[:, 2] += track_loc[:, 0]
            track_loc[:, 3] += track_loc[:, 1]

            track_data = np.concatenate([track_loc, conf], axis=1)
            online_targets = tracker.update(track_data, [IMAGE_SHAPE[0], IMAGE_SHAPE[1]], [IMAGE_SHAPE[0], IMAGE_SHAPE[1]])
            for target in online_targets:
                box_owners[target.detection_index] = str(target.track_id)
        
        frame_out[:] = frame_in
        info_out[:] = info_in
        q_out.put((count, box_owners))


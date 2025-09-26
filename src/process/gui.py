from multiprocessing import Queue, shared_memory
import numpy as np
import cv2 as cv
from utils.paint import draw_fps
import time
from inference import IMAGE_SHAPE
from inference import LOC_LENGTH, CONF_LENGTH, LANDS_LENGTH, IMAGE_SHAPE
from constants import SYNC_FPS, GLOBAL_MESSAGE_LENGTH, MAX_PEOPLE

def gui(shms: tuple[str, ...], q_in: Queue):
    info_shape = (int((LOC_LENGTH + CONF_LENGTH + LANDS_LENGTH)/16800)*MAX_PEOPLE, )
    shm_global_msg, shm_frame_in, shm_info_in, _shm_vec_in = shms
    shm_frame_in = shared_memory.SharedMemory(name=shm_frame_in)
    shm_info_in = shared_memory.SharedMemory(name=shm_info_in)
    shm_global_msg = shared_memory.SharedMemory(name=shm_global_msg)

    frame_in = np.ndarray(IMAGE_SHAPE, dtype=np.uint8, buffer=shm_frame_in.buf)
    info_in = np.ndarray(info_shape, dtype=np.float32, buffer=shm_info_in.buf)
    global_message = np.ndarray((GLOBAL_MESSAGE_LENGTH), dtype=np.uint8, buffer=shm_global_msg.buf)

    time_prev, fps = time.time(), 0.0
    while global_message[0]:
        count = 0
        if SYNC_FPS:
            count = q_in.get()

        if(count>0):
            loc, conf, lands = np.split(info_in[0:count*15], [4 * count, 4 * count + 1 * count])
            loc = loc.reshape(count, 4).astype(int)
            conf = conf.reshape(count, 1)
            lands = lands.reshape(count, 10).astype(int)

            for i in range(count):
                if conf[i][0] < 0.6:
                    continue
                text = "{:.4f}".format(conf[i][0])
                cv.rectangle(frame_in, (loc[i][0], loc[i][1]), (loc[i][2], loc[i][3]), (0, 0, 255), 2)
                cx = loc[i][0]
                cy = loc[i][1] + 12
                cv.putText(frame_in, text, (cx, cy),
                            cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                cv.circle(frame_in, (lands[i][0], lands[i][1]), 1, (0, 0, 255), 4)
                cv.circle(frame_in, (lands[i][2], lands[i][3]), 1, (0, 255, 255), 4)
                cv.circle(frame_in, (lands[i][4], lands[i][5]), 1, (255, 0, 255), 4)
                cv.circle(frame_in, (lands[i][6], lands[i][7]), 1, (0, 255, 0), 4)
                cv.circle(frame_in, (lands[i][8], lands[i][9]), 1, (255, 0, 0), 4)

        fps, time_prev = draw_fps(frame_in, fps, time_prev)
        final_frame = cv.resize(frame_in, (1080, 720))
        cv.imshow("Webcam (press q or ESC to quit)", final_frame)
        key = cv.waitKey(1) & 0xFF
        # q or ESC
        if key == ord("q") or key == 27:
            global_message[0] = 0
            break


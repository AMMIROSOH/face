from multiprocessing import Queue, shared_memory
import numpy as np
import cv2 as cv
from utils.paint import draw_fps
import time
from inference import IMAGE_SHAPE
from inference import LOC_LENGTH, CONF_LENGTH, LANDS_LENGTH, IMAGE_SHAPE
from constants import GLOBAL_MESSAGE_LENGTH, MAX_PEOPLE, IMAGE_SHAPE_GUI

def gui(shms: tuple[str, ...], q_in: Queue):
    info_shape = (int((LOC_LENGTH + CONF_LENGTH + LANDS_LENGTH)/16800)*MAX_PEOPLE, )
    shm_global_msg, shm_frame_in, shm_info_in, shm_cap_gui_frame = shms
    shm_frame_in = shared_memory.SharedMemory(name=shm_frame_in)
    shm_info_in = shared_memory.SharedMemory(name=shm_info_in)
    shm_global_msg = shared_memory.SharedMemory(name=shm_global_msg)

    gui_frame_in = np.ndarray(IMAGE_SHAPE_GUI, dtype=np.uint8, buffer=shm_cap_gui_frame.buf)
    info_in = np.ndarray(info_shape, dtype=np.float32, buffer=shm_info_in.buf)
    global_message = np.ndarray((GLOBAL_MESSAGE_LENGTH), dtype=np.uint8, buffer=shm_global_msg.buf)

    gui_frame_temp = gui_frame_in.copy()
    time_prev, fps = time.time(), 0.0
    while global_message[0]:
        count, box_owners = q_in.get()
        gui_frame_temp[:] = gui_frame_in

        if(count>0):
            loc, lands, conf = np.split(info_in[0:count*15], [4 * count, 14 * count])
            loc = loc.reshape(count, 4).astype(int)
            lands = lands.reshape(count, 10).astype(int)
            conf = conf.reshape(count, 1)

            loc[:, 0] = loc[:, 0] * IMAGE_SHAPE_GUI[1] / IMAGE_SHAPE[1]
            loc[:, 2] = loc[:, 2] * IMAGE_SHAPE_GUI[1] / IMAGE_SHAPE[1]
            loc[:, 1] = loc[:, 1] * IMAGE_SHAPE_GUI[0] / IMAGE_SHAPE[0]
            loc[:, 3] = loc[:, 3] * IMAGE_SHAPE_GUI[0] / IMAGE_SHAPE[0]
            lands[:, 0::2] = lands[:, 0::2] * IMAGE_SHAPE_GUI[1] / IMAGE_SHAPE[1]
            lands[:, 1::2] = lands[:, 1::2] * IMAGE_SHAPE_GUI[0] / IMAGE_SHAPE[0]


            for i in range(count):
                if conf[i][0] < 0.6:
                    continue
                text = "{:.2f}".format(conf[i][0])
                cv.rectangle(gui_frame_temp, (loc[i][0], loc[i][1]), (loc[i][2], loc[i][3]), (0, 0, 255), 2)
                cx = loc[i][0]
                cy = loc[i][1] + 12
                if("unkown" not in box_owners[i]):
                    cv.putText(gui_frame_temp, box_owners[i], (cx, cy),
                                cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                # cv.putText(gui_frame_temp, text, (cx, cy + 15),
                #             cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                cv.circle(gui_frame_temp, (lands[i][0], lands[i][1]), 1, (0, 0, 255), 4)
                cv.circle(gui_frame_temp, (lands[i][2], lands[i][3]), 1, (0, 255, 255), 4)
                cv.circle(gui_frame_temp, (lands[i][4], lands[i][5]), 1, (255, 0, 255), 4)
                cv.circle(gui_frame_temp, (lands[i][6], lands[i][7]), 1, (0, 255, 0), 4)
                cv.circle(gui_frame_temp, (lands[i][8], lands[i][9]), 1, (255, 0, 0), 4)

        fps, time_prev = draw_fps(gui_frame_temp, fps, time_prev)
        cv.imshow("Webcam (press q or ESC to quit)", gui_frame_temp)
        key = cv.waitKey(1) & 0xFF
        # q or ESC
        if key == ord("q") or key == 27:
            global_message[0] = 0
            break


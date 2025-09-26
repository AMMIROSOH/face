from inference import Inference, LOC_LENGTH, CONF_LENGTH, LANDS_LENGTH, IMAGE_SHAPE
from utils.retinaface import PriorBox, decode, decode_landm
from models.retinaface import cfg_re50
from utils.paint import draw_fps
from constants import SYNC_FPS, GLOBAL_MESSAGE_LENGTH, MAX_PEAPLE, CAMERA_URL
from multiprocessing import Queue, Process, shared_memory
import numpy as np
import cv2 as cv
import torchvision
import torch
import time

INFO_SHAPE = (int((LOC_LENGTH + CONF_LENGTH + LANDS_LENGTH)/16800)*MAX_PEAPLE, )


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


def face_detection(shms: tuple[str, ...], q_in: Queue, q_out: Queue):
    retinaModel = Inference(model="RetinaFace-R50_fp16")
    shm_global_msg, shm_frame_in, shm_frame_out, shm_info_out = shms

    shm_global_msg = shared_memory.SharedMemory(name=shm_global_msg)
    shm_frame_in = shared_memory.SharedMemory(name=shm_frame_in)
    shm_frame_out = shared_memory.SharedMemory(name=shm_frame_out)
    shm_info_out = shared_memory.SharedMemory(name=shm_info_out)

    global_message = np.ndarray((GLOBAL_MESSAGE_LENGTH), dtype=np.uint8, buffer=shm_global_msg.buf)
    frame_in = np.ndarray(IMAGE_SHAPE, dtype=np.uint8, buffer=shm_frame_in.buf)
    frame_out = np.ndarray(IMAGE_SHAPE, dtype=np.uint8, buffer=shm_frame_out.buf)
    info_out = np.ndarray((LOC_LENGTH + CONF_LENGTH + LANDS_LENGTH), dtype=np.float32, buffer=shm_info_out.buf)

    while global_message[0]:
        if SYNC_FPS:
            q_in.get()

        # todo: transfer this to capture
        frame = np.array(frame_in, dtype=np.int32)
        frame -= (104, 117, 123)
        frame = frame.transpose(2, 0, 1)

        loc, conf, landms = retinaModel.infer(frame)
        # todo: transfer this to infer
        frame_out[:] = frame_in

        # todo: concat may not be needed
        info_out[:] = np.concatenate((loc[0], conf[0], landms[0]))

        if SYNC_FPS:
            q_out.put(0)


def postprocess(shms: tuple[str, ...], q_in: Queue, q_out: Queue):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shm_global_msg, shm_frame_in, shm_info_in, shm_frame_out, shm_info_out = shms
    shm_frame_in = shared_memory.SharedMemory(name=shm_frame_in)
    shm_info_in = shared_memory.SharedMemory(name=shm_info_in)
    shm_frame_out = shared_memory.SharedMemory(name=shm_frame_out)
    shm_info_out = shared_memory.SharedMemory(name=shm_info_out)
    shm_global_msg = shared_memory.SharedMemory(name=shm_global_msg)

    frame_in = np.ndarray(IMAGE_SHAPE, dtype=np.uint8, buffer=shm_frame_in.buf)
    info_in = np.ndarray((LOC_LENGTH + CONF_LENGTH + LANDS_LENGTH), dtype=np.float32, buffer=shm_info_in.buf)
    frame_out = np.ndarray(IMAGE_SHAPE, dtype=np.uint8, buffer=shm_frame_out.buf)
    info_out = np.ndarray(INFO_SHAPE, dtype=np.float32, buffer=shm_info_out.buf)
    global_message = np.ndarray((GLOBAL_MESSAGE_LENGTH), dtype=np.uint8, buffer=shm_global_msg.buf)

    im_height, im_width, _ = frame_in.shape
    priorbox = PriorBox(cfg_re50, image_size=(im_height, im_width))
    priors = priorbox.forward().to(device)

    scale_box = torch.tensor([im_width, im_height, im_width, im_height], device=device, dtype=torch.float32)
    scale_lands = torch.tensor([im_width, im_height] * 5, device=device, dtype=torch.float32)

    while global_message[0]:
        if SYNC_FPS:
            q_in.get()

        loc, conf, lands = np.split(info_in, [LOC_LENGTH, LOC_LENGTH + CONF_LENGTH])
        loc = loc.reshape(16800, 4)
        conf = conf.reshape(16800, 2)
        lands = lands.reshape(16800, 10)

        scores_all = conf[:, 1]
        inds = np.where(scores_all > 0.02)[0]
        # todo: select topKs only if its needed < len(x)
        order = inds[np.argsort(scores_all[inds])[::-1][:5000]]

        count = 0
        if order.size != 0:
            loc = torch.from_numpy(loc[order]).to(device=device, dtype=torch.float32)
            lands = torch.from_numpy(lands[order]).to(device=device, dtype=torch.float32) 
            scores = torch.from_numpy(scores_all[order]).to(device=device, dtype=torch.float32) 
            priors_filtered = priors[torch.from_numpy(order)]
            priors_filtered = priors_filtered.to(device)

            boxes = decode(loc, priors_filtered, cfg_re50["variance"])
            boxes = boxes * scale_box

            landms = decode_landm(lands, priors_filtered, cfg_re50["variance"])
            landms = landms * scale_lands

            keep = torchvision.ops.nms(boxes, scores, 0.4)
            if keep.numel() > MAX_PEAPLE:
                keep = keep[:MAX_PEAPLE]
            boxes = boxes[keep].cpu().numpy()
            scores = scores[keep].cpu().numpy()
            landms = landms[keep].cpu().numpy()

            count = len(boxes)
            info = np.concatenate((boxes.flatten(), scores.flatten(), landms.flatten()))
            info_out[0:len(info)] = info
        frame_out[:] = frame_in
        q_out.put(count)


def face_recognition(shms: tuple[str, ...], q_in: Queue, q_out: Queue):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shm_global_msg, shm_frame_in, shm_info_in, shm_frame_out, shm_info_out, shm_vec_out = shms
    shm_frame_in = shared_memory.SharedMemory(name=shm_frame_in)
    shm_info_in = shared_memory.SharedMemory(name=shm_info_in)
    shm_frame_out = shared_memory.SharedMemory(name=shm_frame_out)
    shm_info_out = shared_memory.SharedMemory(name=shm_info_out)
    shm_global_msg = shared_memory.SharedMemory(name=shm_global_msg)

    frame_in = np.ndarray(IMAGE_SHAPE, dtype=np.uint8, buffer=shm_frame_in.buf)
    info_in = np.ndarray(INFO_SHAPE, dtype=np.float32, buffer=shm_info_in.buf)
    frame_out = np.ndarray(IMAGE_SHAPE, dtype=np.uint8, buffer=shm_frame_out.buf)
    info_out = np.ndarray(INFO_SHAPE, dtype=np.float32, buffer=shm_info_out.buf)
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


def gui(shms: tuple[str, ...], q_in: Queue):
    shm_global_msg, shm_frame_in, shm_info_in, _shm_vec_in = shms
    shm_frame_in = shared_memory.SharedMemory(name=shm_frame_in)
    shm_info_in = shared_memory.SharedMemory(name=shm_info_in)
    shm_global_msg = shared_memory.SharedMemory(name=shm_global_msg)

    frame_in = np.ndarray(IMAGE_SHAPE, dtype=np.uint8, buffer=shm_frame_in.buf)
    info_in = np.ndarray(INFO_SHAPE, dtype=np.float32, buffer=shm_info_in.buf)
    global_message = np.ndarray((GLOBAL_MESSAGE_LENGTH), dtype=np.uint8, buffer=shm_global_msg.buf)

    im_height, im_width, _ = frame_in.shape

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


if __name__ == "__main__":
    size_global_message = int(np.prod((GLOBAL_MESSAGE_LENGTH)) * np.dtype(np.uint8).itemsize)
    size_frame = int(np.prod(IMAGE_SHAPE) * np.dtype(np.uint8).itemsize)
    size_info = int(np.prod(LOC_LENGTH + CONF_LENGTH + LANDS_LENGTH) * np.dtype(np.float32).itemsize)
    size_filtered_info = int(np.prod(INFO_SHAPE[0]) * np.dtype(np.float32).itemsize)
    size_vec = int(np.prod(512*MAX_PEAPLE) * np.dtype(np.float32).itemsize)

    shm_global_msg = shared_memory.SharedMemory(create=True, size=size_global_message)
    shm_cap_frame = shared_memory.SharedMemory(create=True, size=size_frame)
    shm_fd_frame = shared_memory.SharedMemory(create=True, size=size_frame)
    shm_fd_info = shared_memory.SharedMemory(create=True, size=size_info)
    shm_fr_frame = shared_memory.SharedMemory(create=True, size=size_frame)
    shm_fr_info = shared_memory.SharedMemory(create=True, size=size_filtered_info)
    shm_gui_frame = shared_memory.SharedMemory(create=True, size=size_frame)
    shm_gui_info = shared_memory.SharedMemory(create=True, size=size_filtered_info)
    shm_gui_vec = shared_memory.SharedMemory(create=True, size=size_vec)

    shm_global_msg.buf[0] = 1

    q1, q2, q3, q4 = Queue(), Queue(), Queue(), Queue()

    shms_capture = (shm_global_msg.name, shm_cap_frame.name)
    shms_fd = (shm_global_msg.name, shm_cap_frame.name, shm_fd_frame.name, shm_fd_info.name)
    shms_post = (shm_global_msg.name, shm_fd_frame.name, shm_fd_info.name, shm_fr_frame.name, shm_fr_info.name)
    shms_fr = (shm_global_msg.name, shm_fr_frame.name, shm_fr_info.name, shm_gui_frame.name, shm_gui_info.name, shm_gui_vec.name)
    shms_gui = (shm_global_msg.name, shm_gui_frame.name, shm_gui_info.name, shm_gui_vec.name)

    p1 = Process(target=capture, args=(shms_capture, q1))
    p2 = Process(target=face_detection, args=(shms_fd, q1, q2))
    p3 = Process(target=postprocess, args=(shms_post, q2, q3))
    p4 = Process(target=face_recognition, args=(shms_fr, q3, q4))
    p5 = Process(target=gui, args=(shms_gui, q4))

    p1.start(); p2.start(); p3.start(); p4.start(); p5.start()
    p1.join(); p2.join(); p3.join(); p4.join(); p5.join()

    shm_global_msg.close(); shm_global_msg.unlink()
    shm_cap_frame.close(); shm_cap_frame.unlink()
    shm_fd_frame.close(); shm_fd_frame.unlink()
    shm_fd_info.close(); shm_fd_info.unlink()
    shm_fr_frame.close(); shm_fr_frame.unlink()
    shm_fr_info.close(); shm_fr_info.unlink()
    shm_gui_frame.close(); shm_gui_frame.unlink()
    shm_gui_info.close(); shm_gui_info.unlink()
    shm_gui_vec.close(); shm_gui_vec.unlink()

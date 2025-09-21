from retinaface import RetinFace, LOC_LENGTH, CONF_LENGTH, LANDS_LENGTH, IMAGE_SHAPE
from multiprocessing import shared_memory
import multiprocessing as mp
from utils.utils import draw_fps
from utils.retinaface import decode, decode_landm, py_cpu_nms
from models.config import cfg_re50
from utils.priorbox import PriorBox
import numpy as np
import cv2 as cv
import torchvision
import torch
import time

def capture(shm, q_out):
    shm = shared_memory.SharedMemory(name=shm)
    frame = np.ndarray(IMAGE_SHAPE, dtype=np.uint8, buffer=shm.buf)

    cap = cv.VideoCapture("http://192.168.1.102:4747/video/force/1280x720")
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera")
        q_out.put("stop")
        return

    try:
        while True:
            ret, frame_temp = cap.read()
            cv.resize(frame_temp, (640, 640), frame)
    except Exception:
        print(Exception)
        q_out.put("stop")

def inference(shms: tuple, q_in, q_out):
    retinaModel = RetinFace()
    shm_frame_in, shm_frame_out, shm_loc_out, shm_conf_out, shm_lands_out = shms

    shm_frame_in = shared_memory.SharedMemory(name=shm_frame_in)
    shm_frame_out = shared_memory.SharedMemory(name=shm_frame_out)
    shm_loc_out = shared_memory.SharedMemory(name=shm_loc_out)
    shm_conf_out = shared_memory.SharedMemory(name=shm_conf_out)
    shm_lands_out = shared_memory.SharedMemory(name=shm_lands_out)

    frame_in = np.ndarray(IMAGE_SHAPE, dtype=np.uint8, buffer=shm_frame_in.buf)
    frame_out = np.ndarray(IMAGE_SHAPE, dtype=np.uint8, buffer=shm_frame_out.buf)
    loc_out = np.ndarray((LOC_LENGTH), dtype=np.float32, buffer=shm_loc_out.buf)
    conf_out = np.ndarray((CONF_LENGTH), dtype=np.float32, buffer=shm_conf_out.buf)
    lands_out = np.ndarray((LANDS_LENGTH), dtype=np.float32, buffer=shm_lands_out.buf)

    step = 0
    step_size = 10

    while True:
        if(step%step_size==0):
            step = 0
            msg = q_in.get()
            if msg == "stop":
                break
        step+=1

        frame = np.array(frame_in, dtype=np.int32)            
        frame -= (104, 117, 123)
        frame = frame.transpose(2, 0, 1)
        
        loc, conf, landms = retinaModel.infer(frame)
        frame_out[:] = frame_in
        loc_out[:] = loc[0]
        conf_out[:] = conf[0]
        lands_out[:] = landms[0]

def postprocess(shms: tuple, q_in):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shm_frame_in, shm_loc_in, shm_conf_in, shm_lands_in = shms
    shm_frame_in = shared_memory.SharedMemory(name=shm_frame_in)
    shm_loc_in = shared_memory.SharedMemory(name=shm_loc_in)
    shm_conf_in = shared_memory.SharedMemory(name=shm_conf_in)
    shm_lands_in = shared_memory.SharedMemory(name=shm_lands_in)

    frame_in = np.ndarray(IMAGE_SHAPE, dtype=np.uint8, buffer=shm_frame_in.buf)
    loc_in = np.ndarray((LOC_LENGTH), dtype=np.float32, buffer=shm_loc_in.buf)
    conf_in = np.ndarray((CONF_LENGTH), dtype=np.float32, buffer=shm_conf_in.buf)
    lands_in = np.ndarray((LANDS_LENGTH), dtype=np.float32, buffer=shm_lands_in.buf)

    im_height, im_width, _ = frame_in.shape
    priorbox = PriorBox(cfg_re50, image_size=(im_height, im_width))
    priors = priorbox.forward().to(device)

    scale_box = torch.tensor([im_width, im_height, im_width, im_height], device=device, dtype=torch.float32)
    scale_lands = torch.tensor([im_width, im_height] * 5, device=device, dtype=torch.float32)

    time_prev, fps = time.time(), 0.0

    step = 0
    step_size = 10

    while True:
        if(step%step_size==0):
            step = 0
            msg = q_in.get()
            if msg == "stop":
                break
        step+=1

        loc = loc_in.reshape((1, 16800, 4)).squeeze(0)
        conf = conf_in.reshape((1, 16800, 2)).squeeze(0)
        lands = lands_in.reshape((1, 16800, 10)).squeeze(0)

        scores_all = conf[:, 1]
        inds = np.where(scores_all > 0.02)[0]
        # todo: select topKs only if its needed < len(x)
        order = inds[np.argsort(scores_all[inds])[::-1][:5000]]
        
        if order.size != 0:
            loc = torch.from_numpy(loc[order]).to(device=device, dtype=torch.float32)
            lands = torch.from_numpy(lands[order]).to(device=device, dtype=torch.float32) 
            scores = torch.from_numpy(scores_all[order])

            # todo: remove copy()
            priors_filtered = priors[torch.from_numpy(order)]
            priors_filtered = priors_filtered.to(device)

            boxes = decode(loc, priors_filtered, cfg_re50['variance'])
            boxes = boxes * scale_box

            landms = decode_landm(lands, priors_filtered, cfg_re50['variance'])
            landms = landms * scale_lands

            keep = torchvision.ops.nms(boxes, scores, 0.4)
            if keep.numel() > 10:
                keep = keep[:10]            
            boxes = boxes[keep].cpu().numpy()
            scores = scores[keep].cpu().numpy()
            landms = landms[keep].cpu().numpy()

            # # do NMS
            # dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            # keep = py_cpu_nms(dets, 0.4)
            # # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            # dets = dets[keep, :]
            # landms = landms[keep]

            dets = np.hstack((scores, scores[:, np.newaxis])).astype(np.float32, copy=False)
            # concatenate landms
            dets = np.concatenate((dets, landms), axis=1)
            for d in dets:
                if d[4] < 0.6:
                    continue
                text = "{:.4f}".format(d[4])
                d = list(map(int, d))
                cv.rectangle(frame_in, (d[0], d[1]), (d[2], d[3]), (0, 0, 255), 2)
                cx = d[0]
                cy = d[1] + 12
                cv.putText(frame_in, text, (cx, cy),
                            cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv.circle(frame_in, (d[5], d[6]), 1, (0, 0, 255), 4)
                cv.circle(frame_in, (d[7], d[8]), 1, (0, 255, 255), 4)
                cv.circle(frame_in, (d[9], d[10]), 1, (255, 0, 255), 4)
                cv.circle(frame_in, (d[11], d[12]), 1, (0, 255, 0), 4)
                cv.circle(frame_in, (d[13], d[14]), 1, (255, 0, 0), 4)

        fps, time_prev = draw_fps(frame_in, fps, time_prev)
        final_frame = cv.resize(frame_in, (1080, 720))
        cv.imshow("Webcam (press q or ESC to quit)", final_frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # q or ESC
            break

if __name__ == "__main__":
    size = int(np.prod(IMAGE_SHAPE) * np.dtype(np.uint8).itemsize)
    size_loc = int(np.prod(LOC_LENGTH) * np.dtype(np.float32).itemsize)
    size_conf = int(np.prod(CONF_LENGTH) * np.dtype(np.float32).itemsize)
    size_lands = int(np.prod(LANDS_LENGTH) * np.dtype(np.float32).itemsize)

    shm_cap_frame = shared_memory.SharedMemory(create=True, size=size)
    shm_inference_frame = shared_memory.SharedMemory(create=True, size=size)
    shm_inference_loc = shared_memory.SharedMemory(create=True, size=size_loc)
    shm_inference_conf = shared_memory.SharedMemory(create=True, size=size_conf)
    shm_inference_lands = shared_memory.SharedMemory(create=True, size=size_lands)

    q1 = mp.Queue()
    q2 = mp.Queue()

    shms_inference = (shm_cap_frame.name, shm_inference_frame.name, shm_inference_loc.name, shm_inference_conf.name, shm_inference_lands.name)
    shms_post = (shm_inference_frame.name, shm_inference_loc.name, shm_inference_conf.name, shm_inference_lands.name)

    p1 = mp.Process(target=capture, args=(shm_cap_frame.name, q1))
    p2 = mp.Process(target=inference, args=(shms_inference, q1, q2))
    p3 = mp.Process(target=postprocess, args=(shms_post, q2))

    p1.start(); p2.start(); p3.start()
    p1.join(); p2.join(); p3.join()

    shm_cap_frame.close(); shm_cap_frame.unlink()
    shm_inference_frame.close(); shm_inference_frame.unlink()
    shm_inference_loc.close(); shm_inference_loc.unlink()
    shm_inference_conf.close(); shm_inference_conf.unlink()
    shm_inference_lands.close(); shm_inference_lands.unlink()

from multiprocessing import Queue, shared_memory
import numpy as np
import torchvision
import torch
from models.retinaface import cfg_re50
from inference import LOC_LENGTH, CONF_LENGTH, LANDS_LENGTH, IMAGE_SHAPE
from constants import SYNC_FPS, GLOBAL_MESSAGE_LENGTH, MAX_PEOPLE
from utils.retinaface import PriorBox, decode, decode_landm

def face_candidates(shms: tuple[str, ...], q_in: Queue, q_out: Queue):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info_shape = (int((LOC_LENGTH + CONF_LENGTH + LANDS_LENGTH)/16800) * MAX_PEOPLE, )

    shm_global_msg, shm_frame_in, shm_info_in, shm_frame_out, shm_info_out = shms
    shm_frame_in = shared_memory.SharedMemory(name=shm_frame_in)
    shm_info_in = shared_memory.SharedMemory(name=shm_info_in)
    shm_frame_out = shared_memory.SharedMemory(name=shm_frame_out)
    shm_info_out = shared_memory.SharedMemory(name=shm_info_out)
    shm_global_msg = shared_memory.SharedMemory(name=shm_global_msg)

    frame_in = np.ndarray(IMAGE_SHAPE, dtype=np.uint8, buffer=shm_frame_in.buf)
    info_in = np.ndarray((LOC_LENGTH + CONF_LENGTH + LANDS_LENGTH), dtype=np.float32, buffer=shm_info_in.buf)
    frame_out = np.ndarray(IMAGE_SHAPE, dtype=np.uint8, buffer=shm_frame_out.buf)
    info_out = np.ndarray(info_shape, dtype=np.float32, buffer=shm_info_out.buf)
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
            if keep.numel() > MAX_PEOPLE:
                keep = keep[:MAX_PEOPLE]
            boxes = boxes[keep].cpu().numpy()
            scores = scores[keep].cpu().numpy()
            landms = landms[keep].cpu().numpy()

            count = len(boxes)
            info = np.concatenate((boxes.flatten(), scores.flatten(), landms.flatten()))
            info_out[0:len(info)] = info
        frame_out[:] = frame_in
        q_out.put(count)

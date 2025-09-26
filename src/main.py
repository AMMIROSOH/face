from multiprocessing import Queue, Process, shared_memory
import numpy as np
from inference import LOC_LENGTH, CONF_LENGTH, LANDS_LENGTH, IMAGE_SHAPE
from constants import GLOBAL_MESSAGE_LENGTH, MAX_PEAPLE
from process import capture, face_detection, face_candidates, face_recognition, gui


if __name__ == "__main__":
    info_shape = (int((LOC_LENGTH + CONF_LENGTH + LANDS_LENGTH)/16800)*MAX_PEAPLE, )
    
    size_global_message = int(np.prod((GLOBAL_MESSAGE_LENGTH)) * np.dtype(np.uint8).itemsize)
    size_frame = int(np.prod(IMAGE_SHAPE) * np.dtype(np.uint8).itemsize)
    size_info = int(np.prod(LOC_LENGTH + CONF_LENGTH + LANDS_LENGTH) * np.dtype(np.float32).itemsize)
    size_filtered_info = int(np.prod(info_shape[0]) * np.dtype(np.float32).itemsize)
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
    p3 = Process(target=face_candidates, args=(shms_post, q2, q3))
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

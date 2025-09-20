import multiprocessing as mp
import numpy as np
from multiprocessing import shared_memory
import time

# ----- PROCESS FUNCTIONS -----

def capture(shm_name, shape, dtype, q_out):
    shm = shared_memory.SharedMemory(name=shm_name)
    frame = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    for i in range(5):
        frame[:] = np.random.randint(0, 255, shape, dtype=dtype)
        print(f"[Capture] wrote frame {i}")
        q_out.put("ready")  # tell inference a new frame is ready
        time.sleep(0.5)
    q_out.put("stop")


def inference(shm_name, shape, dtype, q_in, q_out, shm2_name):
    shm_in = shared_memory.SharedMemory(name=shm_name)
    frame_in = np.ndarray(shape, dtype=dtype, buffer=shm_in.buf)

    shm_out = shared_memory.SharedMemory(name=shm2_name)
    frame_out = np.ndarray(shape, dtype=dtype, buffer=shm_out.buf)

    while True:
        msg = q_in.get()
        if msg == "stop":
            q_out.put(("stop", None))
            break
        # fake inference: compute mean
        result = frame_in.mean()
        frame_out[:] = frame_in  # copy frame to output buffer
        print(f"[Inference] processed frame, mean={result:.2f}")
        q_out.put(("ready", result))


def postprocess(shm_name, shape, dtype, q_in):
    shm = shared_memory.SharedMemory(name=shm_name)
    frame = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    while True:
        msg, result = q_in.get()
        if msg == "stop":
            break
        print(f"[Postprocess] got frame with result={result:.2f} | frame[0,0]={frame[0,0]}")


# ----- MAIN -----

if __name__ == "__main__":
    shape = (480, 640, 3)
    dtype = np.uint8
    size = int(np.prod(shape) * np.dtype(dtype).itemsize)

    shm1 = shared_memory.SharedMemory(create=True, size=size)  # capture → inference
    shm2 = shared_memory.SharedMemory(create=True, size=size)  # inference → postproc

    q1 = mp.Queue()
    q2 = mp.Queue()

    p1 = mp.Process(target=capture, args=(shm1.name, shape, dtype, q1))
    p2 = mp.Process(target=inference, args=(shm1.name, shape, dtype, q1, q2, shm2.name))
    p3 = mp.Process(target=postprocess, args=(shm2.name, shape, dtype, q2))

    p1.start(); p2.start(); p3.start()
    p1.join(); p2.join(); p3.join()

    shm1.close(); shm1.unlink()
    shm2.close(); shm2.unlink()

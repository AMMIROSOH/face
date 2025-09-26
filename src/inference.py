import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA driver
import numpy.typing as npt
from typing import Literal
import numpy as np
from constants import MAX_PEAPLE

LOC_LENGTH, CONF_LENGTH, LANDS_LENGTH = 67200, 33600, 168000
IMAGE_SHAPE = (640, 640, 3)

class Inference:
    def __init__(self, model: Literal["RetinaFace-R50_fp16", "arcface-r100-glint360k_fp16"]):
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(f"checkpoints/{model}.engine", "rb") as f:
            runtime = trt.Runtime(trt_logger)
            engine = runtime.deserialize_cuda_engine(f.read())

        self.model = model
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = ([], [], [], cuda.Stream())

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            size = trt.volume(engine.get_tensor_shape(tensor_name))
            if(model=="arcface-r100-glint360k_fp16"):
                size = trt.volume((MAX_PEAPLE, 3, 112, 112))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append((host_mem, device_mem))
            else:
                self.outputs.append((host_mem, device_mem))

    def infer(self, image: npt.ArrayLike):
        # Copy input data to host buffer
        np.copyto(self.inputs[0][0], image.ravel())

        # Transfer to GPU
        cuda.memcpy_htod_async(self.inputs[0][1], self.inputs[0][0], self.stream)

        # Execute
        self.context.execute_v2(self.bindings)

        # Transfer outputs back
        if("Retina" in self.model):
            cuda.memcpy_dtoh_async(self.outputs[0][0], self.outputs[0][1], self.stream)
            cuda.memcpy_dtoh_async(self.outputs[1][0], self.outputs[1][1], self.stream)
            cuda.memcpy_dtoh_async(self.outputs[2][0], self.outputs[2][1], self.stream)
        else:
            cuda.memcpy_dtoh_async(self.outputs[0][0], self.outputs[0][1], self.stream)
        self.stream.synchronize()

        return self.outputs

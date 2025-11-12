import numpy as np
from cv2 import estimateAffinePartial2D, LMEDS
IMG_SIZE = 112 
LANDS_TEMPLATE = np.array([
    [30.2946, 51.6963],  # left eye
    [65.5318, 51.5014],  # right eye
    [48.0252, 71.7366],  # nose
    [33.5493, 92.3655],  # left mouth
    [62.7299, 92.2041],  # right mouth
], dtype=np.float32)

def estimate_norm(landmarks, template=LANDS_TEMPLATE):
    src = np.array(landmarks, dtype=np.float32)
    dst = template.copy()
    tform = estimateAffinePartial2D(src.reshape((5, 2)), dst, method=LMEDS)[0]
    return tform
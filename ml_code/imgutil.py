import numpy as np

def to_8bit(img):
    """
    Converts image `img` to 8-bit
    """
    bit8 = np.zeros(img.shape, dtype=np.uint8)
    bit8[:,:,:] = np.round(np.clip(img, a_max=1, a_min=0) * 255)
    return bit8

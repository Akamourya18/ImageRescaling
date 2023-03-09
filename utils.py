import numpy as np
import uuid
from imageio import imread
from ml_code.srcnn import SRCNN2x, SRCNN4x
from ml_code.vdsr import VDSR2x, VDSR4x
from skimage.metrics import structural_similarity as SSIM
from typing import Tuple

def random_name() -> str:
    """
    Returns a random unique filename using Universally unique identifier (UUID) method
    """
    return uuid.uuid4().hex

def upscale_image(original_fname: str,
                  upscaled_fname: str,
                  compressed_fname: str,
                  algorithm: str,
                  scale: str,
                  save_bicubic: bool = False,
                  bicubic_fname: str = None) -> Tuple[int]:
    """
    Upscales an image using specified algorthm and saves the result

    Parameters
    ------------
    original_fname : filepath to original image
    upscaled_fname : filepath for saving algorithmically upscaled image
    compressed_fname : filepath to compressed image
    algorithm : upscaling algorithm name
    scale : scaling factor
    save_bicubic : whether to save the upscaled bicubic version too
    bicubic_fname :  filepath for saving bicubic version, needed to be specified iff save_bicubic is True

    Returns
    ------------
    Tuple containing (width_original, height_original, width_upscaled, height_upscaled)
    """
    ALGOS = {
        "srcnn" : {
            "4x" : SRCNN4x(),
            "2x" : SRCNN2x()
        },
        "vdsr" : {
            "4x" : VDSR4x(),
            "2x" : VDSR2x()
        }
    }

    return ALGOS[algorithm][scale].apply(original_fname,
                                  upscaled_fname,
                                  compressed_fname,
                                  save_bicubic=save_bicubic,
                                  bicubic_nm=bicubic_fname)

def get_psnr(original, interpol):
    """
    Calculates Peak-Signal-to-Noise-Ratio (PSNR) for `interpol` image against `original` image. 
    Images should be in 8-bit format 
    """
    orig_img = imread(original)
    intp_img = imread(interpol)
    
    mse = np.mean((orig_img - intp_img) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def get_ssim(original , interpol):
    """
    Calculates Structural Similarity Index (SSIM) for `interpol` image against `original` image. 
    Images should be in 8-bit format 
    """
    return SSIM(imread(original), 
                imread(interpol), 
                data_range=255, 
                multichannel=True)

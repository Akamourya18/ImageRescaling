import keras
import numpy as np
from imageio import imread, imsave
from keras.models import load_model
from skimage.transform import resize
from typing import Tuple
from .imgutil import to_8bit

class VDSR2x:
    """
    Very Deep Super Resolution (VDSR) algorithm for 2x upscaling
    """

    MODEL_FILE = "model-vdsr-2x.h5"
    model = load_model(f"models/{MODEL_FILE}")

    @staticmethod
    def gen_output(inputs: np.array,
                   model: keras.models.Model) -> np.array:
        """
        Returns `model` output for a given sequence of `inputs`
        """
        preds = model.predict(inputs)
        return preds + inputs

    def apply(self,
              original_nm: str,
              upscaled_nm: str,
              compressed_nm: str,
              save_bicubic: bool = False,
              bicubic_nm: str = None) -> Tuple[int]:
        """
        Apply VDSR algorithm to image

        Parameters
        ------------
        original_nm: filepath to original image
        upscaled_nm: filepath for saving model enhanced image
        compressed_nm: filepath to compressed image
        save_bicubic: whether to save the bicubic upscaled version as well
        bicubic_nm: filepath for saving bicubic version, needed to be specified iff save_bicubic is True

        Returns
        ------------
        Tuple containing (width_original, height_original, width_upscaled, height_upscaled)
        """
        img_orig = imread(original_nm)
        w_orig, h_orig, _ = img_orig.shape

        # compressed downscaled
        img_comp = resize(img_orig, (w_orig//2, h_orig//2, 3), order=3)
        imsave(im=to_8bit(img_comp), uri=compressed_nm)

        # bicubic upscale
        img_bicu = resize(img_comp, (w_orig, h_orig, 3), order=3)

        if save_bicubic and bicubic_nm is not None:
            imsave(im=to_8bit(img_bicu), uri=bicubic_nm)

        # model enhanced
        img_upsc = __class__.gen_output(img_bicu.reshape(1, *img_bicu.shape), __class__.model)[0]
        img_upsc = np.clip(img_upsc, a_max=1, a_min=0)
        imsave(im=to_8bit(img_upsc), uri=upscaled_nm)

        return w_orig, h_orig, w_orig//2, h_orig//2


class VDSR4x:
    """
    Very Deep Super Resolution (VDSR) algorithm for 4x upscaling
    """

    MODEL_FILE = "model-vdsr-4x.h5"
    model = load_model(f"models/{MODEL_FILE}")

    @staticmethod
    def gen_output(inputs: np.array,
                   model: keras.models.Model) -> np.array:
        """
        Returns `model` output for a given sequence of `inputs`
        """
        preds = model.predict(inputs)
        return preds + inputs

    def apply(self,
              original_nm: str,
              upscaled_nm: str,
              compressed_nm: str,
              save_bicubic: bool = False,
              bicubic_nm: str = None) -> Tuple[int]:
        """
        Apply VDSR algorithm to image

        Parameters
        ------------
        original_nm: filepath to original image
        upscaled_nm: filepath for saving model enhanced image
        compressed_nm: filepath to compressed image
        save_bicubic: whether to save the bicubic upscaled version as well
        bicubic_nm: filepath for saving bicubic version, needed to be specified iff save_bicubic is True

        Returns
        ------------
        Tuple containing (width_original, height_original, width_upscaled, height_upscaled)
        """
        img_orig = imread(original_nm)
        w_orig, h_orig, _ = img_orig.shape

        # compressed downscaled
        img_comp = resize(img_orig, (w_orig//4, h_orig//4, 3), order=3)
        imsave(im=to_8bit(img_comp), uri=compressed_nm)
        
        # bicubic upscale
        img_bicu = resize(img_comp, (w_orig, h_orig, 3), order=3)

        if save_bicubic and bicubic_nm is not None:
            imsave(im=to_8bit(img_bicu), uri=bicubic_nm)

        # model enhanced
        img_upsc = __class__.gen_output(img_bicu.reshape(1, *img_bicu.shape), __class__.model)[0]
        img_upsc = np.clip(img_upsc, a_max=1, a_min=0)
        imsave(im=to_8bit(img_upsc), uri=upscaled_nm)

        return w_orig, h_orig, w_orig//4, h_orig//4

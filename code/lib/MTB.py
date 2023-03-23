import cv2 as cv
import numpy as np
from .utility import shift, BGR2GRAY
from numpy.typing import NDArray


def bit_and_Xor(img: NDArray[np.uint8], tolerance: float = 4) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    threshold = np.median(img)
    bitmap = img < threshold
    mask = (img < (threshold-tolerance)) | (img > (threshold+tolerance))
    return bitmap, mask


def eval(bitmap_s: NDArray[np.bool_], bitmap_t: NDArray[np.bool_], mask_s: NDArray[np.bool_], mask_t: NDArray[np.bool_]) -> int:
    t = np.logical_xor(bitmap_s , bitmap_t)
    t = t & mask_s
    t = t & mask_t
    return np.sum(t, dtype=int)


def find_offset(basic: NDArray[np.uint8], img: NDArray[np.uint8]) -> tuple[int, int]:

    if basic.shape[0] <= 2 or basic.shape[1] <= 2:
        return (0, 0)

    (lx, ly) = find_offset(cv.pyrDown(basic), cv.pyrDown(basic))

    lx, ly = lx*2, ly*2

    bitmap_s, mask_s = bit_and_Xor(img)
    bitmap_t, mask_t = bit_and_Xor(basic)

    
    best_loss, best_x, best_y = 1e8, 0, 0

    for x in range(-1, 2):
        for y in range(-1, 2):
            sft_bitmap = shift(bitmap_s, (lx+x,ly+y))
            sft_mask = shift(mask_s, (lx+x,ly+y))
            
            loss = eval(sft_bitmap , bitmap_t , sft_mask , mask_t)
            if loss < best_loss:
                best_loss = loss
                best_x = lx+x
                best_y = ly+y

    return best_x, best_y


def mtb(basic: NDArray[np.uint8], images: list[NDArray[np.uint8]]) -> list[NDArray[np.uint8]]:
    sft_imgs = []
    g_basic = BGR2GRAY(basic)
    for image in images:
        g_image = BGR2GRAY(image)
        offset = find_offset(g_basic, g_image)
        print(f"Offset = {offset}")
        sft_img = shift(image, offset)
        sft_imgs.append(sft_img)
    return sft_imgs

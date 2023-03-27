import cv2 as cv
import numpy as np
from .utility import shift, BGR2GRAY
from numpy.typing import NDArray
import math


def bit_and_Xor(img: NDArray[np.uint8], tolerance: float = 8) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    threshold = np.median(img)
    bitmap = img < threshold
    mask = (img < (threshold-tolerance)) | (img > (threshold+tolerance))
    return bitmap, mask


def eval(bitmap_s: NDArray[np.bool_], bitmap_t: NDArray[np.bool_], mask_s: NDArray[np.bool_], mask_t: NDArray[np.bool_]) -> int:
    t = np.logical_xor(bitmap_s , bitmap_t)
    t = t & mask_s
    t = t & mask_t
    return np.sum(t, dtype=int)


def find_offset(basic: NDArray[np.uint8], img: NDArray[np.uint8], search_range: int) -> tuple[int, int]:

    if search_range <= 0:
        return (0, 0)

    (lx, ly) = find_offset(cv.pyrDown(basic), cv.pyrDown(img), search_range-1)

    lx, ly = lx*2, ly*2

    bitmap_s, mask_s = bit_and_Xor(img)
    bitmap_t, mask_t = bit_and_Xor(basic)

    
    best_loss, best_x, best_y = 1e8, 0, 0

    xyrange = [0, 1, -1]
    for x in xyrange:
        for y in xyrange:
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
    search_range = int(math.log2(min(g_basic.shape) / 8))
    print(f"\nSearch Range: ±{2 ** search_range - 1}")
    for image in images:
        g_image = BGR2GRAY(image)
        offset = find_offset(g_basic, g_image, search_range)
        print(f"Offset = {offset}")
        sft_img = shift(image, offset)
        sft_imgs.append(sft_img)
    return sft_imgs


# if __name__ == "__main__":
#     img = cv.imread("../../exposures/img05.jpg")
#     img_sh = shift(img, (17, -9))
#     print(find_offset(img, img_sh, 6))
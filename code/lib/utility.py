import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from typing import Any


def shift(img: NDArray[Any], offset: tuple[int, int]) -> NDArray[Any]:
    translation_matrix = np.array(
        [[1, 0, offset[0]], [0, 1, offset[1]]],
        dtype=np.float64
    )
    img_shift = cv.warpAffine(img.astype(np.uint8), translation_matrix, (img.shape[1],img.shape[0]))
    return img_shift


def BGR2GRAY(img: NDArray[np.uint8], BGRratio: list[float] = [19/256, 183/256, 54/256]) -> NDArray[np.uint8]:
    g_img = np.zeros((img.shape[0],img.shape[1]))
    for i in range(3):
        g_img += img[:,:,i] * BGRratio[i]
    g_img = np.clip(g_img , 0 , 255).astype(np.uint8)
    return g_img

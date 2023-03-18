import cv2 as cv
import numpy as np
from utility import *

def bit_and_Xor(img, tolerance = 4):
    threshold = np.median(img)
    bitmap = img < threshold
    mask = (img < (threshold-tolerance)) | (img > (threshold+tolerance))
    return bitmap, mask

def eval(bitmap_s , bitmap_t , mask_s , mask_t):
    t = np.logical_xor(bitmap_s , bitmap_t)
    t = t & mask_s
    t = t & mask_t
    return np.sum(t)

def find_offset(basic, img):

    if((basic.shape[0] <= 2)):
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


def MTB(basic, images):
    sft_imgs = []
    g_basic = BGR2GRAY(basic)
    for image in images:
        g_image = BGR2GRAY(image)
        offset = find_offset(g_basic, g_image)
        sft_img = shift(image, offset)
        sft_imgs.append(sft_img)
    return sft_imgs



imgs = []
for i in range(1,14):
    img = cv.imread("..\..\data\exposures\exposures\img{:02d}.jpg".format(i))
    imgs.append(img)

sft_imgs = MTB(imgs[0], imgs[1:14])

# cv.imshow('Image', sft_imgs[0])
# cv.waitKey()
for i in range(0,12):
    cv.imwrite("..\..\data\output\img{:02d}.jpg".format(i+2), sft_imgs[i])
import cv2 as cv
import numpy as np


def shift(img, offset):
    translation_matrix = np.float32([ [1,0,offset[0]], [0,1,offset[1]] ])   
    img_shift = cv.warpAffine(img.astype(np.uint8), translation_matrix, (img.shape[1],img.shape[0]))
    return img_shift

def BGR2GRAY(img, BGRratio = [19/256, 183/256, 54/256]):
    g_img = np.zeros((img.shape[0],img.shape[1]))
    for i in range(3):
        g_img += img[:,:,i] * BGRratio[i]
    g_img = np.clip(g_img , 0 , 255).astype(np.uint8)
    return g_img

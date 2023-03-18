import cv2 as cv
import numpy as np

def tonemapping(hdr, alpha = 0.36 , bgr_ratio = [0.23, 0.67, 0.10]):
    
    L_w = 0.0
    for i in range(3):
        L_w += bgr_ratio[i] * hdr[:,:,i] 

    zelta = 1e-8

    avg_L_w = np.exp(np.mean(np.log(zelta+L_w)))

    L = alpha*L_w/avg_L_w

    L_white = np.max(L_w)

    L_d = L * (1 + L / (L_white**2)) / (1 + L)

    ldr = np.zeros(hdr.shape)

    for i in range(3):
        ldr[ : , : , i] = hdr[ : , : , i] / L_w * L_d

    ldr = np.clip(ldr*255 , 0 , 255).astype(np.uint8)
    cv.imshow('Image', ldr)
    cv.waitKey()
    return ldr


img = cv.imread("./result.hdr")

ldr = tonemapping(img)
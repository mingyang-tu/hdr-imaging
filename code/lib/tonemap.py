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
    # cv.imshow('Image', ldr)
    # cv.waitKey()
    return ldr

def local_tonemapping(hdr, alpha = 0.18, bgr_ratio = [0.23, 0.67, 0.10], phi = 8, eps = 0.5):
    L_w = 0.0
    for i in range(3):
        L_w += bgr_ratio[i] * hdr[:,:,i] 

    zelta = 1e-8

    avg_L_w = np.exp(np.mean(np.log(zelta+L_w)))

    L = alpha*L_w/avg_L_w

    finish = np.zeros(L.shape)
    L_d = np.zeros(L.shape)
    L_blur_last = cv.GaussianBlur(L , (5 , 5) , 1)

    for i in range(1, 11):
        var = 1.6**i
        L_blur_now = cv.GaussianBlur(L , (5 , 5) , var)
        V = (L_blur_last - L_blur_now) / (2**phi * alpha / (var/1.6)**2 + L_blur_last)

        idx = (np.absolute(V) < eps) & np.logical_not(finish)
        L_d[idx] = (L / (1 + L_blur_last))[idx]
        finish[idx] = True

        L_blur_last = L_blur_now

    idx = np.logical_not(finish)
    L_d[idx] = (L / (1 + L_blur_last))[idx]

    ldr = np.zeros(hdr.shape)

    for i in range(3):
        ldr[ : , : , i] = hdr[ : , : , i] / L_w * L_d

    ldr = np.clip(ldr*255 , 0 , 255).astype(np.uint8)
    return ldr

# hdr = cv.imread("./result.hdr")
# cv.imshow('Image', hdr)
# cv.waitKey()

# ldr = local_tonemapping(hdr)
# cv.imshow('Image', ldr)
# cv.waitKey()
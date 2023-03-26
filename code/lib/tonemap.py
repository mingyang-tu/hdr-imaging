import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from argparse import Namespace
import math


def reinhard(hdr: NDArray[np.float64], algorithm: str, args: Namespace) -> NDArray[np.uint8]:
    print(f"alpha = {args.tmo_alpha}, gamma = {args.tmo_gamma}")

    bgr_ratio = [0.06, 0.67, 0.27]
    ROW, COL, _ = hdr.shape

    L_w = np.zeros((ROW, COL), dtype=np.float64)
    for i in range(3):
        L_w += bgr_ratio[i] * hdr[:,:,i] 

    zelta = 1e-8

    avg_L_w = np.exp(np.mean(np.log(zelta+L_w)))

    L = args.tmo_alpha * L_w / avg_L_w

    L_white = np.max(L_w)

    if algorithm == "global":
        L_d = global_operator(L, L_white)
    elif algorithm == "local":
        L_d = local_operator(L, args.tmo_alpha, args.tmo_phi, args.tmo_eps)
    else:
        raise ValueError("Algorithm not supported.")

    ldr = np.zeros((ROW, COL, 3), dtype=np.float64)

    for i in range(3):
        ldr[:, :, i] = (hdr[:, :, i] * (L_d / (L_w + 1e-8))) ** (args.tmo_gamma)

    ldr = np.clip(ldr*255 , 0 , 255).astype(np.uint8)
    return ldr


def global_operator(L: NDArray[np.float64], L_white: np.float64) -> NDArray[np.float64]:
    L_d = L * (1 + L / (L_white**2)) / (1 + L)
    return L_d


def local_operator(L: NDArray[np.float64], alpha: float, phi: float = 8, eps: float = 0.05):
    print(f"phi = {phi}, eps = {eps}")

    finish = np.zeros(L.shape, dtype=np.float64)
    L_d = np.zeros(L.shape, dtype=np.float64)
    sigma = 0.5
    radius, patch_size = cal_rps(sigma)
    L_blur_last = cv.GaussianBlur(L, (patch_size, patch_size), sigma)

    for _ in range(8):
        sigma *= 1.6
        radius, patch_size = cal_rps(sigma)

        L_blur_now = cv.GaussianBlur(L, (patch_size, patch_size), sigma)
        V = (L_blur_last - L_blur_now) / (2**phi * alpha / radius**2 + L_blur_last)

        idx = (np.absolute(V) > eps) & np.logical_not(finish)
        L_d[idx] = (L / (1 + L_blur_last))[idx]
        finish[idx] = True

        L_blur_last = L_blur_now

    idx = np.logical_not(finish)
    L_d[idx] = (L / (1 + L_blur_last))[idx]

    return L_d


def cal_rps(sigma: float) -> tuple[int, int]:
    radius = math.ceil(sigma * 2)
    patch_size = 2 * radius + 1
    return radius, patch_size

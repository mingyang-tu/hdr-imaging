import numpy as np
from numpy.typing import NDArray
from typing import Any


def solve_g(Z_samp: NDArray[np.uint8], weight: NDArray[Any], ln_delta_t: NDArray[Any], lamb: float) -> NDArray[np.float32]:
    weight, ln_delta_t = weight.astype(np.float32), ln_delta_t.astype(np.float32)

    N_SAMP, P_IMGS = Z_samp.shape
    Z_MAX = 256

    A_mat = np.zeros((N_SAMP * P_IMGS + Z_MAX + 1, N_SAMP + Z_MAX), dtype=np.float32)
    b_vec = np.zeros((N_SAMP * P_IMGS + Z_MAX + 1, 1), dtype=np.float32)

    curr = 0
    for i in range(N_SAMP):
        for j in range(P_IMGS):
            Zij = Z_samp[i, j]
            wij = weight[Zij]
            A_mat[curr, Zij] = wij
            A_mat[curr, i + Z_MAX] = -wij
            b_vec[curr, 0] = wij * ln_delta_t[j]
            curr += 1

    A_mat[curr, 128] = 1
    curr += 1

    for z in range(1, Z_MAX - 1):
        wz = weight[z]
        A_mat[curr, z-1] = lamb * wz
        A_mat[curr, z] = -lamb * wz * 2
        A_mat[curr, z+1] = lamb * wz
        curr += 1

    x_vec = np.dot(np.linalg.pinv(A_mat), b_vec).astype(np.float32)

    return x_vec[:Z_MAX]


def debevec(images: list[NDArray[np.uint8]], delta_t: list[float], lamb: float) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    P_IMGS = len(images)
    N_SAMP = max(int(255 / (P_IMGS - 1)) + 1, 50)

    ROW, COL = images[0].shape

    row_idx = np.random.randint(10, ROW-10, N_SAMP)
    col_idx = np.random.randint(10, COL-10, N_SAMP)

    Z_samp = np.zeros((N_SAMP, P_IMGS), dtype=np.uint8)

    for i in range(P_IMGS):
        Z_samp[:, i] = images[i][row_idx, col_idx]

    weight = np.concatenate([np.arange(0, 128), np.arange(127, -1, -1)], dtype=np.float32)
    ln_delta_t = np.log2(np.array(delta_t, dtype=np.float32))

    g_transform = solve_g(Z_samp, weight, ln_delta_t, lamb).reshape(-1)

    image_stack = np.array(images)
    w_stack = weight[image_stack]
    g_image_stack = g_transform[image_stack]

    for i in range(P_IMGS):
        g_image_stack[i, :, :] -= ln_delta_t[i]

    ln_E = np.sum(w_stack * g_image_stack, axis=0) / (np.sum(w_stack, axis=0) + 1e-9)

    return np.exp2(ln_E), g_transform

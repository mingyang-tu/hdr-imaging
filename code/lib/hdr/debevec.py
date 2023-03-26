import numpy as np
from numpy.typing import NDArray


class Debevec:
    def __init__(self, images: list[NDArray[np.uint8]], delta_t: list[float], lamb: float) -> None:
        print(f"lamb = {lamb}")

        self.P_IMGS = len(images)
        self.SHAPE = images[0].shape
        self.lamb = lamb

        self.N_SAMP = max(int(255 / (self.P_IMGS - 1)) + 1, 50)
        self.row_idx = np.random.randint(10, self.SHAPE[0]-10, self.N_SAMP)
        self.col_idx = np.random.randint(10, self.SHAPE[1]-10, self.N_SAMP)

        self.image_stacks = [
            np.array([im[:, :, i] for im in images], dtype=np.uint8)
            for i in range(3)
        ]
        self.weight = np.concatenate([np.arange(0, 128), np.arange(127, -1, -1)], dtype=np.float64)
        self.ln_delta_t = np.log2(np.array(delta_t, dtype=np.float64))

    def fit(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        hdrs = np.zeros(self.SHAPE, dtype=np.float64)
        gs = np.zeros((256, 3), dtype=np.float64)
        for i in range(3):
            hdr, g = self.fit_channel(i)
            hdrs[:, :, i] = hdr
            gs[:, i] = g
        return hdrs, gs

    def fit_channel(self, channel: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        Z_samp = np.zeros((self.N_SAMP, self.P_IMGS), dtype=np.uint8)

        for i in range(self.P_IMGS):
            Z_samp[:, i] = self.image_stacks[channel][i, self.row_idx, self.col_idx]

        g_transform = self.solve_g(Z_samp).reshape(-1)

        w_stack = self.weight[self.image_stacks[channel]]
        g_image_stack = g_transform[self.image_stacks[channel]]

        for i in range(self.P_IMGS):
            g_image_stack[i, :, :] -= self.ln_delta_t[i]

        ln_E = np.sum(w_stack * g_image_stack, axis=0) / (np.sum(w_stack, axis=0) + 1e-8)

        return np.exp2(ln_E), g_transform

    def solve_g(self, Z_samp: NDArray[np.uint8]) -> NDArray[np.float64]:
        Z_MAX = 256

        A_mat = np.zeros((self.N_SAMP * self.P_IMGS + Z_MAX + 1, self.N_SAMP + Z_MAX), dtype=np.float64)
        b_vec = np.zeros((self.N_SAMP * self.P_IMGS + Z_MAX + 1, 1), dtype=np.float64)

        curr = 0
        for i in range(self.N_SAMP):
            for j in range(self.P_IMGS):
                Zij = Z_samp[i, j]
                wij = self.weight[Zij]
                A_mat[curr, Zij] = wij
                A_mat[curr, i + Z_MAX] = -wij
                b_vec[curr, 0] = wij * self.ln_delta_t[j]
                curr += 1

        A_mat[curr, 128] = 1
        curr += 1

        for z in range(1, Z_MAX - 1):
            wz = self.weight[z]
            A_mat[curr, z-1] = self.lamb * wz
            A_mat[curr, z] = -self.lamb * wz * 2
            A_mat[curr, z+1] = self.lamb * wz
            curr += 1

        x_vec = np.dot(np.linalg.pinv(A_mat), b_vec).astype(np.float64)

        return x_vec[:Z_MAX]

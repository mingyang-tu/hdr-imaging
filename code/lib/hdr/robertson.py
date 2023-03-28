import numpy as np
from numpy.typing import NDArray


class Robertson:
    def __init__(self, images: list[NDArray[np.uint8]], delta_t: list[float], threshold: float) -> None:
        print(f"threshold = {threshold}")

        self.P_IMGS = len(images)
        self.SHAPE = images[0].shape
        self.threshold = threshold

        self.total_pixels = self.SHAPE[0] * self.SHAPE[1] * self.P_IMGS

        self.weight = np.exp(-4 * np.square((np.arange(0, 256, dtype=np.float64) - 127.5) / 127.5))
        self.weight -= np.min(self.weight)
        self.weight /= np.max(self.weight)

        self.image_stacks = [
            np.array([im[:, :, i] for im in images], dtype=np.uint8)
            for i in range(3)
        ]
        self.delta_t = np.array(delta_t, dtype=np.float64)

        self.histogram = []
        for i in range(3):
            values, counts = np.unique(self.image_stacks[i], return_counts=True)
            self.histogram.append({v: c for v, c in zip(values, counts)})

    def fit(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        COLOR = "BGR"
        hdrs = np.zeros(self.SHAPE, dtype=np.float64)
        gs = np.zeros((256, 3), dtype=np.float64)
        for i in range(3):
            print(f"\nChannel {COLOR[i]}")
            hdr, g = self.fit_channel(i)
            hdrs[:, :, i] = hdr
            gs[:, i] = g
        return hdrs, gs

    def fit_channel(self, channel: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        g_transform = np.arange(0, 256, dtype=np.float64) / 128.
        last_sum = np.inf

        for i in range(1, 31):
            energy = self.optimize_E(channel, g_transform)
            g_transform = self.optimize_g(channel, energy)

            curr_sum = self.cal_objective(channel, g_transform, energy)
            diff = last_sum - curr_sum
            print(f"Iteration: {i:2d}, curr_sum: {curr_sum:.9f}, diff: {diff:.9f}")
            if diff < self.threshold:
                break
            last_sum = curr_sum

        return self.optimize_E(channel, g_transform), g_transform

    def optimize_E(self, channel: int, g_transform: NDArray[np.float64]) -> NDArray[np.float64]:
        w_stack = self.weight[self.image_stacks[channel]]
        for i in range(self.P_IMGS):
            w_stack[i, :, :] *= self.delta_t[i]

        w_stack_t2 = np.copy(w_stack)
        for i in range(self.P_IMGS):
            w_stack_t2[i, :, :] *= self.delta_t[i]

        g_image_stack = g_transform[self.image_stacks[channel]]

        return np.sum(w_stack * g_image_stack, axis=0) / (np.sum(w_stack_t2, axis=0) + 1e-8)

    def optimize_g(self, channel: int, energy: NDArray[np.float64]) -> NDArray[np.float64]:
        g_transform = np.zeros(256, dtype=np.float64)
        for i in range(256):
            for j in range(self.P_IMGS):
                idxs = self.image_stacks[channel][j, :, :] == i
                g_transform[i] += np.sum(energy[idxs]) * self.delta_t[j]
        for i in range(256):
            if i in self.histogram[channel]:
                g_transform[i] /= self.histogram[channel][i]
        g_transform /= g_transform[128]
        return g_transform

    def cal_objective(self, channel: int, g_transform: NDArray[np.float64], energy: NDArray[np.float64]) -> float:
        w_stack = self.weight[self.image_stacks[channel]]
        g_image_stack = g_transform[self.image_stacks[channel]]
        total_sum = 0.
        for i in range(self.P_IMGS):
            total_sum += np.sum(w_stack[i, :, :] * np.square(g_image_stack[i, :, :] - energy * self.delta_t[i]))
        return total_sum / self.total_pixels

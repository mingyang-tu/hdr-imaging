from argparse import Namespace
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .debevec import Debevec


def get_hdr(images: list[NDArray[np.uint8]], delta_t: list[float], algorithm: str, args: Namespace) -> NDArray[np.float32]:
    ROW, COL, _ = images[0].shape

    assert len(images) > 1 and len(images) == len(delta_t)
    assert all([(i.shape[0] == ROW) and (i.shape[1] == COL) for i in images])

    hdrs, g_trans = np.zeros((ROW, COL, 3), dtype=np.float32), np.zeros((256, 3), dtype=np.float32)

    if algorithm == "debevec":
        deb = Debevec(images, delta_t, getattr(args, "lamb"))
        hdrs, g_trans = deb.fit()
    else:
        raise ValueError("Algorithm not supported.")

    COLOR = "bgr"
    if getattr(args, "verbose", False):
        _, axes1 = plt.subplots(2, 2)
        for i in range(3):
            axes1[i // 2][i % 2].plot(g_trans[:, i], range(0, 256), 'x', c=COLOR[i], markersize=3)
            axes1[1][1].plot(g_trans[:, i], range(0, 256), 'x', c=COLOR[i], markersize=3)

        fig2, axes2 = plt.subplots()
        mat = axes2.matshow(
            np.mean(hdrs, axis=2),
            cmap="jet", norm=LogNorm()
        )
        fig2.colorbar(mat)
        plt.show()

    return hdrs

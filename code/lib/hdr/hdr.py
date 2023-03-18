from argparse import Namespace
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .debevec import debevec


def hdr_bgr(images: list[NDArray[np.uint8]], delta_t: list[float], algorithm: str, args: Namespace) -> NDArray[np.float32]:
    ROW, COL, _ = images[0].shape

    assert len(images) > 1 and len(images) == len(delta_t)
    assert all([(i.shape[0] == ROW) and (i.shape[1] == COL) for i in images])

    hdrs, g_trans = [], []

    if algorithm == "debevec":
        for i in range(3):
            hdr, g = debevec(
                [im[:, :, i] for im in images],
                delta_t,
                getattr(args, "lamb", 30.)
            )
            hdrs.append(hdr)
            g_trans.append(g)
    else:
        raise ValueError("Algorithm not supported.")

    COLOR = "bgr"
    if getattr(args, "verbose", False):
        _, axes1 = plt.subplots(2, 2)
        for i in range(3):
            axes1[i // 2][i % 2].plot(g_trans[i], range(0, 256), 'x', c=COLOR[i], markersize=3)
            axes1[1][1].plot(g_trans[i], range(0, 256), 'x', c=COLOR[i], markersize=3)

        fig2, axes2 = plt.subplots()
        mat = axes2.matshow(
            (19 * hdrs[0] + 183 * hdrs[1] + 54 * hdrs[2]) / 256,
            cmap="jet", norm=LogNorm()
        )
        fig2.colorbar(mat)
        plt.show()

    return np.array(hdrs).transpose(1, 2, 0)

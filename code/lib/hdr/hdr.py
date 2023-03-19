from argparse import Namespace
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .debevec import Debevec
from .robertson import Robertson


def get_hdr(images: list[NDArray[np.uint8]], delta_t: list[float], algorithm: str, args: Namespace) -> NDArray[np.float32]:
    ROW, COL, _ = images[0].shape

    assert len(images) > 1 and len(images) == len(delta_t)
    assert all([(i.shape[0] == ROW) and (i.shape[1] == COL) for i in images])

    if algorithm == "debevec":
        deb = Debevec(images, delta_t, args.lamb)
        hdrs, g_trans = deb.fit()
    elif algorithm == "robertson":
        rob = Robertson(images, delta_t, args.threshold)
        hdrs, g_trans = rob.fit()
    else:
        raise ValueError("Algorithm not supported.")

    COLOR = "bgr"
    if args.verbose:
        _, axes1 = plt.subplots(2, 2)
        for i in range(3):
            axes1[i // 2][i % 2].plot(g_trans[:, i], range(0, 256), 'x', c=COLOR[i], markersize=3)
            axes1[1][1].plot(g_trans[:, i], range(0, 256), 'x', c=COLOR[i], markersize=3)
        if algorithm == "robertson":
            for i in range(2):
                for j in range(2):
                    axes1[i][j].set_xscale("log", nonpositive='clip', base=2)

        fig2, axes2 = plt.subplots()
        mat = axes2.matshow(
            np.mean(hdrs, axis=2),
            cmap="jet", norm=LogNorm()
        )
        fig2.colorbar(mat)
        plt.show()

    return hdrs

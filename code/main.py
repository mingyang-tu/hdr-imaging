from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.debevec import debevec


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--images_dir",
        type=Path
    )
    parser.add_argument(
        "-s",
        "--shutter_speed_file",
        type=Path
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path, 
        default=""
    )
    parser.add_argument(
        "-l",
        "--lamb",
        type=float,
        default=30.
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    root = args.images_dir
    ss_file = args.shutter_speed_file

    images, delta_t = [], []
    with open(root / ss_file) as f:
        lines = f.readlines()
        for line in lines[3:]:
            temp = line.split()
            images.append(cv2.imread(str(root / temp[0])))
            delta_t.append(1 / float(temp[1]))

    color = ['b', 'g', 'r']
    hdrs = []

    plt.figure(f"lambda = {args.lamb}")
    for i in range(3):
        hdr, g = debevec(
            [im[:, :, i] for im in images],
            delta_t,
            args.lamb
        )
        hdrs.append(hdr)
        plt.plot(g, range(0, 256), 'x', c=color[i], markersize=3)

    hdrs_np = np.exp2(np.array(hdrs).transpose(1, 2, 0))

    if args.output_dir:
        cv2.imwrite(str(args.output_dir / "result.hdr"), hdrs_np)

    plt.figure("HDR image")
    plt.imshow(
        (19 * hdrs[0] + 183 * hdrs[1] + 54 * hdrs[2]) / 256,
        cmap="jet"
    )
    plt.colorbar()
    plt.show()

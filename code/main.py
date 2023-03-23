from argparse import ArgumentParser, Namespace
from pathlib import Path
from numpy.typing import NDArray
import numpy as np
import cv2

from lib import *


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-i", "--images_dir", type=Path, required=True)
    parser.add_argument("-s", "--shutter_speed_file", type=Path, required=True)
    parser.add_argument("-o", "--output_dir", type=Path, default=None)
    parser.add_argument("--mtb", action="store_true")
    parser.add_argument("--show_hdr", action="store_true")
    parser.add_argument("--hdr_alg", type=str, default="robertson")
    parser.add_argument("--hdr_lamb", type=float, default=30.)
    parser.add_argument("--hdr_threshold", type=float, default=1e-6)
    parser.add_argument("--tmo_alg", type=str, default="local")
    parser.add_argument("--tmo_alpha", type=float, default=0.18)
    parser.add_argument("--tmo_phi", type=float, default=8)
    parser.add_argument("--tmo_eps", type=float, default=0.5)
    args = parser.parse_args()
    return args


def read_files(args: Namespace) -> tuple[list[NDArray[np.uint8]], list[float]]:
    root = args.images_dir

    images, delta_t = [], []
    with open(root / args.shutter_speed_file) as f:
        lines = f.readlines()
        for line in lines[3:]:
            temp = line.split()
            images.append(cv2.imread(str(root / temp[0])))
            delta_t.append(1 / float(temp[1]))

    print(f"Exposure:\n{delta_t}")

    # s = images[0].shape
    # for i in range(len(images)):
    #     images[i] = cv2.resize(images[i], (s[1] // 4, s[0] // 4))

    return images, delta_t


if __name__ == "__main__":
    args = parse_args()

    images, delta_t = read_files(args)

    sft_imgs = [images[0]]
    if args.mtb:
        print("\nStart MTB...")
        sft_imgs.extend(mtb(images[0], images[1:]))
    else:
        sft_imgs.extend(images[1:])

    print("\nStart HDR...")
    hdr_image = get_hdr(sft_imgs, delta_t, args.hdr_alg, args)

    print("\nStart tonemapping...")
    output = reinhard(hdr_image, args.tmo_alg, args)

    if args.output_dir:
        cv2.imwrite(str(args.output_dir / "result.jpg"), output)
        cv2.imwrite(str(args.output_dir / "result.hdr"), hdr_image)

    cv2.imshow('Tonemapping Result', output)
    cv2.waitKey()

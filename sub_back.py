import os
import sys
from functools import partial
from glob import glob

import numpy as np
from tifffile import imread, imwrite
from tqdm import tqdm


def correct(
    frame: np.ndarray, flatfield: np.ndarray, darkframe: np.ndarray
) -> np.ndarray:
    return (
        (frame - darkframe) / (flatfield - darkframe) * np.mean(flatfield - darkframe)
    )


def sub_back(
    tif_path: str,
    out_dir: str,
    flatfield_path: str,
    darkframe_path: str,
    window: int,
    num: int | None = None,
) -> None:
    tif_files = sorted(glob(tif_path))
    half_window = window // 2
    os.makedirs(out_dir, exist_ok=True)
    frame_sum = np.zeros(shape=(1022, 1024), dtype=np.float32)

    preprocess = partial(
        lambda path, flatfield, darkframe: correct(
            imread(path).astype(np.float64), flatfield, darkframe
        ),
        flatfield=imread(flatfield_path).astype(np.float64),
        darkframe=imread(darkframe_path).astype(np.float64),
    )

    # initial sum
    for i in tqdm(range(window), desc="Initialization"):
        frame = preprocess(tif_files[i])
        frame_sum += frame.astype(np.float32)

    count = 0
    ids = range(half_window + 1, len(tif_files) - half_window)
    for i in tqdm(ids, desc="Processing", total=num if num else len(ids)):
        if num and count >= num:
            break

        frame_remove = preprocess(tif_files[i - half_window - 1])
        frame_sum -= frame_remove.astype(np.float32)

        frame_add = preprocess(tif_files[i + half_window])
        frame_sum += frame_add.astype(np.float32)

        background = frame_sum / window
        current_frame = preprocess(tif_files[i]).astype(np.float32)
        frame_noback = np.abs(current_frame - background)

        frame_noback = np.clip(
            frame_noback,
            np.percentile(frame_noback, 98),
            np.percentile(frame_noback, 99.99),
        )
        frame_noback = (
            (frame_noback - frame_noback.min())
            / (frame_noback.max() - frame_noback.min())
            * 255
        )

        out_path = os.path.join(out_dir, os.path.basename(tif_files[i]))
        imwrite(out_path, frame_noback.astype(np.uint8))

        count += 1


if __name__ == "__main__":
    basename = sys.argv[1]

    tif_path = "/media/yihang-geng/ExpDataGYH/BF_PHC/" + basename + "_raw/t*.tif"
    out_dir = "/media/yihang-geng/ExpDataGYH/BF_PHC/" + basename + "_noback"
    flatfield_path = (
        "/media/yihang-geng/ExpDataGYH/BF_PHC/" + basename + "_flatfield.tif"
    )
    darkframe_path = "/media/yihang-geng/ExpDataGYH/BF_PHC/dark_frame.tif"

    sub_back(tif_path, out_dir, flatfield_path, darkframe_path, window=401, num=None)

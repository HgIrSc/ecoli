import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims
import trackpy as tp


def traj_extract(
    tif_files: str,
    out_path: str,
    locating_params: dict,
    linking_params: dict,
    nframes: int | None = None,
) -> None:
    tifs = pims.ImageSequence(tif_files)

    pixel_size = 0.66
    fps = 20

    tp.quiet()
    print("Locating ...")
    f = tp.batch(tifs[:nframes], **locating_params)
    print("Locating finished")

    print("Linking ...")
    f = tp.predict.NearestVelocityPredict().link_df(f, **linking_params)
    print("Linking finished")

    print("Calculating velocity ...")
    diff_traj = f.groupby("particle").diff()
    f["vx"] = diff_traj.x / diff_traj.frame * pixel_size * fps
    f["vy"] = diff_traj.y / diff_traj.frame * pixel_size * fps
    f["v"] = (f["vx"].pow(2) + f["vy"].pow(2)).pow(0.5)
    f.to_parquet(out_path, compression="gzip")
    print("Calculating velocity finished")
    print(f"Trajectories saved to {out_path}")
    print(f"{f.particle.nunique()} trajectories were found")


def check_quality(tif_files: str, traj_file: str) -> None:
    traj = pd.read_parquet(traj_file)
    tifs = pims.ImageSequence(tif_files)

    frame = np.random.choice(traj.frame.unique())

    tp.annotate(traj[traj.frame == frame], tifs[frame])
    plt.show()
    tp.subpx_bias(traj)
    plt.show()


if __name__ == "__main__":
    basename = sys.argv[1]

    tif_files = "/media/yihang-geng/ExpDataGYH/Ecoli/" + basename + "_noback/t*.tif"
    out_path = "trajs/" + basename + ".parquet.gzip"

    # Locating parameters
    locating_params = {
        "diameter": 13,
        # "threshold": 2000,
        "engine": "numba",
        "processes": 16,
    }

    # Linking parameters
    linking_params = {"search_range": 3, "memory": 3, "link_strategy": "numba"}

    traj_extract(tif_files, out_path, locating_params, linking_params)

    check_quality(tif_files, out_path)

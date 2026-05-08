import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import trackpy as tp
import os
import sys
from skimage import exposure, morphology, measure
from tifffile import imread, imwrite
from glob import glob
from tqdm import tqdm
from collections import deque
from typing import List, Tuple, Dict


def load_config(config_path) -> Dict:
    with open(config_path, "r") as f:
        return json.load(f)


class Preprocessor:
    def __init__(self, frame_seq: List[str], config: Dict):
        self.config = config
        frame0 = self.crop(self.read_frame(frame_seq[0]))
        self.frame_sum = np.zeros_like(frame0, dtype=np.float32)
        self.frame_deque = deque()
        for frame_path in frame_seq[: self.config["average_frame_length"] + 1]:
            frame = self.crop(self.read_frame(frame_path))
            self.frame_sum += frame
            self.frame_deque.append(frame)

    def read_frame(self, path: str) -> np.ndarray:
        origin = imread(path).astype(np.float32)
        if self.config["frame_subsampling"]:
            return (
                (
                    origin[
                        :: self.config["frame_subsampling"],
                        :: self.config["frame_subsampling"],
                    ]
                )
                / 2
            )
        else:
            return origin

    def crop(self, frame: np.ndarray) -> np.ndarray:
        if self.config["frame_cropping"]:
            hrange = self.config["frame_height_range"]
            cropped = frame[hrange[0] : hrange[1], :]
            return cropped
        else:
            return frame

    def normalize(self, frame: np.ndarray, type: str) -> np.ndarray:
        frame_min, frame_max = np.min(frame), np.max(frame)
        if type == "uint8":
            result = (frame - frame_min) / (frame_max - frame_min) * 255.0
            return result.astype(np.uint8)
        elif type == "float":
            result = (frame - frame_min) / (frame_max - frame_min)
            return result
        else:
            raise RuntimeError(f"{type} is not a normalization type")

    def max_entropy_threshold(self, image: np.ndarray):
        # 1. Get the histogram and normalize it to get probabilities
        hist, bin_edges = np.histogram(image, bins=256, range=(0, 255))
        prob = hist / float(np.sum(hist))

        # 2. Pre-calculate cumulative sums (P1) and cumulative entropy (H1)
        # Adding a tiny epsilon to avoid log(0)
        epsilon = 1e-10

        # Cumulative probability of the background
        P1 = np.cumsum(prob)
        # Cumulative probability of the foreground
        P2 = 1.0 - P1

        # Element-wise entropy: -p * log(p)
        entropy_elements = -prob * np.log(prob + epsilon)

        # Cumulative entropy
        H_cum = np.cumsum(entropy_elements)
        H_total = H_cum[-1]

        # 3. Calculate Entropy for all possible thresholds (t)
        # Background Entropy: H_b = H_cum / P1 + log(P1)
        # Foreground Entropy: H_f = (H_total - H_cum) / P2 + log(P2)

        # We use 'where' to avoid division by zero at the histogram edges
        H_b = np.zeros(256)
        H_f = np.zeros(256)
        valid_idx = (P1 > 0) & (P2 > 0)
        H_b[valid_idx] = H_cum[valid_idx] / P1[valid_idx] + np.log(P1[valid_idx])
        H_f[valid_idx] = (H_total - H_cum[valid_idx]) / P2[valid_idx] + np.log(
            P2[valid_idx]
        )

        # Total entropy to maximize
        H_total_sum = H_b + H_f

        # 4. Find the index (threshold) that yields the maximum entropy
        optimal_threshold = np.argmax(H_total_sum)

        return optimal_threshold

    def run(
        self, return_intermediates: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        background = self.frame_sum / self.config["average_frame_length"]

        # Remove background, denoise then do histogram enhancement
        noback_denoised = morphology.erosion(
            self.normalize(
                self.frame_deque[self.config["average_frame_length"] // 2] / background,
                "uint8",
            ),
            morphology.disk(self.config["denoise_erosion_radius"]),
        )
        rescale_thresh = np.percentile(noback_denoised, q=1)
        noback_enhanced = self.normalize(
            exposure.rescale_intensity(
                noback_denoised,
                in_range=(0, rescale_thresh),
            ),
            "uint8",
        )

        # Maximum entropy threshold segmentation
        bin_frame = noback_enhanced < self.max_entropy_threshold(noback_enhanced)

        result_frame = morphology.closing(
            morphology.opening(
                bin_frame, morphology.disk(self.config["denoise_openclose_radius"])
            ),
            morphology.disk(self.config["denoise_openclose_radius"]),
        )

        if return_intermediates:
            return (
                self.frame_deque[self.config["average_frame_length"] // 2],
                noback_enhanced,
                bin_frame,
                result_frame,
            )
        else:
            return result_frame

    def update(self, frame_path: str) -> None:
        new_frame = self.crop(self.read_frame(frame_path))
        self.frame_sum += new_frame
        self.frame_sum -= self.frame_deque.popleft()
        self.frame_deque.append(new_frame)


def test_preprocess(preprocessor: Preprocessor) -> None:
    """
    We use this function to test the preprocess methods.
    """
    (frame_origin, frame_noback, frame_bin, frame_final) = preprocessor.run(True)

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    fig.set_layout_engine("constrained")
    ax[0, 0].imshow(frame_origin, cmap="gray")
    ax[0, 0].set_xlabel("x [\u03bcm]")
    ax[0, 0].set_ylabel("y [\u03bcm]")
    ax[0, 0].set_title("Origin")
    ax[0, 1].imshow(frame_noback, cmap="gray")
    ax[0, 1].set_xlabel("x [\u03bcm]")
    ax[0, 1].set_ylabel("y [\u03bcm]")
    ax[0, 1].set_title("Background subtracted and enhanced")
    ax[1, 0].imshow(frame_bin, cmap="gray")
    ax[1, 0].set_xlabel("x [\u03bcm]")
    ax[1, 0].set_ylabel("y [\u03bcm]")
    ax[1, 0].set_title("Binary")
    ax[1, 1].imshow(frame_final, cmap="gray")
    ax[1, 1].set_xlabel("x [\u03bcm]")
    ax[1, 1].set_ylabel("y [\u03bcm]")
    ax[1, 1].set_title("Final")
    plt.show()


def process_all(save_bin_image: bool = False) -> None:
    config = load_config(sys.argv[1])

    output_dir = os.path.join(
        config["frame_dir"],
        f"../{os.path.basename(config['frame_dir'])}_noback",
    )
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    frame_glob = os.path.join(config["frame_dir"], "t*.tif")
    frame_seq = sorted(glob(frame_glob))[config["frame_start"] : config["frame_end"]]

    preprocessor = Preprocessor(frame_seq, config)

    origin_traj = []
    for id, frame_path in enumerate(
        tqdm(frame_seq[config["average_frame_length"] + 1 :], desc="iter frames")
    ):
        result_frame = preprocessor.run()

        if save_bin_image:
            imwrite(
                os.path.join(output_dir, f"t{id:06d}.tif"),
                np.asarray(result_frame * 255, dtype=np.uint8),
            )

        # Label and measure connected components
        label_frame = measure.label(result_frame)
        regions = pd.DataFrame(
            measure.regionprops_table(
                label_frame, properties=["label", "area", "centroid", "orientation"]
            )
        )
        regions["frame"] = id * np.ones(len(regions), dtype=np.int32)
        regions.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)

        # Filtering regions
        origin_traj.append(regions[regions["area"].between(*config["label_thresh"])])

        preprocessor.update(frame_path)

    origin_traj = pd.concat(origin_traj)

    tp.quiet()
    origin_traj = tp.predict.NearestVelocityPredict().link_df(
        origin_traj,
        search_range=config["link_search_range"],
        memory=config["link_memory"],
    )
    print(f"Find {len(origin_traj.particle.unique())} particles")
    origin_traj_path = os.path.join(
        config["frame_dir"],
        f"../trajs_morph/{os.path.basename(config['frame_dir'])}.parquet.gzip",
    )
    origin_traj.to_parquet(origin_traj_path, compression="gzip")


if __name__ == "__main__":
    # config = load_config(sys.argv[1])
    # frame_glob = os.path.join(config["frame_dir"], "t*.tif")
    # frame_seq = sorted(glob(frame_glob))[config["frame_start"] : config["frame_end"]]
    # preprocessor = Preprocessor(frame_seq, config)
    # test_preprocess(preprocessor)

    process_all(save_bin_image=False)

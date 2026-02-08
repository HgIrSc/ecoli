import sys
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp
from scipy.stats import linregress


# Calculate MSD of particles
def msd_lagtime(traj: pd.DataFrame, on_coord: str = "x"):
    def _msd_lagtime(traj, on_coord):
        disp = traj[on_coord]
        N = len(disp)
        D = np.append(np.square(disp), 0)
        F = np.fft.fft(disp, n=2 * N)
        S2 = np.fft.ifft(F * F.conjugate())[:N].real / np.arange(1, N + 1)[::-1]
        Q = 2 * D.sum()
        S1 = np.zeros(N)
        for m in range(N):
            Q = Q - D[m - 1] - D[N - m]
            S1[m] = Q / (N - m)
        return S1 - 2 * S2

    if on_coord == "both":
        return _msd_lagtime(traj, "x") + _msd_lagtime(traj, "y")
    else:
        return _msd_lagtime(traj, on_coord)


def plot_msd_all(traj_file, pixel_size=0.33, fps=20):
    # Load filtered trajectories
    traj = tp.filter_stubs(pd.read_parquet(traj_file), 10)

    # Group by particle id
    grouped = traj.groupby("particle")
    print(f"{grouped.ngroups} particles are loaded")

    all_msd = []
    for _, _ptraj in grouped:
        all_msd.append(msd_lagtime(_ptraj, "both"))

    times = np.concatenate([np.arange(len(msd)) for msd in all_msd]) / fps
    msds = np.concatenate(all_msd) * pixel_size**2

    num_bins = 400
    tedges = np.geomspace(times[times > 0].min(), times.max(), num_bins)
    msdedges = np.geomspace(1e-3, msds.max(), num_bins)

    fig, ax = plt.subplots()
    cmap = plt.colormaps["plasma"]
    cmap = cmap.with_extremes(bad=cmap(0))
    hist, _, _ = np.histogram2d(times, msds, bins=(tedges, msdedges))
    pcm = ax.pcolormesh(tedges, msdedges, hist.T, cmap=cmap, norm=mcolors.LogNorm())
    ax.axline(
        (tedges.min(), 1e-2),
        slope=1,
        linestyle="dashed",
        color="white",
        label="slope = 1",
    )
    ax.axline(
        (tedges.min(), 1e-2),
        slope=2,
        linestyle="dashed",
        color="white",
        label="slope = 2",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Lag time [s]")
    ax.set_ylabel("$MSD$ [\u03bcm$^2$]")
    ax.legend(loc="upper left")
    fig.colorbar(pcm, ax=ax, label="# points", extend="min")
    plt.show()


# Fit powerlaw MSD
def powerlaw_msd(traj: pd.DataFrame, max_lagtime: int = 10, on_coord: str = "x"):
    msd = msd_lagtime(traj, on_coord)

    log_lagtime = np.log(np.arange(1, len(msd)))
    log_msd = np.log(msd[1:])

    return linregress(log_lagtime[:max_lagtime], log_msd[:max_lagtime])


def best_powerlaw_msd(traj: pd.DataFrame, min_lagtime: int = 10, on_coord: str = "x"):
    msd = msd_lagtime(traj, on_coord)

    log_lagtime = np.log(np.arange(1, len(msd)))
    log_msd = np.log(msd[1:])

    results = []
    for max_lagtime in range(min_lagtime, len(msd)):
        results.append(linregress(log_lagtime[:max_lagtime], log_msd[:max_lagtime]))

    return results[np.argmin([result.stderr for result in results])]


def plot_meanv_exponent(traj_file, pixel_size=0.33, fps=20):
    # Load filtered trajectories
    traj = tp.filter_stubs(pd.read_parquet(traj_file), 100)
    traj = traj.dropna()

    # Group by particle id
    grouped = traj.groupby("particle")
    print(f"{grouped.ngroups} particles are loaded")

    # Fit best powerlaw MSD for all particles
    all_index = []
    all_meany = []
    all_meanv = []
    for _, _ptraj in grouped:
        all_index.append(powerlaw_msd(_ptraj, 20, "both").slope)
        all_meany.append(_ptraj.y.mean() * pixel_size)
        all_meanv.append(_ptraj.v.mean())

    fig, ax = plt.subplots(
        2,
        2,
        sharex="col",
        gridspec_kw={"height_ratios": [1, 3], "width_ratios": [20, 1]},
    )
    fig.set_size_inches(6, 6)
    fig.set_layout_engine("constrained")
    ax[0, 0].hist(all_index, bins=100, histtype="step")
    ax[0, 0].set_ylabel("# particles")
    ax[0, 1].set_visible(False)
    cmap = plt.colormaps["plasma"]
    cmap = cmap.with_extremes(bad=cmap(0))
    hist, xedges, yedges = np.histogram2d(all_index, all_meanv, bins=100)
    # hist, xedges, yedges = np.histogram2d(all_index, all_meany, bins=100)
    pcm = ax[1, 0].pcolormesh(xedges, yedges, hist.T, cmap=cmap, norm=mcolors.LogNorm())
    ax[1, 0].set_xlabel("Estimated power-law exponent")
    ax[1, 0].set_ylabel("Mean velocity [um s$^{-1}$]")
    # ax[1, 0].set_ylabel('Mean y [\u03bcm]')
    # fig.colorbar(pcm, cax=ax[1, 1], label='# particles')
    fig.colorbar(pcm, cax=ax[1, 1], label="# particles", extend="min")
    plt.show()


def apply_hybrid_filter(traj_file, out_path, exponent_thresh=0, velocity_thresh=0):
    # Load filtered trajectories
    traj = tp.filter_stubs(pd.read_parquet(traj_file), 100)
    traj = traj.dropna()

    # Group by particle id
    grouped = traj.groupby("particle")
    print(f"{grouped.ngroups} particles are loaded")

    # Filter
    filtered_traj = grouped.filter(
        lambda par: (
            powerlaw_msd(par, 20, "both").slope > exponent_thresh
            and par.v.mean() > velocity_thresh
        )
    )
    filtered_traj.to_parquet(out_path)
    print(f"filtered to {filtered_traj.particle.nunique()} particles")


if __name__ == "__main__":
    basename = sys.argv[1]

    in_traj_file = "trajs/" + basename + ".parquet.gzip"
    out_traj_file = "trajs/" + basename + ".filtered.parquet.gzip"

    # plot_meanv_exponent(in_traj_file, pixel_size=0.66, fps=20)
    apply_hybrid_filter(in_traj_file, out_traj_file, exponent_thresh=1.8)

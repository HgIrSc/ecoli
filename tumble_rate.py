import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore


def calc_curvature(traj: pd.DataFrame, pused: int = 3) -> pd.Series:
    """
    Calculate curvature along a trajectory

    Parameters
    ----------
    traj: pd.DataFrame
        trajectory
    pused: int=3
        number of points used to calculate curvature

    Returns
    -------
    pd.Series
        curvature along the trajectory [pixel^{-1}]
    """
    dx = traj.x.diff(pused - 1) / (pused - 1)
    dy = traj.y.diff(pused - 1) / (pused - 1)
    ddx = traj.x.diff(pused - 2).diff(pused - 2) / (pused - 2) ** 2
    ddy = traj.y.diff(pused - 2).diff(pused - 2) / (pused - 2) ** 2

    curv = (dx * ddy - dy * ddx) / (dx.pow(2) + dy.pow(2)).pow(1.5)
    curv.index = traj.frame

    return curv


def locate_tumbles(
    ecoli: pd.DataFrame,
    window_width: int = 7,
    detect_window: int = 1,
    min_distance: int = 40,
    std_threshold: int = 1,
    full: bool = False,
):
    """
    Locate tumbles by mean velocity and curvature

    Parameters
    ----------
    ecoli: pd.DataFrame
        trajectory
    window_width: int=7
        window width used to calculate mean velocity and curvature
    detect_window: int=1
        window width used to detect tumbles
    min_distance: int=40
        min distance between tumbles (frames)
    std_threshold: int=2
        threshold of std(mean velocity) and std(abs(curvature))
    full: bool=False
        if full=True, return normalized mean velocity and normalized curvature

    Returns
    -------
    pd.Series or tuple[pd.Series, pd.Series, pd.Series]
        indices of tumbles. if full is True,
        also return normalized mean velocity and normalized curvature
    """
    # Mean velocity
    v = np.sqrt(
        ecoli.vx.rolling(window_width).mean() ** 2
        + ecoli.vy.rolling(window_width).mean() ** 2
    )
    # Normalized mean velocity
    norm_v = (v - v.mean()) / v.std()

    # Curvature
    curv = calc_curvature(ecoli, window_width)
    # Normalized curvature
    norm_curv = (curv - curv.mean()) / curv.std()

    istumble = (
        (norm_v < -std_threshold)
        .rolling(detect_window)
        .agg(np.logical_or.reduce)
        .astype(np.bool)
    ) & (norm_curv.abs() > std_threshold)

    # Clean up tumbles within min_distance
    _idx = 0
    while _idx < len(istumble):
        if istumble.iloc[_idx] == True:
            istumble.iloc[_idx + 1 : _idx + min_distance + 1] = False
            _idx += min_distance + 1
        else:
            _idx += 1

    if full:
        return ecoli.index[istumble], norm_v, norm_curv
    else:
        return ecoli.index[istumble]


if __name__ == "__main__":
    traj = pd.read_parquet(
        "/media/yihang-geng/DATA1/Data/Escherichia Coli Viscotaxi/250923/msd1.8_v5_filtered_ecoli_traj_5-15min2.parquet.gzip"
    )
    traj = traj.groupby("particle").filter(lambda x: len(x) > 100)
    grouped = traj.groupby("particle")

    # Extract runnings
    runnings = []
    count = 0
    for _, _ptraj in grouped:
        istumble = locate_tumbles(
            _ptraj,
            window_width=7,
            detect_window=10,
            min_distance=20,
            std_threshold=1,
            full=False,
        )
        if len(istumble) < 2:
            continue
        for _running_start, _running_end in zip(istumble[:-1], istumble[1:]):
            one_running = _ptraj.loc[_running_start:_running_end, :].copy()
            one_running["running"] = count
            runnings.append(one_running)
            count += 1

    all_runnnings = pd.concat(runnings, ignore_index=True)
    grouped_runnings = all_runnnings.groupby("running")
    print(f"{count} runnings are found")

    fig, ax = plt.subplots()
    fig.set_layout_engine("constrained")

    fps = 20

    # Running time
    hist, edges, _ = ax.hist(
        (grouped_runnings.frame.max() - grouped_runnings.frame.min()) / fps,
        bins=100,
        histtype="step",
        label="Distribution",
    )
    edges_middle = (edges[:-1] + edges[1:]) / 2
    mask = hist > 1
    coeff = np.polyfit(edges_middle[mask], np.log(hist[mask] + 1e-6), 1)
    print(coeff)
    ax.plot(
        edges_middle[mask],
        np.exp(np.polyval(coeff, edges_middle[mask])),
        "-",
        label="Exponential fit",
    )
    ax.text(
        7,
        5,
        "$\\langle t_{running}\\rangle=%.2f$ [s] = %d [frame]\ntumble rate = %.2f [s$^{-1}$]"
        % (-1 / coeff[0], -1 / coeff[0] * fps, -coeff[0]),
    )
    ax.axvline(20 / fps, linestyle="dashed")
    ax.set_yscale("log")
    ax.set_xlabel("Running time [s]")
    ax.set_ylabel("Count")
    # ax.set_ylim(bottom=1)
    ax.legend()

    plt.show()

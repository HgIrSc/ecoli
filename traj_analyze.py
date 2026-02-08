import sys
from typing import Callable, Literal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def filtered_speed(
    traj: pd.DataFrame,
    window: int = 51,
    polyorder: int = 3,
    delta: float = 1.0,
) -> pd.Series:
    dx = traj.groupby("particle")["x"].transform(
        lambda x: savgol_filter(x, window, polyorder, deriv=1, delta=delta)
    )
    dy = traj.groupby("particle")["y"].transform(
        lambda x: savgol_filter(x, window, polyorder, deriv=1, delta=delta)
    )

    return (dx.pow(2) + dy.pow(2)).pow(0.5) * 0.66 / 20


def filtered_curv(
    traj: pd.DataFrame,
    window: int = 51,
    polyorder: int = 3,
    delta: float = 1.0,
    cutoff: tuple[float, float] = (1 / 2044, 1 / 10),
) -> pd.Series:
    dx = traj.groupby("particle")["x"].transform(
        lambda x: savgol_filter(x, window, polyorder, deriv=1, delta=delta)
    )
    dy = traj.groupby("particle")["y"].transform(
        lambda x: savgol_filter(x, window, polyorder, deriv=1, delta=delta)
    )
    ddx = traj.groupby("particle")["x"].transform(
        lambda x: savgol_filter(x, window, polyorder, deriv=2, delta=delta)
    )
    ddy = traj.groupby("particle")["y"].transform(
        lambda x: savgol_filter(x, window, polyorder, deriv=2, delta=delta)
    )

    numerator = np.abs((dx * ddy - ddx * dy).to_numpy())
    denominator = ((dx.pow(2) + dy.pow(2)).pow(1.5)).to_numpy()
    curv = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(denominator),
        where=(denominator != 0),
    )
    curv[curv > cutoff[1]] = np.nan
    curv[curv < cutoff[0]] = np.nan

    return curv


def locate_tumbles(
    traj: pd.DataFrame,
    filter_window: int = 7,
    detect_window: int = 1,
    min_distance: int = 40,
    thresh: int = 1,
    full: bool = False,
):
    traj["filtered_v"] = filtered_speed(traj, window=filter_window)
    traj["filtered_curv"] = filtered_curv(traj, window=filter_window)

    traj["norm_v"] = (
        traj["filtered_v"] - traj.groupby("particle")["filtered_v"].mean()
    ) / traj.groupby("particle")["filtered_v"].std()
    traj["norm_curv"] = (
        traj["filtered_curv"] - traj.groupby("particle")["filtered_curv"].mean()
    ) / traj.groupby("particle")["filtered_curv"].std()

    tumbles = np.logical_and(
        traj.groupby("particle")["norm_v"].transform(
            lambda nv: (nv.lt(-thresh))
            .rolling(detect_window)
            .aggregate(np.logical_or.reduce)
        ),
        traj.groupby("particle")["norm_curv"].transform(
            lambda nc: (nc.abs().gt(thresh))
            .rolling(detect_window)
            .aggregate(np.logical_or.reduce)
        ),
    )

    # Clean up tumbles within min_distance
    # idx = 0
    # while idx < len(tumbles):
    #     if tumbles.iloc[idx]:
    #         tumbles.iloc[idx + 1 : idx + min_distance + 1] = False
    #         idx += min_distance + 1
    #     else:
    #         idx += 1

    return tumbles


class DataVisualizationCanvas:
    def __init__(
        self,
        traj_file: str,
        scaling: int = 1,
        filter_func: Callable[[pd.Series], bool] = (lambda x: True),
    ) -> None:
        font_size = 8.0 * scaling
        large_font_size = 9.0 * scaling
        small_font_size = 7.0 * scaling
        # medium_font_size = font_size

        plt.rcParams.update(
            {
                "lines.linewidth": 0.7 * scaling,
                "font.family": "sans-serif",
                "font.sans-serif": "Noto Sans SC",
                "font.size": font_size,
                "axes.linewidth": 0.5 * scaling,
                "axes.titlesize": large_font_size,
                "axes.labelsize": small_font_size,
                "xtick.labelsize": small_font_size,
                "ytick.labelsize": small_font_size,
                "xtick.major.size": 2 * scaling,
                "xtick.minor.size": 1 * scaling,
                "xtick.direction": "in",
                "xtick.major.width": 0.5 * scaling,
                "xtick.minor.width": 0.5 * scaling,
                "ytick.major.size": 2 * scaling,
                "ytick.minor.size": 1 * scaling,
                "ytick.direction": "in",
                "ytick.major.width": 0.5 * scaling,
                "ytick.minor.width": 0.5 * scaling,
                "legend.fontsize": small_font_size,
                "figure.figsize": (2.76 * scaling, 2.76 * 0.6 * scaling),
                "figure.titlesize": large_font_size,
                "figure.labelsize": large_font_size,
                "savefig.dpi": "figure",
            }
        )

        self.fig, self.ax = plt.subplots()
        self.fig.set_layout_engine("constrained")

        self.traj = pd.read_parquet(traj_file)
        self.traj = self.traj.groupby("particle").filter(filter_func)
        self.traj.dropna(inplace=True)
        if len(self.traj) == 0:
            raise RuntimeError("No traj survived after filtering")

    def test_plot(self) -> None:
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x)
        self.ax.plot(x, y)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

    def plot_all_traj(self, alpha: float = 0.8) -> None:
        # for pid, ptraj in self.traj.groupby("particle"):
        #     self.ax.plot(
        #         ptraj.x * 0.66, ptraj.y * 0.66, "b-", alpha=alpha, linewidth=0.5
        #     )

        self.ax.plot(self.traj.x, self.traj.y, "bo", alpha=alpha, markersize=0.1)

        self.ax.set_xlim(0, 1024 * 0.66)
        self.ax.set_ylim(0, 1022 * 0.66)
        self.ax.set_xlabel("x [\u03bcm]")
        self.ax.set_ylabel("y [\u03bcm]")
        self.ax.set_aspect("equal")

    def plot_t_y_pdf(
        self, frame_span: tuple[int | None, int | None] | None = None
    ) -> None:
        if frame_span is None:
            fspan = np.linspace(self.traj.frame.min(), self.traj.frame.max(), 100)
        else:
            fmin = 0 if frame_span[0] is None else frame_span[0]
            fmax = self.traj.frame.max() if frame_span[1] is None else frame_span[1]
            fspan = np.linspace(fmin, fmax, 100)
        yedges = np.linspace(0, 1022 * 0.66, 100)
        ydist_all = []
        for fstart, fend in zip(fspan[:-1], fspan[1:]):
            part_traj = self.traj[self.traj.frame.between(fstart, fend)]
            hist, _ = np.histogram(part_traj.y * 0.66, bins=yedges)
            ydist_all.append(hist / hist.sum() / 0.66)
        ydist_all_np = np.array(ydist_all)

        aximg = self.ax.imshow(
            ydist_all_np.T,
            origin="lower",
            extent=(fspan[0] / 20, fspan[-1] / 20, yedges[0], yedges[-1]),
            aspect="auto",
            cmap="viridis",
            interpolation="bilinear",
        )
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("y [\u03bcm]")
        plt.colorbar(
            aximg, ax=self.ax, pad=0.02, label="1D PDF along $y$ [\u03bcm$^{-1}$]"
        )

    def plot_y_pdf(self, frame: int | None = None, frame_len: int = 100) -> None:
        f = self.traj.frame.max() if frame is None else frame
        traj = self.traj[self.traj.frame.between(f - frame_len, f)]
        yedges = np.linspace(0, 1022 * 0.66, 100)
        hist, _ = np.histogram(traj.y * 0.66, bins=yedges)

        self.ax.step(yedges[1:], hist / hist.sum() / 0.66)
        self.ax.set_xlabel("y [\u03bcm]")
        self.ax.set_ylabel("PDF [\u03bcm$^{-1}$]")

    def plot_lagtime_msd_counts(self, coord: Literal["x", "y", "both"] = "y") -> None:
        def msd_lagtime(
            _traj: pd.DataFrame, _coord: Literal["x", "y", "both"] = "y"
        ) -> pd.DataFrame:
            def _msd_lagtime(__traj, __coord):
                disp = __traj[__coord]
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

            if _coord == "both":
                return _msd_lagtime(_traj, "x") + _msd_lagtime(_traj, "y")
            else:
                return _msd_lagtime(_traj, _coord)

        grouped = self.traj.groupby("particle")
        all_msd = []
        for _, ptraj in grouped:
            all_msd.append(msd_lagtime(ptraj, "y"))

        times = np.concatenate([np.arange(len(msd)) for msd in all_msd]) / 20
        msds = np.concatenate(all_msd) * 0.66**2

        tedges = np.geomspace(times[times > 0].min(), times.max(), 100)
        msdedges = np.geomspace(1e-3, msds.max(), 100)

        cmap = plt.colormaps["viridis"]
        cmap = cmap.with_extremes(bad=cmap(0))
        hist, _, _ = np.histogram2d(times, msds, bins=(tedges, msdedges))
        pcm = self.ax.pcolormesh(tedges, msdedges, hist.T, cmap=cmap)
        self.ax.axline(
            (tedges.min(), 1e-2),
            slope=1,
            linestyle="dashed",
            color="white",
            label="slope = 1",
        )
        self.ax.axline(
            (tedges.min(), 1e-2),
            slope=2,
            linestyle="dashed",
            color="white",
            label="slope = 2",
        )
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlabel("Lag time [s]")
        self.ax.set_ylabel("$MSD$ [\u03bcm$^2$]")
        self.ax.legend(loc="upper left")
        plt.colorbar(pcm, ax=self.ax, label="# points")

    def plot_time_y_speed(self, tstep: int = 100, ystep: int = 10) -> None:
        tspan = np.arange(0, self.traj.frame.max(), tstep)
        yspan = np.arange(0, self.traj.y.max(), ystep)

        speed_mesh = np.zeros(shape=(tspan.size, yspan.size))
        count_mesh = np.zeros(shape=(tspan.size, yspan.size))

        for row in range(len(self.traj)):
            i = int(self.traj.frame.iloc[row] / tstep)
            j = int(self.traj.y.iloc[row] / ystep)
            count_mesh[i, j] += 1
            speed_mesh[i, j] += self.traj.v.iloc[row]

        speed_mesh /= count_mesh

        cmap = plt.colormaps["viridis"]
        cmap = cmap.with_extremes(bad=cmap(0))

        pcm = self.ax.pcolormesh(
            tspan / 20, yspan * 0.66, speed_mesh.T, shading="nearest", cmap=cmap
        )
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("y [\u03bcm]")

        plt.colorbar(pcm, ax=self.ax, label="Speed [\u03bcm/s]")

    def plot_vmean_y_pdf(self, vbins: int = 100, ybins: int = 100) -> None:
        vmean = self.traj.groupby("particle")["v"].transform(lambda x: x.mean())

        vspan = np.linspace(vmean.min(), vmean.max(), vbins)
        yspan = np.linspace(0, self.traj.y.max(), ybins)

        hist, xedges, yedges = np.histogram2d(vmean, self.traj.y, bins=(vspan, yspan))
        hist_sum = hist.sum(axis=1, keepdims=True)
        hist = np.divide(hist, hist_sum, out=np.zeros_like(hist), where=(hist_sum > 0))
        # hist = hist[: len(vspan[vspan < 30]) - 1, :]
        # vspan = vspan[vspan < 30]

        pcm = self.ax.pcolormesh(vspan, yspan * 0.66, hist.T, cmap="viridis")
        self.ax.set_xlabel("Mean speed [\u03bcm/s]")
        self.ax.set_ylabel("y [\u03bcm]")
        plt.colorbar(pcm, ax=self.ax, label="PDF [\u03bcm$^{-1}$]")

    def plot_y_vmean_pdf(self, vbins: int = 100, ybins: int = 100) -> None:
        vmean = self.traj.groupby("particle")["v"].transform(lambda x: x.mean())

        vspan = np.linspace(vmean.min(), vmean.max(), vbins)
        yspan = np.linspace(0, self.traj.y.max(), ybins)

        hist, xedges, yedges = np.histogram2d(self.traj.y, vmean, bins=(yspan, vspan))
        hist_sum = hist.sum(axis=1, keepdims=True)
        hist = np.divide(hist, hist_sum, out=np.zeros_like(hist), where=(hist_sum > 0))

        pcm = self.ax.pcolormesh(yspan * 0.66, vspan, hist.T)
        self.ax.set_xlabel("y [\u03bcm]")
        self.ax.set_ylabel("Mean speed [\u03bcm/s]")
        plt.colorbar(pcm, ax=self.ax, label="PDF [\u03bcm$^{-1}$ s]")

    def plot_radius_y_pdf(self, rstep: int = 10, ystep: int = 10) -> None:
        radius = 1 / filtered_curv(self.traj)

        self.traj["radius"] = radius
        self.traj.dropna(inplace=True)

        rspan = np.linspace(
            self.traj.radius.min(),
            self.traj.radius.max(),
            int((self.traj.radius.max() - self.traj.radius.min()) / rstep),
        )
        yspan = np.arange(0, self.traj.y.max(), ystep)

        hist, xedges, yedges = np.histogram2d(
            self.traj.radius, self.traj.y, bins=(rspan, yspan)
        )
        hist_sum = hist.sum(axis=1, keepdims=True)
        hist_sum[hist_sum == 0] = 1
        hist /= hist_sum

        pcm = self.ax.pcolormesh(rspan * 0.66, yspan * 0.66, hist.T)
        self.ax.set_xlabel("$R=1/\\kappa$ [\u03bcm]")
        self.ax.set_ylabel("y [\u03bcm]")
        plt.colorbar(pcm, ax=self.ax, label="PDF [\u03bcm$^{-1}$]")

    def plot_curv_y_pdf(
        self, window: int = 51, cbins: int = 100, ystep: int = 10
    ) -> None:
        curv = filtered_curv(self.traj, window=window)

        self.traj["radius"] = curv
        self.traj.dropna(inplace=True)

        cspan = np.linspace(self.traj.radius.min(), self.traj.radius.max(), cbins)
        yspan = np.arange(0, self.traj.y.max(), ystep)

        hist, xedges, yedges = np.histogram2d(
            self.traj.radius, self.traj.y, bins=(cspan, yspan)
        )
        hist_sum = hist.sum(axis=1, keepdims=True)
        hist = np.divide(hist, hist_sum, out=np.zeros_like(hist), where=(hist_sum > 0))

        pcm = self.ax.pcolormesh(cspan / 0.66, yspan * 0.66, hist.T)
        self.ax.set_xlabel("$\\kappa$ [\u03bcm$^{-1}$]")
        self.ax.set_ylabel("y [\u03bcm]")
        plt.colorbar(pcm, ax=self.ax, label="PDF [\u03bcm$^{-1}$]")

    def test_savitzky_golay_filer(self) -> None:
        pid = np.random.choice(self.traj.particle.unique())

        traj = self.traj[self.traj.particle == pid]

        curv = filtered_curv(traj, window=11)

        self.ax.scatter(traj.x, traj.y, c=curv)
        self.ax.set_xlabel("x [pixel]")
        self.ax.set_xlabel("y [pixel]")
        self.ax.set_aspect("equal")

    def test_tumbles(self) -> None:
        pid = np.random.choice(self.traj.particle.unique())

        traj = self.traj[self.traj.particle == pid].copy()

        tumbles = locate_tumbles(
            traj, filter_window=9, detect_window=3, min_distance=3, thresh=1
        )

        self.ax.plot(traj.x, traj.y, "-", label="traj")
        self.ax.scatter(
            traj.loc[tumbles, "x"],
            traj.loc[tumbles, "y"],
            marker="^",
            color="r",
            label="tumbles",
        )
        self.ax.set_xlabel("x [pixel]")
        self.ax.set_xlabel("y [pixel]")
        self.ax.set_aspect("equal")
        self.ax.legend()

    def test_length_stat(self) -> None:
        length = (
            self.traj.groupby("particle").size()
            * self.traj.groupby("particle")["v"].mean()
            / 20
        )

        hist, xedges = np.histogram(length.to_numpy(), bins=100, density=True)
        hist = hist / np.sum(hist)

        self.ax.step(xedges[:-1], hist)
        self.ax.set_xlabel("traj.length [\u03bcm]")
        self.ax.set_ylabel("Count")


if __name__ == "__main__":
    basename = sys.argv[1]

    suffix = ".filtered.parquet.gzip"
    suffix_figure = ".vmsd_filtered.after300s"

    canvas = DataVisualizationCanvas(
        "trajs/" + basename + suffix,
        scaling=3,
        filter_func=(lambda x: x.frame.min() > 300 * 20),
    )
    canvas.plot_t_y_pdf()
    plt.savefig("figures/pdf/" + basename + suffix_figure + ".ydist_time.pdf")

    canvas = DataVisualizationCanvas(
        "trajs/" + basename + suffix,
        scaling=3,
        filter_func=(lambda x: x.frame.min() > 300 * 20),
    )
    canvas.plot_vmean_y_pdf()
    plt.savefig("figures/pdf/" + basename + suffix_figure + ".vmean_y_pdf.pdf")

    canvas = DataVisualizationCanvas(
        "trajs/" + basename + suffix,
        scaling=3,
        filter_func=(lambda x: x.frame.min() > 300 * 20 and len(x) > 100),
    )
    canvas.plot_curv_y_pdf()
    plt.savefig("figures/pdf/" + basename + suffix_figure + ".curvature_y_pdf.pdf")

    # canvas.plot_all_traj(alpha=0.2)
    # plt.show()

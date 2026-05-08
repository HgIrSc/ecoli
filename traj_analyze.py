# %%
# Import all libraries and define all functions
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.signal import savgol_filter


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


config = load_config("conf/config_1.25.json")


def msd_lagtime(
    traj: pd.DataFrame,
    on_coord: str,
) -> np.ndarray:
    def _msd_lagtime(traj, on_coord):
        disp = traj[on_coord]
        N = len(disp)

        D = np.append(np.square(disp), 0)
        Q = 2 * np.sum(D)
        S1 = np.zeros(N)
        for m in range(N):
            Q = Q - D[m - 1] - D[N - m]
            S1[m] = Q / (N - m)

        F = np.fft.fft(disp, n=(2 * N))
        PSD = F * F.conjugate()
        S2 = np.fft.ifft(PSD)[:N].real / np.arange(1, N + 1)[::-1]

        msd = (S1 - 2 * S2)[1:-1]
        return msd

    if on_coord == "xy":
        return _msd_lagtime(traj, "x") + _msd_lagtime(traj, "y")
    elif on_coord == "yz":
        return _msd_lagtime(traj, "y") + _msd_lagtime(traj, "z")
    elif on_coord == "zx":
        return _msd_lagtime(traj, "x") + _msd_lagtime(traj, "z")
    elif on_coord == "xyz":
        return (
            _msd_lagtime(traj, "x") + _msd_lagtime(traj, "y") + _msd_lagtime(traj, "z")
        )
    else:
        return _msd_lagtime(traj, on_coord)


def powerlaw_msd(
    traj: pd.DataFrame,
    on_coord: str,
):
    if len(traj) < 20:
        return np.nan

    msd = msd_lagtime(traj, on_coord)[: len(traj // 5)]

    log_lagtime = np.log(np.arange(1, len(msd) + 1))
    log_msd = np.log(msd, out=np.zeros_like(msd), where=msd > 0)

    alpha = np.polyfit(log_lagtime, log_msd, 1)[0]
    return alpha


def locate_tumble(
    ptraj,
    omega_thresh=35,
    subsampling=2,
    return_subsampling=False,
    do_smooth=False,
):
    n_origin = len(ptraj)
    ptraj = ptraj[::subsampling]

    if do_smooth:
        if len(ptraj) > 4:
            ptraj["x"] = savgol_filter(ptraj.x, 3, 2)
            ptraj["y"] = savgol_filter(ptraj.y, 3, 2)

    # Locate tumble
    vx = np.diff(ptraj.x, append=np.nan)
    vy = np.diff(ptraj.y, append=np.nan)
    v = np.sqrt(vx**2 + vy**2)

    omega = (
        np.arccos((vx[1:-1] * vx[:-2] + vy[1:-1] * vy[:-2]) / v[1:-1] / v[:-2])
        / np.pi
        * 180
    )

    n = len(ptraj)
    is_tumble = np.zeros(n, dtype=bool)

    current_state = "tumbling"
    for i in range(n - 2):
        if i <= n - 3:
            if all(omega[i : i + 3] < omega_thresh):
                current_state = "running"

        if current_state == "running":
            if i <= n - 2:
                if all(omega[i : i + 2] > omega_thresh):
                    current_state = "tumbling"

            if omega[i] > omega_thresh:
                if i >= 2 and i <= n - 3:
                    vx_after = ptraj.x.iloc[i + 2] - ptraj.x.iloc[i]
                    vy_after = ptraj.y.iloc[i + 2] - ptraj.y.iloc[i]
                    vx_before = ptraj.x.iloc[i] - ptraj.x.iloc[i - 2]
                    vy_before = ptraj.y.iloc[i] - ptraj.y.iloc[i - 2]
                    v_after = np.sqrt(vx_after**2 + vy_after**2)
                    v_before = np.sqrt(vx_before**2 + vy_before**2)
                    delta_dir = (
                        np.arccos(
                            (vx_after * vx_before + vy_after * vy_before)
                            / v_after
                            / v_before
                        )
                        / np.pi
                        * 180
                    )
                    if delta_dir > omega_thresh:
                        current_state = "tumbling"

        is_tumble[i + 1] = current_state == "tumbling"

    if return_subsampling:
        return is_tumble
    else:
        return np.repeat(is_tumble, subsampling)[:n_origin]


def get_tumble_data(bulk_trajs, subsampling, omega_thresh, do_smooth):
    plot_data = []
    for pid, ptraj in bulk_trajs.groupby("particle"):
        if len(ptraj) < 10:
            continue
        is_tumble = locate_tumble(
            ptraj,
            subsampling=subsampling,
            omega_thresh=omega_thresh,
            do_smooth=do_smooth,
        )
        tumble_start = np.nonzero(np.diff(is_tumble.astype(int), prepend=1) > 0)[0]
        tumble_end = np.nonzero(np.diff(is_tumble.astype(int), append=1) < 0)[0]

        if np.sum(tumble_start) < 2:
            continue

        tumble_start = np.append(tumble_start, len(ptraj) - 1)
        tumble_end = np.insert(tumble_end, 0, 0)

        running_delta_x = (
            ptraj["x"].to_numpy()[tumble_start] - ptraj["x"].to_numpy()[tumble_end]
        )
        running_delta_y = (
            ptraj["y"].to_numpy()[tumble_start] - ptraj["y"].to_numpy()[tumble_end]
        )

        out_delta_x = running_delta_x[1:] * config["um_per_pixel"]
        out_delta_y = running_delta_y[1:] * config["um_per_pixel"]
        in_delta_x = running_delta_x[:-1] * config["um_per_pixel"]
        in_delta_y = running_delta_y[:-1] * config["um_per_pixel"]

        out_angle = np.atan2(out_delta_x, -out_delta_y) / np.pi * 180
        in_angle = np.atan2(in_delta_x, -in_delta_y) / np.pi * 180
        theta = np.abs(out_angle - in_angle)
        theta[theta > 180] = 360 - theta[theta > 180]

        runnning_time = (
            ptraj["frame"].to_numpy()[tumble_start[1:]]
            - ptraj["frame"].to_numpy()[tumble_end[1:]]
        ) / config["frame_per_second"]
        runnning_time[-1] = np.nan

        plot_data.append(
            pd.DataFrame(
                {
                    "running_time": runnning_time,
                    "tumble_start_frame": ptraj["frame"].to_numpy()[tumble_start[:-1]],
                    "tumble_start_y": ptraj["y"].to_numpy()[tumble_start[:-1]]
                    * config["um_per_pixel"],
                    "out_angle": out_angle,
                    "in_angle": in_angle,
                    "theta": theta,
                }
            )
        )
    plot_data = pd.concat(plot_data)
    return plot_data


def bin_index(x, bins):
    deltax = (x.max() - x.min()) / bins
    return ((x - x.min()) / deltax).astype(int)


def bin_edges(x, bins):
    return np.linspace(x.min(), x.max(), bins + 1)


# %%
# Load trajectories
trajs = pd.read_parquet(
    os.path.join(
        config["frame_dir"],
        f"../trajs_morph/{os.path.basename(config['frame_dir'])}.parquet.gzip",
    )
)
print(trajs.frame.nunique())

# %%
# Filtering
# filtered_trajs = trajs.groupby("particle").filter(lambda p: len(p) > 20)
filtered_trajs = trajs.groupby("particle").filter(lambda p: powerlaw_msd(p, "y") > 1.8)
filtered_trajs = filtered_trajs[filtered_trajs.y.between(200, 600)]
# fig, ax = plt.subplots()
# ax.hist(
#     (
#         trajs
#         .groupby("particle")
#         .apply(lambda p: powerlaw_msd(p, "y"))
#         .dropna()
#     ),
#     bins=100
# )
# ax.set_xlim(0, 2)
# plt.show()

# %%
# Test filtering
fig, ax = plt.subplots(2, 2)
ax[0, 0].hist(trajs.x, bins=20)
ax[0, 1].hist(trajs.y, bins=20)
ax[0, 0].set_xlabel("X")
ax[0, 1].set_xlabel("Y")
ax[1, 0].hist(filtered_trajs.x, bins=20)
ax[1, 1].hist(filtered_trajs.y, bins=20)
ax[1, 0].set_xlabel("X")
ax[1, 1].set_xlabel("Y")
plt.show()

# %%
# Generate velocity data
subsampling = 2
window = 4
polyorder = 2
velocity_data = []
for pid, ptraj in filtered_trajs.groupby("particle"):
    if len(ptraj) < subsampling:
        continue
    sub_ptraj = ptraj[::subsampling]
    if len(sub_ptraj) > window:
        vx = (
            np.diff(savgol_filter(sub_ptraj.x, window, polyorder), append=np.nan)
            / subsampling
            * config["frame_per_second"]
            * config["um_per_pixel"]
            / subsampling
        )
        vy = (
            np.diff(savgol_filter(sub_ptraj.y, window, polyorder), append=np.nan)
            / subsampling
            * config["frame_per_second"]
            * config["um_per_pixel"]
            / subsampling
        )
    else:
        vx = (
            np.diff(sub_ptraj.x, append=np.nan)
            / config["frame_per_second"]
            / subsampling
            * config["um_per_pixel"]
        )
        vy = (
            np.diff(sub_ptraj.y, append=np.nan)
            / config["frame_per_second"]
            / subsampling
            * config["um_per_pixel"]
        )
    v = np.sqrt(vx**2 + vy**2)
    velocity_data.append(
        pd.DataFrame(
            {
                "frame": ptraj.frame,
                "y_um": ptraj.y * config["um_per_pixel"],
                "x_um": ptraj.x * config["um_per_pixel"],
                "vx": np.repeat(vx, subsampling)[: len(ptraj)],
                "vy": np.repeat(vy, subsampling)[: len(ptraj)],
                "v": np.repeat(v, subsampling)[: len(ptraj)],
            }
        )
    )
velocity_data = pd.concat(velocity_data)

# Drift analyze
# drift = velocity_data.vx.mean()
# velocity_data["vx"] = velocity_data.vx - drift
# velocity_data["v"] = np.sqrt(velocity_data.vx**2 + velocity_data.vy**2)

# Save velocity data
velocity_data.to_parquet(
    os.path.join(
        config["frame_dir"],
        "../trajs_morph/",
        f"{os.path.basename(config['frame_dir'])}_velocity.parquet.gzip",
    ),
    compression="gzip",
)

# %%
# Plot \rho v - v curve
velocity_data = pd.read_parquet(
    os.path.join(
        config["frame_dir"],
        "../trajs_morph/",
        f"{os.path.basename(config['frame_dir'])}_velocity.parquet.gzip",
    ),
)

grouped_velocity_data = velocity_data.groupby(bin_index(velocity_data.y_um, 20))

density = grouped_velocity_data.size()[:-1]
mean_v = grouped_velocity_data["v"].mean()[:-1]
yedges = bin_edges(velocity_data.y_um, 20)

density = density / np.sum(density) / (yedges[1] - yedges[0])
mean_v = mean_v

coeff = np.polyfit(np.log(mean_v), np.log(density * mean_v), 1)

fig, ax = plt.subplots()
ax.plot(
    mean_v, density * mean_v, "o", color="black", markerfacecolor="none", label="Data"
)
ax.plot(
    [mean_v.min(), mean_v.max()],
    np.exp(np.polyval(coeff, np.log([mean_v.min(), mean_v.max()]))),
    "--",
    color="black",
    label=f"$\\alpha$ = {coeff[0]:.2f}",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.tick_params(axis="both", which="both", direction="in")
ax.legend()
plt.show()

# %%
# Test tumble recognizer
fig, ax = plt.subplots()
for thresh in [35, 40, 45, 50]:
    plot_data = get_tumble_data(
        filtered_trajs, subsampling=1, omega_thresh=thresh, do_smooth=True
    )

    # ax.hist(plot_data["deltavy"], bins=400)
    # ax.set_xlim(-10, 10)
    ax.hist(plot_data["theta"], bins=40, label=f"{thresh}", alpha=0.5)
    # hist, xedges, yedges = np.histogram2d(plot_data["out"], theta, bins=(20, 20))
    # hist, xedges, yedges = np.histogram2d(plot_data["out"], plot_data["running_time"], bins=(20, 20))
    # hist, xedges, yedges = np.histogram2d(plot_data["in"], plot_data["out"], bins=(20, 20))
    # hist = hist / np.sum(hist, axis=1, keepdims=True)
    # pcm = ax.pcolormesh(xedges, yedges, hist.T)
    # fig.colorbar(pcm, ax=ax)
    # ax.scatter(plot_data["out"], theta, s=1, marker=".", alpha=0.3, color="black")
plt.show()

# %%
# Tumble statistics
plot_data = get_tumble_data(filtered_trajs, 2, 35, True).dropna()
fig, ax = plt.subplots()

# count = 0
# all_theta_mean = []
# for _, data in plot_data.groupby(bin_index(plot_data["tumble_start_frame"], 5)):
#     hist, xedges, yedges = np.histogram2d(data["out_angle"], data["theta"], bins=(5, 20))
#     hist = hist / np.sum(hist, axis=1, keepdims=True)
#     xcenter = (xedges[1:] + xedges[:-1])/2
#     ycenter = (yedges[1:] + yedges[:-1])/2
#     all_theta_mean.append(np.average(np.tile(ycenter, (5, 1)), axis=1, weights=hist))
#     count += 1
#     if count == 5:
#         break
# all_theta_mean = np.asarray(all_theta_mean)
# theta_mean = np.mean(all_theta_mean, axis=0)
# theta_mean_std = np.std(all_theta_mean, axis=0)
# ax.errorbar(xcenter, theta_mean, yerr=theta_mean_std)

hist, xedges, yedges = np.histogram2d(
    plot_data["in_angle"], plot_data["theta"], bins=(10, 20)
)
hist = hist / np.sum(hist, axis=1, keepdims=True)
pcm = ax.pcolormesh(xedges, yedges, hist.T)
fig.colorbar(pcm, ax=ax)

# print(np.mean(np.cos(plot_data["out_angle"])))
# ax.hist(plot_data["out_angle"], bins=20)
# ax.hist(theta, bins=40)

plt.show()

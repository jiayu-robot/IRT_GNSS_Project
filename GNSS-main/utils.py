import numpy as np
import pandas as pd
import pymap3d as pm # pip install pymap3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
import os


#这个函数是给模型性能评估使用的










def weighted_single_point_positioning(sat_positions, pseudoranges, weights, x0=None):
    """
    使用加权最小二乘法 (WLS) 进行单点定位。
    这个函数是对您标准SPP的补充。
    """
    if x0 is None:
        x0 = np.zeros(4)

    def weighted_residuals(x):
        receiver_position, clock_bias = x[:3], x[3]
        predicted_pseudoranges = np.linalg.norm(sat_positions - receiver_position, axis=1) + clock_bias
        residuals = pseudoranges - predicted_pseudoranges
        # 核心区别：返回加权后的残差
        return weights * residuals

    result = least_squares(weighted_residuals, x0, method='lm')
    return result.x[:3], result.x[3]






# Given folder, start and end str, return file name
def find_file(folder_path, start=None, end=None): 
    return [
        f for f in os.listdir(folder_path)
        if (start is None or f.startswith(start)) and (end is None or f.endswith(end))
    ]

# Convert ECEF to LLH         把GNSS解算出的地心坐标 (ECEF) 转换成地理坐标 (纬度lat、经度lon、高度h)。
def ecef_to_llh(ecef_positions):
    llh_positions = [pm.ecef2geodetic(*pos) for pos in ecef_positions]
    return np.array(llh_positions)

# Convert LLH to ECEF     把地理坐标 (LLH) 转换成 地心坐标 (ECEF)。
def llh_to_ecef(llh_positions):
    ecef_positions = [pm.geodetic2ecef(*pos) for pos in llh_positions]
    return np.array(ecef_positions)

# Convert LLH to ENU     把地理坐标 (LLH) 转换成 东北天坐标 (ENU) East-North-Up coordinate system。
def llh_to_enu(llh_positions, origin_llh):
    enu_positions = [
        pm.geodetic2enu(lat, lon, alt, origin_llh[0], origin_llh[1], origin_llh[2])
        for lat, lon, alt in llh_positions
    ]
    return np.array(enu_positions)

# Convert ENU to LLH    把 东北天坐标 (ENU) 转换成地理坐标 (LLH)。
def enu_to_llh(enu_positions, origin_llh):
    # ecef_origin = pm.geodetic2ecef(*origin_llh)
    # ell = pm.Ellipsoid(semimajor_axis=6378137.0, semiminor_axis=6356752.314245)  # Define WGS84

    llh_positions = [
        pm.enu2geodetic(e, n, u, origin_llh[0], origin_llh[1], origin_llh[2])
        for e, n, u in enu_positions
    ]
    return np.array(llh_positions)


def plot_trajectory(pos, title, type = 'llh', twoD = False, scatter = False):
    axislabel = {'llh': ['Longitude / degrees', 'Latitude / degrees', 'Height / m'], 
             'enu': ['East / m', 'North / m', 'Up / m']}
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    if len(pos.keys()) > len(colors):
        colors = plt.cm.plasma(np.linspace(0, 1, len(pos.keys())))
    # print(label[type])
    if twoD:
        fig, ax = plt.subplots()
        for i, (key, value) in enumerate(pos.items()):
            if scatter:
                ax.scatter(value[:, 0], value[:, 1], label=key, color=colors[i])
            else:
                ax.plot(value[:, 0], value[:, 1], label=key, color=colors[i])
        ax.grid(True)
    else: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, (key, value) in enumerate(pos.items()):
            if scatter:
                ax.scatter(value[:, 0], value[:, 1], value[:, 2], label=key, color=colors[i])
            else:
                ax.plot(value[:, 0], value[:, 1], value[:, 2], label=key, color=colors[i])
        ax.set_zlabel(axislabel[type][2])
    ax.set_xlabel(axislabel[type][0])
    ax.set_ylabel(axislabel[type][1])
    ax.set_title(title)
    ax.legend()
    return fig

def plot_trajectory_scatter(pos, sca, title, type = 'llh', twoD = False):
    axislabel = {'llh': ['Longitude / degrees', 'Latitude / degrees', 'Height / m'], 
             'enu': ['East / m', 'North / m', 'Up / m']}
    # colors = ['b', 'g', 'c', 'm', 'y', 'k']
    colors = plt.cm.plasma(np.linspace(0, 1, len(pos.keys())))
    colors_s = plt.cm.seismic(np.linspace(0, 1, len(sca.keys())))
    # print(label[type])
    if twoD:
        fig, ax = plt.subplots()
        for i, (key, value) in enumerate(pos.items()):
            ax.plot(value[:, 0], value[:, 1], label=key, color=colors[i])
        for j, (key, value) in enumerate(sca.items()):
            ax.scatter(value[0], value[1], color='r', marker='*', s=100)
        ax.grid(True)
    else: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, (key, value) in enumerate(pos.items()):
            ax.plot(value[:, 0], value[:, 1], value[:, 2], label=f"PL{key}" if isinstance(key, int) else key, color=colors[i])
        for j, (key, value) in enumerate(sca.items()):
            x, y, z = [p[0] for p in value], [p[1] for p in value], [p[2] for p in value]
            # ax.scatter(value[:, 0], value[:, 1], value[:, 2], color='r', marker='*', s=100)
            ax.scatter(x, y, z, color=colors_s[j], marker='*', s=100, edgecolors='black', label=f"PL{key}")
        ax.set_zlabel(axislabel[type][2])
    ax.set_xlabel(axislabel[type][0])
    ax.set_ylabel(axislabel[type][1])
    ax.set_title(title)
    ax.legend()
    return fig


def save_obs_to_csv(data, file_path='time_series_observations.csv'):
    """
    Extrat features from obseration and save them as a CSV file.
    :param data: Dictionary containing time-series observation data.
    :param file_path: Path to save the CSV file.
    """
    all_obs_data = [
        {
            'gps_time': gps_time,
            'satellite_id': obs.id,
            'sv_id': obs.sv_id,
            # 'flag_GNSS_PL': obs.flag_GNSS_PL,
            # 'flag_static_dynamic': obs.flag_static_dynamic,
            'pseudorange': obs.pseudorange,
            # 'carrier_phase': obs.carrier_phase,
            'doppler_shift': obs.doppler_shift,
            'multipath': obs.multipath,
            # 'multipath_id': obs.multipath_id,
            'rec_pow': obs.rec_pow,
            'cn0': obs.CN0,
            'azimuth': obs.azimuth,
            'elevation': obs.elevation,
            # 'doppler_rate': obs.doppler_rate,
            'pseudorange_residual': obs.pseudorange_residual,
        }
        for gps_time, obs_list in data.items()
        for obs in obs_list
    ]

     # Convert the list of dictionaries to a DataFrame and save it as a CSV file
    pd.DataFrame(all_obs_data).to_csv(file_path, index=False)
    print(f"Obseration data has been saved to {file_path}")


def single_point_positioning(sat_positions, pseudoranges, x0=None):
    # Initial guess for receiver position and clock bias
    # Inputs in meter (ecef), and output in meters (ecef)... even clock_bias (note not in sec, otherwise need to divide it by speed of light)
    if x0 is None:
        x0 = np.zeros(4)  # [x, y, z, clock_bias]
    
    def residuals(x):
        receiver_position = x[:3]
        clock_bias = x[3]
        estimated_pseudoranges = np.linalg.norm(sat_positions - receiver_position, axis=1) + clock_bias
        return pseudoranges - estimated_pseudoranges

    result = least_squares(residuals, x0, method='lm')
    return result.x[:3], result.x[3] 

import numpy as np

def single_point_positioning_lms(sat_positions, pseudoranges, x0=None, mu=0.01, max_iter=1000, tol=1e-6):
    """
    Single Point Positioning using Least Mean Squares (LMS) iterative update.
    
    sat_positions: Nx3 array of satellite ECEF positions (meters)
    pseudoranges:  Nx1 array of measured pseudoranges (meters)
    x0: optional initial guess [x, y, z, clock_bias] (meters)
    mu: step size (learning rate)
    max_iter: maximum number of iterations
    tol: convergence tolerance (meters)
    
    Returns:
        receiver_position (3,), clock_bias (scalar)
    """
    if x0 is None:
        x0 = np.zeros(4)  # [x, y, z, clock_bias]
    
    x = np.array(x0, dtype=float)

    for _ in range(max_iter):
        receiver_position = x[:3]
        clock_bias = x[3]

        # Predicted pseudoranges
        est_pseudoranges = np.linalg.norm(sat_positions - receiver_position, axis=1) + clock_bias

        # Error
        error = pseudoranges - est_pseudoranges

        # Jacobian (H matrix)
        diffs = receiver_position - sat_positions
        distances = np.linalg.norm(diffs, axis=1).reshape(-1, 1)
        H_pos = diffs / distances  # derivatives w.r.t x,y,z
        H = np.hstack((-H_pos, np.ones((len(sat_positions), 1))))  # +1 for clock bias

        # LMS update
        grad = -2 * H.T @ error / len(error)
        x_new = x - mu * grad

        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x[:3], x[3]



import numpy as np
import pandas as pd
import pymap3d as pm # pip install pymap3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
import os
import math
from scipy.spatial.transform import Rotation as R

# Convert ECEF to LLH
def ecef_to_llh(ecef_positions):
    llh_positions = [pm.ecef2geodetic(*pos) for pos in ecef_positions]
    return np.array(llh_positions)

# Convert LLH to ECEF
def llh_to_ecef(llh_positions):
    ecef_positions = [pm.geodetic2ecef(*pos) for pos in llh_positions]
    return np.array(ecef_positions)

# Convert LLH to ENU
def llh_to_enu(llh_positions, origin_llh):
    enu_positions = [
        pm.geodetic2enu(lat, lon, alt, origin_llh[0], origin_llh[1], origin_llh[2])
        for lat, lon, alt in llh_positions
    ]
    return np.array(enu_positions)

# Convert ENU to LLH
def enu_to_llh(enu_positions, origin_llh):
    # ecef_origin = pm.geodetic2ecef(*origin_llh)
    # ell = pm.Ellipsoid(semimajor_axis=6378137.0, semiminor_axis=6356752.314245)  # Define WGS84

    llh_positions = [
        pm.enu2geodetic(e, n, u, origin_llh[0], origin_llh[1], origin_llh[2])
        for e, n, u in enu_positions
    ]
    return np.array(llh_positions)

R_enu2ned = np.array([
        [0, 1,  0],
        [1, 0,  0],
        [0, 0, -1]
    ]) # R_enu2ned = R_ned2enu

R_enu2swu = np.array([
        [ 0, -1,  0],
        [-1,  0,  0],
        [ 0,  0,  1]
    ])

R_enu2test = np.array([
        [ 1, 0,  0],
        [ 0, 1,  0],
        [ 0, 0, -1]
    ])

def get_R_n2ecef(lat, lon, n='enu'):
    """
    Get the rotation matrix from local navigation frame (ENU or NED) to ECEF.

    Args:
        lat (float): Latitude in radians
        lon (float): Longitude in radians
        n (str): Navigation frame type, either 'enu' (East-North-Up) or 'ned' (North-East-Down)

    Returns:
        R_n2ecef (3x3 numpy array): Rotation matrix from local frame to ECEF
    """
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    # ENU to ECEF rotation
    R_enu2ecef = np.array([
        [-sin_lon,             cos_lon,              0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat],
        [ cos_lat * cos_lon,  cos_lat * sin_lon,  sin_lat]
    ])

    if n == 'enu':
        return R_enu2ecef
    
    if n == 'ned': 
        return R_enu2ecef @ R_enu2ned.T # R_b^ned = R_enu^ned * R_b^enu
    
    if n == 'swu':
        return R_enu2ecef @ R_enu2swu.T
        # return R_enu2swu @ R_enu2ecef #R_enu2swu @ R_enu2ecef
    if n == 'test':
        return R_enu2ecef @ R_enu2test.T

    # R_ned2ecef = R_enu2ecef @ R_enu2ned #R_ned^ecef = R_enu^ecef @ R_ned^enu (R_enu2ned = R_ned2enu)
    # return R_ned2ecef

def get_R_b2n(roll, pitch, yaw, n='enu'):
    """
    Get the rotation matrix from body frame to navigation (ENU/NED) frame.

    Args:
        roll (float): Roll angle in radians (rotation around body X-axis)
        pitch (float): Pitch angle in radians (rotation around body Y-axis)
        yaw (float): Yaw angle in radians (rotation around body Z-axis)
        n (str): Navigation frame type, either 'enu' (East-North-Up) or 'ned' (North-East-Down)

    Returns:
        R_b2n (3x3 numpy array): Rotation matrix from body frame to specified navigation frame
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    Ry = np.array([[cp, 0, sp],
                   [ 0, 1,  0],
                   [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]])

    R_b2enu = Rz @ Ry @ Rx # body frame (forward, right, down) to ENU frame (east, north, up)
    
    if n == 'enu': 
        return R_b2enu

    if n == 'ned': 
        return R_enu2ned @ R_b2enu # R_b^ned = R_enu^ned * R_b^enu
    
    if n == 'swu':
        return R_enu2swu @ R_b2enu
    if n == 'test':
        return R_enu2test @ R_b2enu

def get_R_b2ecef(lat, lon, roll, pitch, yaw, n='enu'):
    """
    Compute rotation matrix from body frame to ECEF frame.

    Args:
        lat (float): Latitude in radians
        lon (float): Longitude in radians
        roll (float): Roll angle in radians (rotation around body X-axis)
        pitch (float): Pitch angle in radians (rotation around body Y-axis)
        yaw (float): Yaw angle in radians (rotation around body Z-axis)
        n (str): Navigation frame type, either 'enu' or 'ned'

    Returns:
        R_b2ecef (3x3 numpy array): Rotation matrix from body frame to ECEF
    """
    R_n2ecef = get_R_n2ecef(lat, lon, n=n)
    # R_b2n = get_R_fru2n(roll, pitch, yaw, n=n)

    R_b2n = get_R_b2n(roll, pitch, yaw, n=n)
    R_b2ecef = R_n2ecef @ R_b2n
    return R_b2ecef


def ecef_to_latlon(x, y, z):
    """
    Convert a single ECEF position to latitude and longitude in radians.

    Args:
        x, y, z (float): ECEF coordinates in meters

    Returns:
        lat (float): Latitude in radians
        lon (float): Longitude in radians
    """
    lat_deg, lon_deg, h = pm.ecef2geodetic(x, y, z)
    return np.radians(lat_deg), np.radians(lon_deg), h


def latlon_to_ecef(lat, lon, alt=0.0):
    """
    Convert latitude and longitude in radians to ECEF coordinates.

    Args:
        lat (float): Latitude in radians
        lon (float): Longitude in radians
        alt (float): Altitude in meters (default: 0)

    Returns:
        x, y, z (float): ECEF coordinates in meters
    """
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)
    x, y, z = pm.geodetic2ecef(lat_deg, lon_deg, alt)
    return x, y, z

def get_R_fru2n(roll, pitch, yaw, n='enu'):
    """
    Rotation matrix from FRU body frame to navigation frame (ENU/NED)
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # ZYX order: yaw → pitch → roll
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    Ry = np.array([[cp, 0, sp],
                   [ 0, 1,  0],
                   [-sp, 0, cp]])
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]])

    R_n2fru = Rz @ Ry @ Rx  # world to body (FRU)
    R_fru2n = R_n2fru.T     # body to world

    if n == 'enu':
        return R_fru2n

    if n == 'ned': 
        return R_enu2ned @ R_fru2n # R_b^ned = R_enu^ned * R_b^enu
    
    if n == 'swu':
        return R_enu2swu @ R_fru2n

def frd_quat_to_rpy(qx, qy, qz, qw, nav_frame='enu'):
    """
    Convert quaternion from Forward-Right-Down (body frame)
    to roll, pitch, yaw in ENU or NED navigation frame.

    Parameters:
        qx, qy, qz, qw : float
            Quaternion components (x, y, z, w)
        nav_frame : str
            'enu' or 'ned' to choose the navigation frame

    Returns:
        roll, pitch, yaw : float
            Euler angles in degrees, ZYX convention
    """
    # Convert quaternion to rotation matrix (body w.r.t NED)
    rot_ned = R.from_quat([qx, qy, qz, qw]).as_matrix()

    if nav_frame.lower() == 'enu':
        # Convert to ENU frame
        R_ned2enu = np.array([[0, 1, 0],
                              [1, 0, 0],
                              [0, 0, -1]])
        rot_nav = R_ned2enu @ rot_ned
    elif nav_frame.lower() == 'ned':
        rot_nav = rot_ned
    else:
        raise ValueError("nav_frame must be 'enu' or 'ned'")

    # Convert rotation matrix to Euler angles (ZYX)
    yaw, pitch, roll = R.from_matrix(rot_nav).as_euler('ZYX', degrees=False)

    return roll, pitch, yaw

def nav_to_ecef_rotation_matrix(lat_rad, lon_rad, frame='enu'):
    """
    Compute rotation matrix from ENU or NED navigation frame to ECEF.
    
    Parameters:
    - lat_rad: latitude in radians
    - lon_rad: longitude in radians
    - frame: 'enu' or 'ned' (default: 'enu')
    
    Returns:
    - 3x3 rotation matrix from nav frame (ENU or NED) to ECEF
    """
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    if frame.lower() == 'enu':
        # ENU → ECEF
        R = np.array([
            [-sin_lon,             cos_lon,              0],
            [-sin_lat*cos_lon, -sin_lat*sin_lon,  cos_lat],
            [ cos_lat*cos_lon,  cos_lat*sin_lon,  sin_lat]
        ])
    elif frame.lower() == 'ned':
        # NED → ECEF
        R = np.array([
            [-sin_lat*cos_lon, -sin_lat*sin_lon,  cos_lat],
            [-sin_lon,              cos_lon,             0],
            [-cos_lat*cos_lon, -cos_lat*sin_lon, -sin_lat]
        ])
    else:
        raise ValueError("Unsupported frame type. Use 'enu' or 'ned'.")

    return R.T

def rpy_to_rotmat(roll, pitch, yaw, degrees=False): # R_nav2body
    """
    Convert roll, pitch, yaw angles to a rotation matrix (3×3).
    
    Parameters:
    - roll, pitch, yaw: angles in radians (or degrees if degrees=True)
    - degrees: set True if input angles are in degrees
    
    Returns:
    - 3x3 rotation matrix (numpy array)
    """
    rot = R.from_euler('ZYX', [yaw, pitch, roll], degrees=degrees)
    return rot.as_matrix()
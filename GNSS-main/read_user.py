import pandas as pd
import math
from collections import defaultdict
from utils import ecef_to_llh, llh_to_ecef
import os
from decimal import Decimal


# Define a class for vehicle data
class User:
    def __init__(self):
        self.imu = pd.DataFrame()  # IMU data, initialized as an empty DataFrame
        self.odo = pd.DataFrame()  # Odometer data, initialized as an empty DataFrame
        self.user = pd.DataFrame()  # User position data, initialized as an empty DataFrame
        self.pos_ecef = None 
        self.pos_llh = None
        self.estimated_ecef = None # Dictionary: {gps_time: estimated ECEF positions} 
        self.estimated_cb = None # Dictionary: {gps_time: estimated receiver clock bias} 

    def read_user_data(self, file_path):
        """
        Read user data from a specified file.
        Position (X, Y, Z) [m], Velocity Vector (X, Y, Z) [m/s], Acceleration Vector (X, Y, Z) [m/s²] (all in ECEF) 
        Roll Angle [rad], Pitch Angle [rad], Yaw Angle [rad]. Zero for static scenarios or when vehicle attitude simulation is deactivated.
        Rates of Roll, Pitch, and Yaw Angles [rad/s]. Zero for static scenarios or when vehicle attitude simulation is deactivated.
        Acceleration for rotation axes (X, Y, Z) [rad/s²]. Zero for static scenarios or when vehicle attitude simulation is deactivated.
        """
        user_data = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if not line.startswith('#'):
                    continue  # Skip non-header lines
                parts = line.split()
                try:
                    gps_time = float(parts[1])  # Keep the GPS time as float
                    x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
                    vx, vy, vz = float(parts[8]), float(parts[9]), float(parts[10])
                    ax, ay, az = float(parts[11]), float(parts[12]), float(parts[13])
                    rol, pit, yaw = float(parts[16]), float(parts[17]), float(parts[18])
                    drol, dpit, dyaw = float(parts[19]), float(parts[20]), float(parts[21])
                    ddrol, ddpit, ddyaw = float(parts[22]), float(parts[23]), float(parts[24])
                    user_data.append([gps_time, x, y, z, vx, vy, vz, ax, ay, az, rol, pit, yaw, drol, dpit, dyaw, ddrol, ddpit, ddyaw])
                except (ValueError, IndexError):
                    continue

        # Define column names
        columns = ['GPS Time', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'rol', 'pit', 'yaw', 'drol', 'dpit', 'dyaw', 'ddrol', 'ddpit', 'ddyaw']
        df = pd.DataFrame(user_data, columns=columns)
        df["GPS Time"] = df["GPS Time"].apply(lambda x: Decimal(str(x)).quantize(Decimal("0.01")))
        # df['GPS Time'] = df['GPS Time'].apply(lambda x: float(f"{x:.5f}"))  # Format GPS Time with 5 decimal places as float
        # pd.set_option("display.float_format", "{:.5f}".format)  # Set display option to show full precision
        self.user = df

        self.pos_ecef = [[xi, yi, zi] for xi, yi, zi in zip(self.user.x, self.user.y, self.user.z)]
        self.pos_llh = ecef_to_llh(self.pos_ecef)

    def read_imu_data(self, file_path):
        """
        Read IMU data from a specified file.
        """
        imu_data = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')  # Split by comma
                try:
                    gps_time = float(parts[0])
                    acc_x = float(parts[1])
                    acc_y = float(parts[2])
                    acc_z = float(parts[3])
                    roll_rate = float(parts[4])
                    pitch_rate = float(parts[5])
                    yaw_rate = float(parts[6])
                    imu_data.append([gps_time, acc_x, acc_y, acc_z, roll_rate, pitch_rate, yaw_rate])
                    # print(gps_time)
                    # exit()
                except (ValueError, IndexError):
                    continue

        # Define column names
        # columns = ['GPS Time', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'rol', 'pit', 'yaw', 'drol', 'dpit', 'dyaw', 'ddrol', 'ddpit', 'ddyaw']

        # columns = ['GPS Time', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Roll Rate', 'Pitch Rate', 'Yaw Rate']
        columns = ['GPS Time', 'ax', 'ay', 'az', 'drol', 'dpit', 'dyaw']
        df = pd.DataFrame(imu_data, columns=columns)
        df["GPS Time"] = df["GPS Time"].apply(lambda x: Decimal(str(x)).quantize(Decimal("0.01")))

        # pd.set_option("display.float_format", "{:.10f}".format)  # Set display option for numerical values
        # df['GPS Time'] = df['GPS Time'].apply(lambda x: float(f"{x:.2f}"))  # Format GPS Time with 5 decimal places as float
        self.imu = df

    def read_odo_data(self, file_path):
        """
        Read odometer data.
        """
        wheel_radius = 0.3
        odm_data = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')  # Split by comma
                try:
                    gps_time = float(parts[0])
                    velocity = float(parts[1]) * wheel_radius
                    distance = float(parts[2])
                    odm_data.append([gps_time, velocity, distance])
                except (ValueError, IndexError):
                    continue

        # Define column names
        columns = ['GPS Time', 'Velocity', 'Distance']
        df = pd.DataFrame(odm_data, columns=columns)
        df["GPS Time"] = df["GPS Time"].apply(lambda x: Decimal(str(x)).quantize(Decimal("0.01")))
        # pd.set_option("display.float_format", "{:.10f}".format)  # Set display option for numerical values
        # df['GPS Time'] = df['GPS Time'].apply(lambda x: float(f"{x:.5f}"))  # Format GPS Time with 5 decimal places as float
        self.odo = df


    def read_real_user_data(self, real_user_pos_file_path, real_imu_file_path, real_odo_file_path = None):

        self.user = pd.read_csv(real_user_pos_file_path)
        self.imu = pd.read_csv(real_imu_file_path)

        self.pos_ecef = [[xi, yi, zi] for xi, yi, zi in zip(self.user.x, self.user.y, self.user.z)]
        self.pos_llh = ecef_to_llh(self.pos_ecef)
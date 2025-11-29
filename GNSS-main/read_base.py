import pandas as pd
from datetime import datetime, timedelta
import os
from math import isclose
import pymap3d as pm 
import numpy as np
from utils import * 
from collections import OrderedDict
import copy




# The difference between GPS and UTC is 18 seconds
GPS_UTC_DIFF = 18
# GPS epoch start date (January 6, 1980)
GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)
THERMO_NOISE = -174
RECPOW2CN0_MU = 0.7947163182343785
RECPOW2CN0_SIGMA = 0.12903513950721823


class Obs:
    def __init__(self, gps_time, satellite_id, sv_id, pseudorange, carrier_phase, doppler_shift, cn0=None):
        
        self.gps_time = gps_time    # GPS timestamp
        # self.utc_time = utc_time    # UTC time (datetime object)
        self.id = satellite_id      # Satellite ID (e.g., G03)
        self.sv_id = sv_id
        self.pseudorange = pseudorange # meter
        self.carrier_phase = carrier_phase # cycle
        self.doppler_shift = doppler_shift # Hz
        self.cn0 = cn0 
        # self.rec_pow = rec_pow # dB, received power from simulator
        # self.CN0 = rec_pow - THERMO_NOISE - np.random.normal(RECPOW2CN0_MU, RECPOW2CN0_SIGMA) # waiting for ros hitl ...
        # # self.SNR = None # dB-Hz, simulator does not provide 
        
        # self.satellite_pos = satellite_pos   # Satellite position (km -> m, ecef)
        # self.satellite_vel = satellite_vel   # Satellite velocity (dm/s)
        # self.clock_bias = clock_bias     # Satellite Clock Corrections (microseconds)
        
        # self.multipath = int(multipath)      # Multipath effect initially set to False=0
        # self.multipath_id = multipath_id  # New attribute to store the number of multipath signals
        
        # self.pseudorange_corrected = None # self.pseudorange - 299792458.0 * (self.clock_bias * 1e-6)  
        # self.pseudorange_single_diff = None # from static base station
        
        # self.azimuth, self.elevation = None, None

        # self.prev_obs = None
        # self.doppler_rate = None
        # self.pseudorange_residual = None

        # self.flag_GNSS_PL, self.flag_static_dynamic = None, None

    def calculate_pseudorange_single_diff(self, rho_sat2base, base_pos):
        rho_sat2base_expected = np.linalg.norm(np.array(self.satellite_pos) - np.array(base_pos))
        correction = rho_sat2base - rho_sat2base_expected
        self.pseudorange_single_diff = self.pseudorange - correction
        return self.pseudorange_single_diff

def format_satellite_id(satellite):
    '''Convert satellite id and names'''
    # Remove extra "L1" or other frequency band information, keeping only the satellite ID part
    satellite = satellite.split()[0]  # Keep only "GPS-003" or "G03", etc.

    if 'PG' in satellite or 'VG' in satellite: # Handle satellite ID in SP3, such as "PG03" or "VG03" (position or velocity of G/GPS)
        satellite_number = int(satellite[2:].lstrip('0'))
    elif 'PE' in satellite or 'VE' in satellite: # Handle satellite ID in SP3, such as "PE03" or "VE03" (position or velocity of E/GAL)
        satellite_number = int(satellite[2:].lstrip('0')) + 30
    elif '-' in satellite: # ID in Multipath file
        system, satellite_number = satellite.split('-')  # Split systems and numbers
        if system == "GPS": satellite_number = int(satellite_number.lstrip('0'))
        elif system == "GAL": satellite_number = int(satellite_number.lstrip('0')) + 30
        else: print('Warning!!! Unknown format in satellite id: ', satellite)
    elif satellite.startswith('G'):
        # Handle numbering in RINEX format, e.g., "G03"
        satellite_number = int(satellite[1:].lstrip('0'))
    elif satellite.startswith('E'):
        # Handle numbering in RINEX format, e.g., "G03"
        satellite_number = int(satellite[1:].lstrip('0')) + 30
    else:
        # Handle other unknown formats, extract the number part
        print('Warning!!! Unknown format in satellite id: ', satellite)
        # satellite_number = int(satellite[2:].lstrip('0'))

    # Add prefix based on the number range
    if 1 <= satellite_number <= 30:
        return f'G{str(satellite_number).zfill(2)}', satellite_number  # For satellites numbered 1 to 30, add G prefix
    elif 30 < satellite_number: # <= 40:
        return f'PL{str(satellite_number - 30).zfill(2)}', satellite_number  # For satellites numbered 31 to 40, add PL prefix
    else:
        print("Warning!!! satellite_number incorrent: ", satellite_number)
        return f'G{str(satellite_number).zfill(2)}', satellite_number  # Default to G prefix for other cases

def parse_rinex_observation_data(rinex_file_path):
    '''Given the rinex_file_path, extract observation data, and add info to obs_dir'''
    gps_seconds = None
    obs_dir = {}

    with open(rinex_file_path, 'r') as rinex_file:
        header_ended = False
        for i, line in enumerate(rinex_file):
            if "END OF HEADER" in line:
                header_ended = True
                continue  # Skip header lines
            if not header_ended:
                continue            

            if line.startswith('>'):  # Handle timestamp lines
                year, month, day, hour, minute, second = map(float, line.split()[1:7])
                year, month, day, hour, minute = map(int, [year, month, day, hour, minute])
                gps_time = datetime(year, month, day, hour, minute, int(second)) + timedelta(seconds=second % 1)
                gps_seconds = float((gps_time - GPS_EPOCH).total_seconds())
                # print(year, month, day, hour, minute)
                # print(gps_time, gps_seconds)
                # print('-------------------------------------------')
                obs_dir[gps_seconds] = []

            elif gps_seconds is not None:  # Ensure there is a timestamp              
                parts = line.split()
                if len(parts) < 4: # Skip lines with insufficient fields
                    continue           
                satellite_id, sv_id = format_satellite_id(parts[0])
                pseudorange, carrier_phase, doppler_shift = float(parts[1]), float(parts[2]), float(parts[3])
                obs_dir[gps_seconds].append(Obs(gps_seconds, satellite_id, sv_id, pseudorange, carrier_phase, doppler_shift))

    return obs_dir

def read_real_base_obs(csv_file_path):
    base_obs = pd.read_csv(csv_file_path)
    base_obs_dir = {}
    for _, row in base_obs.iterrows():
        gps_time = float(row['gps_time']) #round(float(row['gps_time']), 1) #float(row['gps_time']) Round to 1 decimal place for consistency
        id = str(row['id']) # str + int
        sv_id = int(row['sv_id'])
        pseudorange = float(row['pseudorange']) # m
        carrier_phase = float(row['carrier_phase']) # cycles
        doppler_shift = float(row['doppler_shift']) # Hz
        # CN0 = float(row['CNo'])
        CN0 = 1
        
        obs = Obs(gps_time, id, sv_id, pseudorange, carrier_phase, doppler_shift, CN0) 

        if gps_time not in base_obs_dir:
            base_obs_dir[gps_time] = []
        base_obs_dir[gps_time].append(obs)
    return base_obs_dir

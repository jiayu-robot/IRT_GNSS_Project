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
    def __init__(self, gps_time, satellite_id, sv_id, pseudorange, carrier_phase, doppler_shift,
                  multipath = False, multipath_id = 0, satellite_pos = None, satellite_vel = None, clock_bias = None, 
                  rec_pow = -100, CN0 = None):
        
        self.gps_time = gps_time    # GPS timestamp
        # self.utc_time = utc_time    # UTC time (datetime object)
        self.id = satellite_id      # Satellite ID (e.g., G03)
        self.sv_id = sv_id # 1-30(37) (GPS) + PL (Gal id + 30), 71-102 (Gal)
        self.pseudorange = pseudorange # meter
        self.carrier_phase = carrier_phase # cycle
        self.doppler_shift = doppler_shift # Hz
        self.rec_pow = rec_pow # dB, received power from simulator
        if CN0 is None:
            self.CN0 = rec_pow - THERMO_NOISE - np.random.normal(RECPOW2CN0_MU, RECPOW2CN0_SIGMA)
        else:
            self.CN0 = CN0
        # self.SNR = None # dB-Hz, simulator does not provide 
        
        self.satellite_pos = satellite_pos   # Satellite position (km -> m, ecef)
        self.satellite_vel = satellite_vel   # Satellite velocity (dm/s)
        self.clock_bias = clock_bias     # Satellite Clock Corrections (microseconds)
        
        self.multipath = int(multipath)      # Multipath effect initially set to False=0
        self.multipath_id = multipath_id  # New attribute to store the number of multipath signals

        self.pseudorange_corrected = None # self.pseudorange - 299792458.0 * (self.clock_bias * 1e-6)  
        if self.clock_bias is not None:
            self.calculate_corrected_pseudorange()
        self.pseudorange_single_diff = None # from static base station
        self.pseudorange_nav = None # pseduorange for navigation solution (LS, KF)
        
        self.azimuth, self.elevation = None, None
        # self.azimuth_gt, self.elevation_gt = None, None

        self.prev_obs = None
        self.doppler_rate = None
        self.pseudorange_residual = None

        self.flag_GNSS_PL, self.flag_static_dynamic = None, None # 0-GNSS/1-PL, 0-static/1-dynamic

    def update_flags(self):
        self.flag_GNSS_PL = 0 if self.sv_id < 30 else 1 # GNSS=0, PL=1
        self.flag_static_dynamic = 0 if np.linalg.norm(self.satellite_vel) == 0 else 1 # static=0, dynamic=1 (mainly for PL)

    
    def update_signal_power(self, value):
        self.rec_pow = value # dB, received power from simulator
        self.CN0 = value - THERMO_NOISE - np.random.normal(RECPOW2CN0_MU, RECPOW2CN0_SIGMA)
    
    def calculate_azimuth_elevation(self, rec_pos_ecef):
        '''Given receiver position, calculate for azimuth and elevation angles'''
        # Convert receiver ECEF to LLH
        rec_llh = pm.ecef2geodetic(*rec_pos_ecef)
        # Convert satellite ECEF to LLH
        sat_llh = pm.ecef2geodetic(*self.satellite_pos)
        # Convert satellite LLH to ENU relative to the receiver's LLH
        e, n, u = pm.geodetic2enu(sat_llh[0], sat_llh[1], sat_llh[2], rec_llh[0], rec_llh[1], rec_llh[2])

        # Calculate azimuth and elevation
        azimuth = np.arctan2(e, n)  # Azimuth in radians
        self.azimuth = np.degrees(azimuth) % 360  # Convert to degrees (0-360)

        elevation = np.arctan2(u, np.sqrt(e**2 + n**2))  # Elevation in radians
        self.elevation = np.degrees(elevation)  # Convert to degrees
    
    def calculate_corrected_pseudorange(self):
        '''Calculate for corrected pseudorange, rho_corr = rho - Speed of light (m/s) * clock_bias (microseconds -> s)'''
        self.pseudorange_corrected = self.pseudorange - 299792458.0 * (self.clock_bias * 1e-6) 

    def calculate_pseudorange_single_diff(self, rho_sat2base, base_pos, ecef_frame=False, corrected_pseudorange=False):
        if ecef_frame:
            base_pos_ecef = base_pos
        else:
            base_pos_ecef = pm.geodetic2ecef(base_pos[0], base_pos[1], base_pos[2])
        rho_sat2base_expected = np.linalg.norm(np.array(self.satellite_pos) - np.array(base_pos_ecef))
        correction = rho_sat2base - rho_sat2base_expected
        if corrected_pseudorange:
            self.pseudorange_single_diff = self.pseudorange_corrected - correction
        else:
            self.pseudorange_single_diff = self.pseudorange - correction
        return self.pseudorange_single_diff
    
    def get_prev_obs(self, data):
        '''Given all previous data, get the most recent data'''
        self.prev_obs = next(
                (obs for values in reversed(data.values()) for obs in values if obs.sv_id == self.sv_id and not obs.multipath),
                None)
        # if self.prev_obs:
        #     print(self.sv_id, self.prev_obs.sv_id, self.gps_time, self.prev_obs.gps_time, self.doppler_shift, self.prev_obs.doppler_shift)
        # else: print('None', self.sv_id)

    def calculate_doppler_rate(self):
        '''Calculate for Doppler rate from most recent obs'''
        # Check if the previous observation exists
        if self.prev_obs is None or self.prev_obs.doppler_shift is None:
            self.doppler_rate = None
        else:
            delta_time = self.gps_time - self.prev_obs.gps_time
            if delta_time <= 0:
                print("\033[31mError\033[0m: prev_obs/time is incorrect")
                self.doppler_rate = None
            else:
                self.doppler_rate = (self.doppler_shift - self.prev_obs.doppler_shift) / delta_time

    def calculate_pseudorange_residual(self, receiver_pos, clock_bias, pseudorange = None):
        """
        Calculate pseudorange residual. Given:
        receiver_pos: Receiver position in ECEF format (meters) from LS/SPP
        clock_bias: Receiver clock bias in meters (NOT in seconds!) from LS/SPP
        """
        if pseudorange is None:
            pseudorange = self.pseudorange_corrected
        # Calculate geometric range
        computed_range = np.linalg.norm(np.array(self.satellite_pos) - np.array(receiver_pos))
        self.pseudorange_residual = pseudorange - computed_range - clock_bias

    # def __str__(self):
    #     return (f"Satellite ID: {self.id}, Pseudorange: {self.pseudorange}, Carrier Phase: {self.carrier_phase}, "
    #             f"Doppler: {self.doppler_shift}, UTC Time: {self.utc_time_str()}")

    # def utc_time_str(self):
    #     return self.utc_time.strftime('%Y-%m-%d %H:%M:%S')

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

def get_sat_posvel(gps_seconds, sv_id):
    '''From sp3_data, get satellite position and velocity, with interpolation if necessary (e.g., sp3 recorded at 1Hz, rnx at 10Hz)'''
    if gps_seconds in sp3_data.keys() and sv_id in sp3_data[gps_seconds].keys():
        return sp3_data[gps_seconds][sv_id]['satellite_pos'], sp3_data[gps_seconds][sv_id]['satellite_vel'], sp3_data[gps_seconds][sv_id]['clock_bias']
    elif gps_seconds not in sp3_data.keys():
        # print('Warning!!! Time not found in SP3 data: ', gps_seconds)
        '''Interpolation'''
        # Find the closest time before and after the current time
        prev_time = max([key for key in sp3_data.keys() if key < gps_seconds and sv_id in sp3_data[key]], default=None)
        next_time = min([key for key in sp3_data.keys() if key > gps_seconds and sv_id in sp3_data[key]], default=None)
        if prev_time is None or next_time is None:
            return None, None, None
        prev_pos, next_pos = sp3_data[prev_time][sv_id]['satellite_pos'], sp3_data[next_time][sv_id]['satellite_pos']
        prev_vel, next_vel = sp3_data[prev_time][sv_id]['satellite_vel'], sp3_data[next_time][sv_id]['satellite_vel']
        prev_cb, next_cb = sp3_data[prev_time][sv_id]['clock_bias'], sp3_data[next_time][sv_id]['clock_bias']
        # Interpolate the satellite position and velocity
        # print(prev_pos, next_pos, prev_time, next_time, gps_seconds)
        # print(prev_vel, next_vel, prev_time, next_time, gps_seconds)
        # if prev_pos is None or next_pos is None or prev_vel is None or next_vel is None:
        #     return None, None, None
        if prev_vel is None: prev_vel = [0, 0, 0]
        if next_vel is None: next_vel = [0, 0, 0] # for static PLs whose vel might not be recorded in sp3
            
        sat_pos = [prev_pos[i] + (next_pos[i] - prev_pos[i]) * (gps_seconds - prev_time) / (next_time - prev_time) for i in range(3)]
        sat_vel = [prev_vel[i] + (next_vel[i] - prev_vel[i]) * (gps_seconds - prev_time) / (next_time - prev_time) for i in range(3)]
        clock_bias = prev_cb + (next_cb - prev_cb) * (gps_seconds - prev_time) / (next_time - prev_time)
        return sat_pos, sat_vel, clock_bias
    return None, None, None



def parse_rinex_observation_data(rinex_file_path):
    '''Given the rinex_file_path, extract observation data, and add info to obs_dir'''
    gps_seconds = None

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
                sat_pos, sat_vel, clock_bias = get_sat_posvel(gps_seconds, sv_id)
                obs_dir[gps_seconds].append(Obs(gps_seconds, satellite_id, sv_id, pseudorange, carrier_phase, doppler_shift, satellite_pos=sat_pos, satellite_vel=sat_vel, clock_bias=clock_bias))
                # try:
                # except ValueError:
                #     continue  # Skip lines that cannot be converted
            # print(len(obs_dir[gps_seconds]))
            # for obs in obs_dir[gps_seconds]:
            #     print(obs.gps_time, obs.id, obs.sv_id, obs.pseudorange, obs.carrier_phase, obs.doppler_shift)
            # if i > 500:
            #     # exit()
            #     break

def get_sat_posvel_from_sp3(sp3_file_path):
    '''Given the sp3_file_path, get satellite pos and vel, and add to existing observation in obs_dir'''
    gps_seconds = None

    with open(sp3_file_path, 'r') as file:
        for line in file:
            if line.startswith('*'):
                year, month, day, hour, minute, second = map(float, line.split()[1:7])
                year, month, day, hour, minute = map(int, [year, month, day, hour, minute])
                gps_time = datetime(year, month, day, hour, minute, int(second)) + timedelta(seconds=second % 1)
                gps_seconds = float((gps_time - GPS_EPOCH).total_seconds())
                if gps_seconds not in sp3_data:
                    sp3_data[gps_seconds] = {}
                continue

            if (line.startswith('P') or line.startswith('V')): # and (gps_seconds in obs_dir.keys()):
                raw_satellite_id = line[0:4].strip()
                satellite_id, sv_id = format_satellite_id(raw_satellite_id)
                if sv_id not in sp3_data[gps_seconds]:
                    sp3_data[gps_seconds][sv_id] = {'satellite_pos': None, 'satellite_vel': None, 'clock_bias': None}
                x_value = float(line[5:18].strip())
                y_value = float(line[19:32].strip())
                z_value = float(line[33:46].strip())
                if line.startswith('P'):
                    sp3_data[gps_seconds][sv_id]['satellite_pos'] = [round(x_value*1000, 3), round(y_value*1000, 3), round(z_value*1000, 3)] # km to m
                    sp3_data[gps_seconds][sv_id]['clock_bias'] = float(line[47:].strip())
                if line.startswith('V'):
                    sp3_data[gps_seconds][sv_id]['satellite_vel'] = [x_value, y_value, z_value]

def read_multipath_file(multipath_file_path):
    '''Given the multipath_file_path, extract the multipath signals, calculate necessary data, and add info to obs_mp'''
    with open(multipath_file_path, 'r') as file:
        # test = 0 
        for line in file:
            parts = line.split(',')
            gps_time = float(parts[0].strip())
            satellite_id_raw = parts[2].strip()
            satellite_id, sv_id = format_satellite_id(satellite_id_raw)

            # matching_key = next((key for key in obs_dir.keys() if isclose(gps_time, key, rel_tol=1e-9)), None)
            matching_key = min(obs_dir.keys(), key=lambda k: abs(k - gps_time))
            if matching_key is None:
                # print('Warning! At time: ', gps_time, '. Multipath does not matched with direct path...')
                continue

            ids = [obs.id for obs in obs_dir[matching_key]]
            if satellite_id not in ids:
                # print('Warning! At time: ', gps_time, '. Multipath does not matched with direct path... from sat id: ', satellite_id)
                continue

            if gps_time not in obs_mp:
                obs_mp[gps_time] = []
            if gps_time not in obs_real:
                obs_real[gps_time] = []

            #added mp_info dic
            if gps_time not in mp_info:
                mp_info[gps_time] = {}
            if satellite_id not in mp_info[gps_time]:
                mp_info[gps_time][satellite_id] = {
                    'no_los_flag': None,
                    'comps': []  # [(signal_id, path_delay, doppler_offset, power_loss), ...]
            }

            no_los_flag = (  # in this case, LOS signal is not visable
                    int(parts[5].strip()) == 0 and
                    all(float(parts[i].strip()) == -100.0 for i in range(6, 9))
            )

            mp_info[gps_time][satellite_id]['no_los_flag'] = no_los_flag

            # Multipath signals (remaining signals)
            index = ids.index(satellite_id)
            multipath_index = 9
            total_signals = int(parts[4].strip())
            obs_dir_sat = obs_dir[matching_key][index]
            for i in range(total_signals - 1):
                signal_id = int(parts[multipath_index].strip())
                path_delay = float(parts[multipath_index + 1].strip())
                doppler_offset = float(parts[multipath_index + 2].strip())
                power_loss = float(parts[multipath_index + 3].strip())
                # print(power_loss)

                mp_info[gps_time][satellite_id]['comps'].append(
                    (signal_id, path_delay, doppler_offset, power_loss)
                )
                
                if power_loss < 1000: # somehow the power_loss can be inf
                    # print("MP: ", gps_time, " Dir: ", obs_dir_sat.gps_time, " Mat: ", matching_key)
                    obs_mp[gps_time].append(Obs(gps_time, obs_dir_sat.id, obs_dir_sat.sv_id, 
                                                obs_dir_sat.pseudorange + path_delay, obs_dir_sat.carrier_phase, obs_dir_sat.doppler_shift + doppler_offset,
                                                multipath = True, multipath_id = signal_id, 
                                                satellite_pos = obs_dir_sat.satellite_pos, satellite_vel = obs_dir_sat.satellite_vel, clock_bias = obs_dir_sat.clock_bias,
                                                rec_pow=(obs_dir_sat.rec_pow - power_loss if obs_dir_sat.rec_pow is not None else None)))
                # if obs_dir_sat.rec_pow - power_loss < -1000:
                #     print(obs_dir_sat.id, obs_dir_sat.sv_id, power_loss) 
                # print("appended", gps_time, obs_dir_sat.id, obs_dir_sat.sv_id)
                multipath_index += 4
            
            ''' Real signals'''
            candidates = [o for o in obs_mp[gps_time] if o.sv_id == obs_dir_sat.sv_id] 
            if no_los_flag and len(candidates) != 0:
                best = max(candidates, key=lambda x: x.rec_pow)
                obs_real[gps_time].append(best)
            else:
                obs_real[gps_time].append(obs_dir_sat)
            # if not no_los_flag: # No LOS signal, add the real observation
            # else: # No LOS signal, select the multipath signal with max rec_pow
            #     if len(candidates) != 0:

            # test += 1
            # if test > 100:
            #     exit()


def read_rec_power(rec_pow_file_path):
    with open(rec_pow_file_path, 'r') as file:
        for line in file:
            parts = line.split(',')
            gps_time = float(parts[0].strip())
            satellite_num = int(parts[1].strip())

            matching_key = next((key for key in obs_dir.keys() if isclose(gps_time, key, rel_tol=1e-9)), None)
            if matching_key is None:
                # print('Warning! Received power does not matched with direct path... from time: ', gps_time)
                continue

            for i in range(satellite_num):
                satellite_id = int(parts[2+2*i].strip())
                if "GAL" in rec_pow_file_path:
                    satellite_id += 30
                ids = [obs.sv_id for obs in obs_dir[matching_key]]
                if satellite_id not in ids:
                    # print('Warning! At time: ', gps_time, '. Received power does not matched with direct path', ids,'... from sat id: ', satellite_id)
                    continue
                index = ids.index(satellite_id)
                
                obs_dir[matching_key][index].update_signal_power(float(parts[2+2*i+1].strip())) 


def read_obs_data(rinex_file_path, sp3_file_path, rec_pow_file_path=None, multipath_file_path=None):
    global sp3_data, obs_dir, obs_mp, obs_all, obs_real, mp_info
    sp3_data, obs_dir, obs_mp, obs_all, obs_real, mp_info = {}, {}, {}, {}, {}, {}

    # Get satellite position and velocity data from SP3 file
    for file in sp3_file_path:
        get_sat_posvel_from_sp3(file)

    # Parse RINEX file to extract all observation data
    parse_rinex_observation_data(rinex_file_path)

    # Get recevied power
    if rec_pow_file_path: # rec_pow_file_path and read_rec_power(rec_pow_file_path)
        for file in rec_pow_file_path:
            read_rec_power(file)

    # Get matipaths
    if multipath_file_path:
        read_multipath_file(multipath_file_path)

    # Combine obs_dir and obs_mp
    obs_all = copy.deepcopy(obs_dir)
    for key, value in obs_mp.items():
        if key in obs_dir: # Append the list from obs_mp to obs_dir's list            
            obs_all[key].extend(value)
        else: # If key is not in obs_dir, add it directly            
            obs_all[key] = value
    
    return obs_dir, obs_mp, obs_all, obs_real


def read_real_rover_obs(csv_file_path, TX_POS=None):
    rover_obs = pd.read_csv(csv_file_path, lineterminator='\n') 
    rover_obs.columns = rover_obs.columns.str.strip()
    # print(rover_obs.columns.tolist())
    rover_obs_real = {}
    for _, row in rover_obs.iterrows():
        if (('G' in row['id']) or ('E' in row['id']) or ('C' in row['id'])) and (row['pos_x'] == 0 and row['pos_y'] == 0 and row['pos_z'] == 0):
            continue
        # if (row['pos_x'] == 0 and row['pos_y'] == 0 and row['pos_z'] == 0) or \
        #    (row['vel_x'] == 0 and row['vel_y'] == 0 and row['vel_z'] == 0):
        #     continue

        gps_time = float(row['gps_time']) #round(float(row['gps_time']), 1) #float(row['gps_time']) Round to 1 decimal place for consistency
        id = str(row['id'])
        sv_id = int(row['sv_id'])
        pseudorange = float(row['pseudorange']) # m
        carrier_phase = float(row['carrier_phase']) # cycles
        doppler_shift = float(row['doppler_shift']) # Hz
        try:
            CN0 = float(row['CN0']) # dB-Hz
        except:
            CN0 = float(row['CNo']) # dB-Hz
        if (('L' in id) or ('P' in id)) and (TX_POS is not None):
            satellite_pos = TX_POS[id]
        else:
            satellite_pos = [float(row['pos_x']), float(row['pos_y']), float(row['pos_z'])]
        satellite_vel = [float(row['vel_x']), float(row['vel_y']), float(row['vel_z'])]
        clock_bias = float(row['clkErr']*1e6) # seconds -> microseconds 

        rover_obs = Obs(gps_time, id, sv_id, pseudorange, carrier_phase, doppler_shift, satellite_pos=satellite_pos, satellite_vel=satellite_vel, CN0=CN0, clock_bias=clock_bias)
        if gps_time not in rover_obs_real:
            rover_obs_real[gps_time] = []
        rover_obs_real[gps_time].append(rover_obs)

    return rover_obs_real


if __name__ == "__main__":
    # Define file paths
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = 'Results0219/Munich10min100m' #'Results0219/Munich10min100m' # 'Circle_radius 100', Eipcycloid, data
    folder_path = os.path.join(dir_path, data)
    # NOTE: simulator old version has .24o file, new version has .rnx file 
    # rinex_file_path = os.path.join(folder_path, find_file(folder_path, 'v', '.24o')[0]) # 's101325q.24o'
    rinex_file_path = os.path.join(folder_path, find_file(folder_path, 'V', '.rnx')[0]) # 's101325q.24o'
    # sp3_file_path = os.path.join(folder_path, find_file(folder_path, 'GPS', '.SP3')[0])
    sp3_file_path = [os.path.join(folder_path, find_file(folder_path, 'GPS', '.SP3')[0]), os.path.join(folder_path, find_file(folder_path, 'GAL', '.SP3')[0])]
    multipath_file_path = os.path.join(folder_path, 'Multipath_U100.log')
    rec_pow_file_path = [os.path.join(folder_path, 'ReceivedPower_101_GPS_L1.log'), os.path.join(folder_path, 'ReceivedPower_101_GALILEO_E1.log')]
    # obs_dir, obs_mp, obs_all = read_obs_data(rinex_file_path, sp3_file_path)
    obs_dir, obs_mp, obs_all, obs_real = read_obs_data(rinex_file_path, sp3_file_path, rec_pow_file_path = rec_pow_file_path, multipath_file_path=multipath_file_path)

    # print(obs_dir.keys)
    # for key in obs_dir.keys():
    #     print(key)
    print('Total obs_dir: ', sum(len(v) for v in obs_dir.values()), 
          ', Total obs_mp: ', sum(len(v) for v in obs_mp.values()), 
          ', Total obs_all: ', sum(len(v) for v in obs_all.values()), 
          ', Total obs_real: ', sum(len(v) for v in obs_real.values()))
    
    # print('Total dir in real obs: ', sum(len(v) for v in obs_real.multipath()))
    total_true = sum(v.multipath for v_list in obs_real.values() for v in v_list)
    total_false = sum(not v.multipath for v_list in obs_real.values() for v in v_list)
    print(f"Real observations with multipath = True: {total_true}")
    print(f"Real observations with multipath = False: {total_false}")


    for i, values in enumerate(obs_real.values()):
        # print(values)
        # print('-------------------------------------------')
        
        for j, value in enumerate(values):
            print(value.multipath, value.id, value.sv_id, value.satellite_pos, value.satellite_vel)
            # if 'PL' in value.id:
            #     print(value.multipath, value.gps_time, value.id, value.sv_id, value.satellite_pos, value.rec_pow, value.CN0) #value.satellite_pos, value.satellite_vel)
        # print('-------------------------------------------')
        if i > 50:
            exit()

    # for key in obs_mp.keys():
    #     print(key)

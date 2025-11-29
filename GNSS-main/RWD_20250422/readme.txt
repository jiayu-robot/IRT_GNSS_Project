ts_rosMsg
- imu in "imu_data"
- ground truth in "novatel_oem7_bestpos" and "novatel_oem7_bestcel"

ts_sdrObs has ts_baseMsg and ts_roverObs

ts_baseMsg provides the observation received by the static base station
ts_roverObs provides the observation received by the rover

The contents/symbols in them are:

Constellation:
    G: GPS
    E: Galileo
    C: Beidou
    LTE
    PL: Pseudolite

Within each entry for a given constellation, the following observables are stored for each satellite (or base station in the case of LTE):
    C: Pseudorange in [m]
    L: Carrier phase in [cycles]
    D: Doppler in [Hz]
    C/No: Carrier-to-noise ratio in [dB-Hz]
    pos: Transmitter position in ECEF
    clkErr: Transmitter clock offset in [s]

For Day1 (22.04.2025), the transmitter and base station locations are:
    [x,y,z] = func_LLH2ECEF(deg2rad(48.0779195), deg2rad(11.6305769), 549.51+45.5);
    [LTE_tx1_x,LTE_tx1_y,LTE_tx1_z] = func_LLH2ECEF(deg2rad(48.0780113), deg2rad(11.6307491), 549.40+45.5);
    [LTE_tx2_x,LTE_tx2_y,LTE_tx2_z] = func_LLH2ECEF(deg2rad(48.0778041), deg2rad(11.6306888), 549.38+45.5);
    [PL_tx1_x,PL_tx1_y,PL_tx1_z] = func_LLH2ECEF(deg2rad(48.0779030), deg2rad(11.6309295), 551.38+45.5);
    [PL_tx2_x,PL_tx2_y,PL_tx2_z] = func_LLH2ECEF(deg2rad(48.0781220), deg2rad(11.6306068), 551.25+45.5);

For Day2 (23.04.2025), the transmitter and base station locations are:
    [x,y,z] = func_LLH2ECEF(deg2rad(48.0779172), deg2rad(11.6305752), 549.57+45.5); % Base station
    [LTE_tx1_x,LTE_tx1_y,LTE_tx1_z] = func_LLH2ECEF(deg2rad(48.0780048), deg2rad(11.6307446), 549.40+45.5);
    [LTE_tx2_x,LTE_tx2_y,LTE_tx2_z] = func_LLH2ECEF(deg2rad(48.0777902), deg2rad(11.6306752), 549.42+45.5);
    [PL_tx1_x,PL_tx1_y,PL_tx1_z] = func_LLH2ECEF(deg2rad(48.0778883), deg2rad(11.6309177), 551.56+45.5);
    [PL_tx2_x,PL_tx2_y,PL_tx2_z] = func_LLH2ECEF(deg2rad(48.0781255), deg2rad(11.6306180), 551.40+45.5);


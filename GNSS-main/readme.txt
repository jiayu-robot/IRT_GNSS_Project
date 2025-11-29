RWD20250422 stores the real world data (it contains GPS, Gal, Bds, PL, LTE)
    Constellation:
    G: GPS
    E: Galileo
    C: Beidou
    LTE
    PL: Pseudolite

for_real stores the simulated data that was designed for matching real world data

main_plot.ipynb reads simulated data, preprocess them, check the outputs, and save the necessary data in csv. 

read_gnss.ipynb reads the real-world data, preprocess them, check the outputs, and save the necessary data in csv.

learning folder
    - main.ipynb, read the real world and simulated data, use SVM for training and testing (note the balance and imbalance data)
    - main2.ipynb, read the real world and simulated data, use LSTM for training and testing
    - Tested with real world data. Save the result in csv.
    
Since real world data does not have ground truth, we can verify the performance by positioning trajectory accuracy with signaled excluding detected NLOS data.

real_pos.ipynb reads the real world data with detection label, and check their positioning accuarcy.




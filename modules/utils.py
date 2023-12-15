'''
Author: Kartik Tiwari
Date: 2023-12-06 
Summary: This file contains the utilities developed for analysing MCA data to measure 
the lifetime of an exctied state of Cs-133 using the fast-slow coincidence setup
'''

# importing libraries
import numpy as np
import matplotlib.pyplot as plt

'''
Function to load the data from the .csv file
input: file_name of the .csv file
output: Channel_No, Read_Counts
'''
def get_data(file_name):
    # appending string with extension
    a = file_name + '.csv'
    path = 'data/' + a
    
    # loading data from csv file delimited by tab
    data = np.loadtxt(path, delimiter=',', skiprows=13)

    Channel_No = data.transpose()[0]
    Read_Counts = data.transpose()[2]
    
    return Channel_No, Read_Counts

'''
Function to pad the measured data with zeros to turn it into a consistent 8192 channel array
input: file_name of the .csv file
output: all_channels, padded_counts
'''
def pad_counts(file_name):
    all_channels = np.arange(1, 8192, 1)
    padded_counts = np.zeros(8191)

    channels, counts = get_data(file_name)

    for i in range(len(channels)):
        index = int(channels[i])
        padded_counts[index] = counts[i]
        
    return all_channels, padded_counts

'''
Function to perform two-point calibration of the MCA channels (linear, with channel 0 corresponding to 0 keV)
input: file_name of the .csv file, peak_e_val - peak energy value in keV
output: calibrated_channels
'''
def calibrate_channels(file_name, peak_e_val):
    
    channels, counts = pad_counts(file_name)
    peak = counts.argmax()
    peak_channel = channels[peak]

    val1, val2 = 0, peak_e_val
    max1, max2 = 0, peak_channel

    calibrated_channels = val1 + (channels - max1) / (max2 - max1) * (val2 - val1)
    
    return calibrated_channels

'''
Function to plot the MCA data
input: file_name of the .csv file
output: plot of the data
optional: calibration (boolean) - if true, calibrates the data using the peak_e_val
'''
def plot_MCA(file_name, calibration=False, peak_e_val=0):
    
    Channel_No, Read_Counts = pad_counts(file_name)

    if calibration:
        Channel_No = calibrate_channels(file_name, peak_e_val)
        plt.xlabel('Energy (keV)')
        file_name = 'calibrated/' + file_name + '_calibrated'

    else:
        plt.xlabel('Channel Number')
        file_name = 'uncalibrated/' + file_name + '_uncalibrated'
        
    plt.scatter(Channel_No, Read_Counts, marker='.', s=4, color='black', label='Data')
    plt.ylabel('Counts')
    plt.grid(alpha=0.5)
    plt.savefig('fig/'+ file_name, dpi=300)

'''
Optimization and calibration related functions
'''
from scipy.optimize import curve_fit

# Fitting a gaussian on spectrum peaks
def gaussian(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

# fitting a linear function to the channel numbers
def linear(x, m, b):
    return m*x + b

# exponential function
def exponential(t, tau, A):
    return A*np.exp(-t/tau)
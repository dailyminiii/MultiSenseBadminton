############
#
# Copyright (c) 2022 MIT CSAIL and Joseph DelPreto
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# See https://action-net.csail.mit.edu for more usage information.
# Created 2021-2022 for the MIT ActionNet project by Joseph DelPreto [https://josephdelpreto.com].
#
############

import h5py
import numpy as np
from scipy import interpolate # for the resampling example

# NOTE: HDFView is a helpful program for exploring HDF5 contents.
#   The official download page is at https://www.hdfgroup.org/downloads/hdfview.
#   It can also be downloaded without an account from https://www.softpedia.com/get/Others/Miscellaneous/HDFView.shtml.

# Specify the downloaded file to parse.
filepath = './data_archive/S28/2023-02-15_14-36-53_badminton-wearables_S28/2023-02-15_14-37-22_streamLog_badminton-wearables_S28.hdf5'

# Open the file.
h5_file = h5py.File(filepath, 'r')

####################################################
# Example of reading sensor data: read gForce Lower EMG data.
####################################################
print()
print('='*65)
print('Extracting gForce Lower EMG data from the HDF5 file')
print('='*65)

device_name = 'armband-gforce-lowerarm'
stream_name = 'pressure_values_N_cm2'
# Get the data as an Nx8 matrix where each row is a timestamp and each column is an EMG channel.
gforce_lower_emg_data = h5_file[device_name][stream_name]['data']
gforce_lower_emg_data = np.array(gforce_lower_emg_data)
# Get the timestamps for each row as seconds since epoch.
gforce_lower_emg_time_s = h5_file[device_name][stream_name]['time_s']
gforce_lower_emg_time_s = np.squeeze(np.array(gforce_lower_emg_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# Get the timestamps for each row as human-readable strings.
gforce_lower_emg_time_str = h5_file[device_name][stream_name]['time_str']
gforce_lower_emg_time_str = np.squeeze(np.array(gforce_lower_emg_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list

print('gForce Lower EMG Data:')
# print(' Shape', gforce_lower_emg_data.shape)
# print(' Preview:')
# print(gforce_lower_emg_data)
# print()
# print('gForce Lower EMG Timestamps')
# print(' Shape', gforce_lower_emg_time_s.shape)
# print(' Preview:')
# print(gforce_lower_emg_time_s)
# print()
# print('gForce Lower EMG Timestamps as Strings')
# print(' Shape', gforce_lower_emg_time_str.shape)
# print(' Preview:')
print(gforce_lower_emg_time_str)
print()
print(' Sampling rate: %0.2f Hz' % ((gforce_lower_emg_data.shape[0]-1)/(max(gforce_lower_emg_time_s) - min(gforce_lower_emg_time_s))))
print()

####################################################
# Example of reading sensor data: read gForce Upper EMG data.
####################################################
print()
print('='*65)
print('Extracting gForce Upper EMG data from the HDF5 file')
print('='*65)

device_name = 'armband-gforce-upperarm'
stream_name = 'pressure_values_N_cm2'
# Get the data as an Nx8 matrix where each row is a timestamp and each column is an EMG channel.
gforce_upper_emg_data = h5_file[device_name][stream_name]['data']
gforce_upper_emg_data = np.array(gforce_upper_emg_data)
# Get the timestamps for each row as seconds since epoch.
gforce_upper_emg_time_s = h5_file[device_name][stream_name]['time_s']
gforce_upper_emg_time_s = np.squeeze(np.array(gforce_upper_emg_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# Get the timestamps for each row as human-readable strings.
gforce_upper_emg_time_str = h5_file[device_name][stream_name]['time_str']
gforce_upper_emg_time_str = np.squeeze(np.array(gforce_upper_emg_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list

print('gForce Upper EMG Data:')
# print(' Shape', gforce_upper_emg_data.shape)
# print(' Preview:')
# print(gforce_upper_emg_data)
# print()
# print('gForce Upper EMG Timestamps')
# print(' Shape', gforce_upper_emg_time_s.shape)
# print(' Preview:')
# print(gforce_upper_emg_time_s)
# print()
# print('gForce Upper EMG Timestamps as Strings')
# print(' Shape', gforce_upper_emg_time_str.shape)
# print(' Preview:')
print(gforce_upper_emg_time_str)
print()
print(' Sampling rate: %0.2f Hz' % ((gforce_upper_emg_data.shape[0]-1)/(max(gforce_upper_emg_time_s) - min(gforce_upper_emg_time_s))))
print()

####################################################
# Example of reading sensor data: read Cognionics EMG data.
####################################################
print()
print('='*65)
print('Extracting Cognionics EMG data from the HDF5 file')
print('='*65)

device_name = 'EMG-DominantLeg-cognionics'
stream_name = 'emg-values'
# Get the data as an Nx4 matrix where each row is a timestamp and each column is an EMG channel.
cognionics_emg_data = h5_file[device_name][stream_name]['data']
cognionics_emg_data = np.array(cognionics_emg_data)
# Get the timestamps for each row as seconds since epoch.
cognionics_emg_time_s = h5_file[device_name][stream_name]['time_s']
cognionics_emg_time_s = np.squeeze(np.array(cognionics_emg_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# Get the timestamps for each row as human-readable strings.
cognionics_emg_time_str = h5_file[device_name][stream_name]['time_str']
cognionics_emg_time_str = np.squeeze(np.array(cognionics_emg_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list

print('Cognionics EMG Data:')
# print(' Shape', cognionics_emg_data.shape)
# print(' Preview:')
# print(cognionics_emg_data)
# print()
# print('Cognionics EMG Timestamps')
# print(' Shape', cognionics_emg_time_s.shape)
# print(' Preview:')
# print(cognionics_emg_time_s)
# print()
# print('Cognionics EMG Timestamps as Strings')
# print(' Shape', cognionics_emg_time_str.shape)
# print(' Preview:')
print(cognionics_emg_time_str)
print()
print(' Sampling rate: %0.2f Hz' % ((cognionics_emg_data.shape[0]-1)/(max(cognionics_emg_time_s) - min(cognionics_emg_time_s))))
print()

# ####################################################
# Example of reading sensor data: read Pupil Gaze data.
####################################################
print()
print('='*65)
print('Extracting Pupil Gaze data from the HDF5 file')
print('='*65)

device_name = 'eye-gaze'
stream_name = 'gaze'
# Get the data as an Nx4 matrix where each row is a timestamp and each column is an EMG channel.
pupil_gaze_data = h5_file[device_name][stream_name]['data']
pupil_gaze_data = np.array(pupil_gaze_data)
# Get the timestamps for each row as seconds since epoch.
pupil_gaze_time_s = h5_file[device_name][stream_name]['time_s']
pupil_gaze_time_s = np.squeeze(np.array(pupil_gaze_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# Get the timestamps for each row as human-readable strings.
pupil_gaze_time_str = h5_file[device_name][stream_name]['time_str']
pupil_gaze_time_str = np.squeeze(np.array(pupil_gaze_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list

print('Pupil Gaze Data:')
# print(' Shape', pupil_gaze_data.shape)
# print(' Preview:')
# print(pupil_gaze_data)
# print()
# print('Pupil Gaze Timestamps')
# print(' Shape', pupil_gaze_time_s.shape)
# print(' Preview:')
# print(pupil_gaze_time_s)
# print()
# print('Pupil Gaze Timestamps as Strings')
# print(' Shape', pupil_gaze_time_str.shape)
# print(' Preview:')
print(pupil_gaze_time_str)
print()
print(' Sampling rate: %0.2f Hz' % ((pupil_gaze_data.shape[0]-1)/(max(pupil_gaze_time_s) - min(pupil_gaze_time_s))))
print()

####################################################
# Example of reading sensor data: read Moticon COP data.
####################################################
print()
print('='*65)
print('Extracting Moticon COP data from the HDF5 file')
print('='*65)

device_name = 'insole-moticon-cop'
stream_name = 'cop'
# Get the data as an Nx4 matrix where each row is a timestamp and each column is an EMG channel.
moticon_cop_data = h5_file[device_name][stream_name]['data']
moticon_cop_data = np.array(moticon_cop_data)
# Get the timestamps for each row as seconds since epoch.
moticon_cop_time_s = h5_file[device_name][stream_name]['time_s']
moticon_cop_time_s = np.squeeze(np.array(moticon_cop_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# Get the timestamps for each row as human-readable strings.
moticon_cop_time_str = h5_file[device_name][stream_name]['time_str']
moticon_cop_time_str = np.squeeze(np.array(moticon_cop_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list

print('Moticon COP Data:')
# print(' Shape', moticon_cop_data.shape)
# print(' Preview:')
# print(moticon_cop_data)
# print()
# print('Moticon COP Timestamps')
# print(' Shape', moticon_cop_time_s.shape)
# print(' Preview:')
# print(moticon_cop_time_s)
# print()
# print('Moticon COP Timestamps as Strings')
# print(' Shape', moticon_cop_time_str.shape)
# print(' Preview:')
print(moticon_cop_time_str)
print()
print(' Sampling rate: %0.2f Hz' % ((moticon_cop_data.shape[0]-1)/(max(moticon_cop_time_s) - min(moticon_cop_time_s))))
print()

####################################################
# Example of reading sensor data: read Moticon Left Acceleration data.
# ####################################################
# print()
# print('='*65)
# print('Extracting Moticon Left Acceleration data from the HDF5 file')
# print('='*65)

device_name = 'insole-moticon-left-acceleration'
stream_name = 'acceleration'
# Get the data as an Nx4 matrix where each row is a timestamp and each column is an EMG channel.
moticon_left_acc_data = h5_file[device_name][stream_name]['data']
moticon_left_acc_data = np.array(moticon_left_acc_data)
# Get the timestamps for each row as seconds since epoch.
moticon_left_acc_time_s = h5_file[device_name][stream_name]['time_s']
moticon_left_acc_time_s = np.squeeze(np.array(moticon_left_acc_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# Get the timestamps for each row as human-readable strings.
moticon_left_acc_time_str = h5_file[device_name][stream_name]['time_str']
moticon_left_acc_time_str = np.squeeze(np.array(moticon_left_acc_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list

# print('Moticon Left Acceleration Data:')
# print(' Shape', moticon_left_acc_data.shape)
# print(' Preview:')
# print(moticon_left_acc_data)
# print()
# print('Moticon Left Acceleration Timestamps')
# print(' Shape', moticon_left_acc_time_s.shape)
# print(' Preview:')
# print(moticon_left_acc_time_s)
# print()
# print('Moticon Left Acceleration Timestamps as Strings')
# print(' Shape', moticon_left_acc_time_str.shape)
# print(' Preview:')
# print(moticon_left_acc_time_str)
# print()
# print(' Sampling rate: %0.2f Hz' % ((moticon_left_acc_data.shape[0]-1)/(max(moticon_left_acc_time_s) - min(moticon_left_acc_time_s))))
# print()

####################################################
# Example of reading sensor data: read Moticon Left Angular Velocity data.
####################################################
# print()
# print('='*65)
# print('Extracting Moticon Left Acceleration data from the HDF5 file')
# print('='*65)

device_name = 'insole-moticon-left-angular'
stream_name = 'angular'
# Get the data as an Nx4 matrix where each row is a timestamp and each column is an EMG channel.
moticon_left_angular_data = h5_file[device_name][stream_name]['data']
moticon_left_angular_data = np.array(moticon_left_angular_data)
# Get the timestamps for each row as seconds since epoch.
moticon_left_angular_time_s = h5_file[device_name][stream_name]['time_s']
moticon_left_angular_time_s = np.squeeze(np.array(moticon_left_angular_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# Get the timestamps for each row as human-readable strings.
moticon_left_angular_time_str = h5_file[device_name][stream_name]['time_str']
moticon_left_angular_time_str = np.squeeze(np.array(moticon_left_angular_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list

# print('Moticon Left Angular Velocity Data:')
# print(' Shape', moticon_left_angular_data.shape)
# print(' Preview:')
# print(moticon_left_angular_data)
# print()
# print('Moticon Left Angular Velocity Timestamps')
# print(' Shape', moticon_left_angular_time_s.shape)
# print(' Preview:')
# print(moticon_left_angular_time_s)
# print()
# print('Moticon Left Angular Velocity Timestamps as Strings')
# print(' Shape', moticon_left_angular_time_str.shape)
# print(' Preview:')
# print(moticon_left_angular_time_str)
# print()
# print(' Sampling rate: %0.2f Hz' % ((moticon_left_angular_data.shape[0]-1)/(max(moticon_left_angular_time_s) - min(moticon_left_angular_time_s))))
# print()

####################################################
# Example of reading sensor data: read Moticon Left Pressure data.
####################################################
# print()
# print('='*65)
# print('Extracting Moticon Left Pressure data from the HDF5 file')
# print('='*65)

device_name = 'insole-moticon-left-pressure'
stream_name = 'pressure_values_N_cm2'
# Get the data as an Nx4 matrix where each row is a timestamp and each column is an EMG channel.
moticon_left_pressure_data = h5_file[device_name][stream_name]['data']
moticon_left_pressure_data = np.array(moticon_left_pressure_data)
# Get the timestamps for each row as seconds since epoch.
moticon_left_pressure_time_s = h5_file[device_name][stream_name]['time_s']
moticon_left_pressure_time_s = np.squeeze(np.array(moticon_left_pressure_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# Get the timestamps for each row as human-readable strings.
moticon_left_pressure_time_str = h5_file[device_name][stream_name]['time_str']
moticon_left_pressure_time_str = np.squeeze(np.array(moticon_left_pressure_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list

# print('Moticon Left Pressure Data:')
# print(' Shape', moticon_left_pressure_data.shape)
# print(' Preview:')
# print(moticon_left_pressure_data)
# print()
# print('Moticon Left Pressure Timestamps')
# print(' Shape', moticon_left_pressure_time_s.shape)
# print(' Preview:')
# print(moticon_left_pressure_time_s)
# print()
# print('Moticon Left Pressure Timestamps as Strings')
# print(' Shape', moticon_left_pressure_time_str.shape)
# print(' Preview:')
# print(moticon_left_pressure_time_str)
# print()
# print(' Sampling rate: %0.2f Hz' % ((moticon_left_pressure_data.shape[0]-1)/(max(moticon_left_pressure_time_s) - min(moticon_left_pressure_time_s))))
# print()

####################################################
# Example of reading sensor data: read Moticon Left Total Force data.
####################################################
# print()
# print('='*65)
# print('Extracting Moticon Left Total Force data from the HDF5 file')
# print('='*65)

device_name = 'insole-moticon-left-totalForce'
stream_name = 'totalForce'
# Get the data as an Nx4 matrix where each row is a timestamp and each column is an EMG channel.
moticon_left_totalForce_data = h5_file[device_name][stream_name]['data']
moticon_left_totalForce_data = np.array(moticon_left_totalForce_data)
# Get the timestamps for each row as seconds since epoch.
moticon_left_totalForce_time_s = h5_file[device_name][stream_name]['time_s']
moticon_left_totalForce_time_s = np.squeeze(np.array(moticon_left_totalForce_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# Get the timestamps for each row as human-readable strings.
moticon_left_totalForce_time_str = h5_file[device_name][stream_name]['time_str']
moticon_left_totalForce_time_str = np.squeeze(np.array(moticon_left_totalForce_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list

# print('Moticon Left Total Force Data:')
# print(' Shape', moticon_left_totalForce_data.shape)
# print(' Preview:')
# print(moticon_left_totalForce_data)
# print()
# print('Moticon Left Total Force Timestamps')
# print(' Shape', moticon_left_totalForce_time_s.shape)
# print(' Preview:')
# print(moticon_left_totalForce_time_s)
# print()
# print('Moticon Left Total Force Timestamps as Strings')
# print(' Shape', moticon_left_totalForce_time_str.shape)
# print(' Preview:')
# print(moticon_left_totalForce_time_str)
# print()
# print(' Sampling rate: %0.2f Hz' % ((moticon_left_totalForce_data.shape[0]-1)/(max(moticon_left_totalForce_time_s) - min(moticon_left_totalForce_time_s))))
# print()

# ####################################################
# # Example of reading sensor data: read Moticon Right Acceleration data.
# ####################################################
# print()
# print('='*65)
# print('Extracting Moticon Right Acceleration data from the HDF5 file')
# print('='*65)
#
# device_name = 'insole-moticon-right-acceleration'
# stream_name = 'acceleration'
# # Get the data as an Nx4 matrix where each row is a timestamp and each column is an EMG channel.
# moticon_right_acc_data = h5_file[device_name][stream_name]['data']
# moticon_right_acc_data = np.array(moticon_right_acc_data)
# # Get the timestamps for each row as seconds since epoch.
# moticon_right_acc_time_s = h5_file[device_name][stream_name]['time_s']
# moticon_right_acc_time_s = np.squeeze(np.array(moticon_right_acc_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# # Get the timestamps for each row as human-readable strings.
# moticon_right_acc_time_str = h5_file[device_name][stream_name]['time_str']
# moticon_right_acc_time_str = np.squeeze(np.array(moticon_right_acc_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list
#
# print('Moticon Right Acceleration Data:')
# print(' Shape', moticon_right_acc_data.shape)
# print(' Preview:')
# print(moticon_right_acc_data)
# print()
# print('Moticon Right Acceleration Timestamps')
# print(' Shape', moticon_right_acc_time_s.shape)
# print(' Preview:')
# print(moticon_right_acc_time_s)
# print()
# print('Moticon Right Acceleration Timestamps as Strings')
# print(' Shape', moticon_right_acc_time_str.shape)
# print(' Preview:')
# print(moticon_right_acc_time_str)
# print()
# print(' Sampling rate: %0.2f Hz' % ((moticon_right_acc_data.shape[0]-1)/(max(moticon_right_acc_time_s) - min(moticon_right_acc_time_s))))
# print()

####################################################
# Example of reading sensor data: read Moticon Right Angular Velocity data.
####################################################
# print()
# print('='*65)
# print('Extracting Moticon Right Acceleration data from the HDF5 file')
# print('='*65)

device_name = 'insole-moticon-right-angular'
stream_name = 'angular'
# Get the data as an Nx4 matrix where each row is a timestamp and each column is an EMG channel.
moticon_right_angular_data = h5_file[device_name][stream_name]['data']
moticon_right_angular_data = np.array(moticon_right_angular_data)
# Get the timestamps for each row as seconds since epoch.
moticon_right_angular_time_s = h5_file[device_name][stream_name]['time_s']
moticon_right_angular_time_s = np.squeeze(np.array(moticon_right_angular_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# Get the timestamps for each row as human-readable strings.
moticon_right_angular_time_str = h5_file[device_name][stream_name]['time_str']
moticon_right_angular_time_str = np.squeeze(np.array(moticon_right_angular_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list

# print('Moticon Right Angular Velocity Data:')
# print(' Shape', moticon_right_angular_data.shape)
# print(' Preview:')
# print(moticon_right_angular_data)
# print()
# print('Moticon Right Angular Velocity Timestamps')
# print(' Shape', moticon_right_angular_time_s.shape)
# print(' Preview:')
# print(moticon_right_angular_time_s)
# print()
# print('Moticon Right Angular Velocity Timestamps as Strings')
# print(' Shape', moticon_right_angular_time_str.shape)
# print(' Preview:')
# print(moticon_right_angular_time_str)
# print()
# print(' Sampling rate: %0.2f Hz' % ((moticon_right_angular_data.shape[0]-1)/(max(moticon_right_angular_time_s) - min(moticon_right_angular_time_s))))
# print()

####################################################
# Example of reading sensor data: read Moticon Right Pressure data.
####################################################
# print()
# print('='*65)
# print('Extracting Moticon Right Pressure data from the HDF5 file')
# print('='*65)

device_name = 'insole-moticon-right-pressure'
stream_name = 'pressure_values_N_cm2'
# Get the data as an Nx4 matrix where each row is a timestamp and each column is an EMG channel.
moticon_right_pressure_data = h5_file[device_name][stream_name]['data']
moticon_right_pressure_data = np.array(moticon_right_pressure_data)
# Get the timestamps for each row as seconds since epoch.
moticon_right_pressure_time_s = h5_file[device_name][stream_name]['time_s']
moticon_right_pressure_time_s = np.squeeze(np.array(moticon_right_pressure_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# Get the timestamps for each row as human-readable strings.
moticon_right_pressure_time_str = h5_file[device_name][stream_name]['time_str']
moticon_right_pressure_time_str = np.squeeze(np.array(moticon_right_pressure_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list
#
# print('Moticon Right Pressure Data:')
# print(' Shape', moticon_right_pressure_data.shape)
# print(' Preview:')
# print(moticon_right_pressure_data)
# print()
# print('Moticon Right Pressure Timestamps')
# print(' Shape', moticon_right_pressure_time_s.shape)
# print(' Preview:')
# print(moticon_right_pressure_time_s)
# print()
# print('Moticon Right Pressure Timestamps as Strings')
# print(' Shape', moticon_right_pressure_time_str.shape)
# print(' Preview:')
# print(moticon_right_pressure_time_str)
# print()
# print(' Sampling rate: %0.2f Hz' % ((moticon_right_pressure_data.shape[0]-1)/(max(moticon_right_pressure_time_s) - min(moticon_right_pressure_time_s))))
# print()

####################################################
# Example of reading sensor data: read Moticon Right Total Force data.
####################################################
# print()
# print('='*65)
# print('Extracting Moticon Left Total Force data from the HDF5 file')
# print('='*65)

device_name = 'insole-moticon-right-totalForce'
stream_name = 'totalForce'
# Get the data as an Nx4 matrix where each row is a timestamp and each column is an EMG channel.
moticon_right_totalForce_data = h5_file[device_name][stream_name]['data']
moticon_right_totalForce_data = np.array(moticon_right_totalForce_data)
# Get the timestamps for each row as seconds since epoch.
moticon_right_totalForce_time_s = h5_file[device_name][stream_name]['time_s']
moticon_right_totalForce_time_s = np.squeeze(np.array(moticon_right_totalForce_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# Get the timestamps for each row as human-readable strings.
moticon_right_totalForce_time_str = h5_file[device_name][stream_name]['time_str']
moticon_right_totalForce_time_str = np.squeeze(np.array(moticon_right_totalForce_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list

# print('Moticon Right Total Force Data:')
# print(' Shape', moticon_right_totalForce_data.shape)
# print(' Preview:')
# print(moticon_right_totalForce_data)
# print()
# print('Moticon Right Total Force Timestamps')
# print(' Shape', moticon_right_totalForce_time_s.shape)
# print(' Preview:')
# print(moticon_right_totalForce_time_s)
# print()
# print('Moticon Right Total Force Timestamps as Strings')
# print(' Shape', moticon_right_totalForce_time_str.shape)
# print(' Preview:')
# print(moticon_right_totalForce_time_str)
# print()
# print(' Sampling rate: %0.2f Hz' % ((moticon_right_totalForce_data.shape[0]-1)/(max(moticon_right_totalForce_time_s) - min(moticon_right_totalForce_time_s))))
# print()

####################################################
# Example of reading sensor data: read Perception Neuron Studio Joint Angular Velocity data.
####################################################
print()
print('='*65)
print('Extracting Perception Neuron Studio Joint Angular Velocity data from the HDF5 file')
print('='*65)

device_name = 'pns-joint-angular-velocity'
stream_name = 'velocity-values'
# Get the data as an Nx4 matrix where each row is a timestamp and each column is an EMG channel.
pns_angular_velocity_data = h5_file[device_name][stream_name]['data']
pns_angular_velocity_data = np.array(pns_angular_velocity_data)
# Get the timestamps for each row as seconds since epoch.
pns_angular_velocity_time_s = h5_file[device_name][stream_name]['time_s']
pns_angular_velocity_time_s = np.squeeze(np.array(pns_angular_velocity_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# Get the timestamps for each row as human-readable strings.
pns_angular_velocity_time_str = h5_file[device_name][stream_name]['time_str']
pns_angular_velocity_time_str = np.squeeze(np.array(pns_angular_velocity_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list

print('Perception Neuron Studio Joint Angular Velocity Data:')
# print(' Shape', pns_angular_velocity_data.shape)
# print(' Preview:')
# print(pns_angular_velocity_data)
# print()
# print('Moticon Perception Neuron Studio Joint Angular Velocity Timestamps')
# print(' Shape', pns_angular_velocity_time_s.shape)
# print(' Preview:')
# print(pns_angular_velocity_time_s)
# print()
# print('Moticon Perception Neuron Studio Joint Angular Velocity Timestamps as Strings')
# print(' Shape', pns_angular_velocity_time_str.shape)
# print(' Preview:')
print(pns_angular_velocity_time_str)
print()
print(' Sampling rate: %0.2f Hz' % ((pns_angular_velocity_data.shape[0]-1)/(max(pns_angular_velocity_time_s) - min(pns_angular_velocity_time_s))))
print()

####################################################
# Example of reading sensor data: read Perception Neuron Studio Joint Euler Angle data.
####################################################
# print()
# print('='*65)
# print('Extracting Perception Neuron Studio Joint Euler Angle data from the HDF5 file')
# print('='*65)

device_name = 'pns-joint-euler'
stream_name = 'angle-values'
# Get the data as an Nx4 matrix where each row is a timestamp and each column is an EMG channel.
pns_euler_data = h5_file[device_name][stream_name]['data']
pns_euler_data = np.array(pns_euler_data)
# Get the timestamps for each row as seconds since epoch.
pns_euler_time_s = h5_file[device_name][stream_name]['time_s']
pns_euler_time_s = np.squeeze(np.array(pns_euler_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# Get the timestamps for each row as human-readable strings.
pns_euler_time_str = h5_file[device_name][stream_name]['time_str']
pns_euler_time_str = np.squeeze(np.array(pns_euler_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list

# print('Perception Neuron Studio Joint Euler Angle  Data:')
# print(' Shape', pns_euler_data.shape)
# print(' Preview:')
# print(pns_euler_data)
# print()
# print('Perception Neuron Studio Joint Euler Angle  Timestamps')
# print(' Shape', pns_euler_time_s.shape)
# print(' Preview:')
# print(pns_euler_time_s)
# print()
# print('Perception Neuron Studio Joint Euler Angle Timestamps as Strings')
# print(' Shape', pns_euler_time_str.shape)
# print(' Preview:')
# print(pns_euler_time_str)
# print()
# print(' Sampling rate: %0.2f Hz' % ((pns_euler_data.shape[0]-1)/(max(pns_euler_time_s) - min(pns_euler_time_s))))
# print()

####################################################
# Example of reading sensor data: read Perception Neuron Studio Joint Local Position data.
####################################################
# print()
# print('='*65)
# print('Extracting Perception Neuron Studio Joint Local Position data from the HDF5 file')
# print('='*65)

device_name = 'pns-joint-local-position'
stream_name = 'cm-values'
# Get the data as an Nx4 matrix where each row is a timestamp and each column is an EMG channel.
pns_local_position_data = h5_file[device_name][stream_name]['data']
pns_local_position_data = np.array(pns_local_position_data)
# Get the timestamps for each row as seconds since epoch.
pns_local_position_time_s = h5_file[device_name][stream_name]['time_s']
pns_local_position_time_s = np.squeeze(np.array(pns_local_position_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# Get the timestamps for each row as human-readable strings.
pns_local_position_time_str = h5_file[device_name][stream_name]['time_str']
pns_local_position_time_str = np.squeeze(np.array(pns_local_position_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list

# print('Perception Neuron Studio Joint Local Position Data:')
# print(' Shape', pns_local_position_data.shape)
# print(' Preview:')
# print(pns_local_position_data)
# print()
# print('Perception Neuron Studio Joint Local Position Timestamps')
# print(' Shape', pns_local_position_time_s.shape)
# print(' Preview:')
# print(pns_local_position_time_s)
# print()
# print('Perception Neuron Studio Joint Local Position Timestamps as Strings')
# print(' Shape', pns_local_position_time_str.shape)
# print(' Preview:')
# print(pns_local_position_time_str)
# print()
# print(' Sampling rate: %0.2f Hz' % ((pns_local_position_data.shape[0]-1)/(max(pns_local_position_time_s) - min(pns_local_position_time_s))))
# print()

####################################################
# Example of reading sensor data: read Perception Neuron Studio Joint Global Position data.
####################################################
# print()
# print('='*65)
# print('Extracting Perception Neuron Studio Joint Global Position data from the HDF5 file')
# print('='*65)

device_name = 'pns-joint-local-position'
stream_name = 'cm-values'
# Get the data as an Nx4 matrix where each row is a timestamp and each column is an EMG channel.
pns_position_data = h5_file[device_name][stream_name]['data']
pns_position_data = np.array(pns_position_data)
# Get the timestamps for each row as seconds since epoch.
pns_position_time_s = h5_file[device_name][stream_name]['time_s']
pns_position_time_s = np.squeeze(np.array(pns_position_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# Get the timestamps for each row as human-readable strings.
pns_position_time_str = h5_file[device_name][stream_name]['time_str']
pns_position_time_str = np.squeeze(np.array(pns_position_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list

# print('Perception Neuron Studio Joint Global Position Data:')
# print(' Shape', pns_position_data.shape)
# print(' Preview:')
# print(pns_position_data)
# print()
# print('Perception Neuron Studio Joint Global Position Timestamps')
# print(' Shape', pns_position_time_s.shape)
# print(' Preview:')
# print(pns_position_time_s)
# print()
# print('Perception Neuron Studio Joint Global Position Timestamps as Strings')
# print(' Shape', pns_position_time_str.shape)
# print(' Preview:')
# print(pns_position_time_str)
# print()
# print(' Sampling rate: %0.2f Hz' % ((pns_position_data.shape[0]-1)/(max(pns_position_time_s) - min(pns_position_time_s))))
# print()

####################################################
# Example of reading sensor data: read Perception Neuron Studio Joint Local Quaternion data.
####################################################
# print()
# print('='*65)
# print('Extracting Perception Neuron Studio Joint Local Quaternion data from the HDF5 file')
# print('='*65)

device_name = 'pns-joint-quaternion'
stream_name = 'angle-values'
# Get the data as an Nx4 matrix where each row is a timestamp and each column is an EMG channel.
pns_quat_data = h5_file[device_name][stream_name]['data']
pns_quat_data = np.array(pns_quat_data)
# Get the timestamps for each row as seconds since epoch.
pns_quat_time_s = h5_file[device_name][stream_name]['time_s']
pns_quat_time_s = np.squeeze(np.array(pns_quat_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# Get the timestamps for each row as human-readable strings.
pns_quat_time_str = h5_file[device_name][stream_name]['time_str']
pns_quat_time_str = np.squeeze(np.array(pns_quat_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list

# print('Perception Neuron Studio Joint Local Quaternion Data:')
# print(' Shape', pns_quat_data.shape)
# print(' Preview:')
# print(pns_quat_data)
# print()
# print('Perception Neuron Studio Joint Local Quaternion Timestamps')
# print(' Shape', pns_quat_time_s.shape)
# print(' Preview:')
# print(pns_quat_time_s)
# print()
# print('Perception Neuron Studio Joint Local Quaternion Timestamps as Strings')
# print(' Shape', pns_quat_time_str.shape)
# print(' Preview:')
# print(pns_quat_time_str)
# print()
# print(' Sampling rate: %0.2f Hz' % ((pns_quat_data.shape[0]-1)/(max(pns_quat_time_s) - min(pns_quat_time_s))))
# print()

####################################################
# Example of reading label data
####################################################
print()
print('='*65)
print('Extracting activity labels from the HDF5 file')
print('='*65)

device_name = 'experiment-activities'
stream_name = 'activities'

# Get the timestamped label data.
# As described in the HDF5 metadata, each row has entries for ['Activity', 'Start/Stop', 'Valid', 'Notes'].
activity_datas = h5_file[device_name][stream_name]['data']
activity_times_s = h5_file[device_name][stream_name]['time_s']
activity_times_s = np.squeeze(np.array(activity_times_s))  # squeeze (optional) converts from a list of single-element lists to a 1D list
# Convert to strings for convenience.
activity_datas = [[x.decode('utf-8') for x in datas] for datas in activity_datas]

# Combine start/stop rows to single activity entries with start/stop times.
#   Each row is either the start or stop of the label.
#   The notes and ratings fields are the same for the start/stop rows of the label, so only need to check one.
exclude_bad_labels = True # some activities may have been marked as 'Bad' or 'Maybe' by the experimenter; submitted notes with the activity typically give more information
activities_labels = []
activities_start_times_s = []
activities_end_times_s = []
activities_ratings = []
activities_notes = []
for (row_index, time_s) in enumerate(activity_times_s):
  label    = activity_datas[row_index][0]
  is_start = activity_datas[row_index][1] == 'Start'
  is_stop  = activity_datas[row_index][1] == 'Stop'
  rating   = activity_datas[row_index][2]
  notes    = activity_datas[row_index][3]
  if exclude_bad_labels and rating in ['Real Collection - Bad', 'Maybe']:
    continue
  # Record the start of a new activity.
  if is_start:
    activities_labels.append(label)
    activities_start_times_s.append(time_s)
    activities_ratings.append(rating)
    activities_notes.append(notes)
  # Record the end of the previous activity.
  if is_stop:
    activities_end_times_s.append(time_s)

# print('Activity Labels:')
# print(activities_labels)
# print()
# print('Activity Start Times')
# print(activities_start_times_s)
# print()
# print('Activity End Times')
# print(activities_end_times_s)
#
#
# ####################################################
# # Example of getting sensor data for a label.
# ####################################################
# print()
# print('='*65)
# print('Extracting EMG data during a specific activity')
# print('='*65)
#
# # Get EMG data for the first instance of the second label.
# target_label = activities_labels[0]
# target_label_instance = 0
#
# # Find the start/end times associated with all instances of this label.
# label_start_times_s = [t for (i, t) in enumerate(activities_start_times_s) if activities_labels[i] == target_label]
# label_end_times_s = [t for (i, t) in enumerate(activities_end_times_s) if activities_labels[i] == target_label]
# # Only look at one instance for now.
# label_start_time_s = label_start_times_s[target_label_instance]
# label_end_time_s = label_end_times_s[target_label_instance]
#
# # Segment the data!
# emg_indexes_forLabel = np.where((gforce_lower_emg_time_s >= label_start_time_s) & (gforce_lower_emg_time_s <= label_end_time_s))[0]
# emg_data_forLabel = gforce_lower_emg_data[emg_indexes_forLabel, :]
# emg_time_s_forLabel = gforce_lower_emg_time_s[emg_indexes_forLabel]
# emg_time_str_forLabel = gforce_lower_emg_time_str[emg_indexes_forLabel]
#
# print('EMG Data for Instance %d of Label "%s"' % (target_label_instance, target_label))
# print()
# print('Label instance start time  :', label_start_time_s)
# print('Label instance end time    :', label_end_time_s)
# print('Label instance duration [s]:', (label_end_time_s-label_start_time_s))
# print()
# print('EMG data during instance:')
# print(' Shape:', emg_data_forLabel.shape)
# print(' Preview:', emg_data_forLabel)
# print()
# print('EMG timestamps during instance:')
# print(' Shape:', emg_time_s_forLabel.shape)
# print(' Preview:', emg_time_s_forLabel)
# print()
# print('EMG timestamps as strings during instance:')
# print(' Shape:', emg_time_str_forLabel.shape)
# print(' Preview:', emg_time_str_forLabel)

# ####################################################
# # Example of resampling data so segmented lengths
# #  can match across sensors with different rates.
# # Note that the below example resamples the entire
# #  data, but it could also be applied to individual
# #  extracted segments if desired.
# ####################################################
# print()
# print('='*65)
# print('Resampling segmented Cognionics EMG data to match the gForce EMG sampling rate')
# print('='*65)
#
# # Get acceleration data.
# device_name = 'EMG-DominantLeg-cognionics'
# stream_name = 'emg-values'
# # Get the data as an Nx3 matrix where each row is a timestamp and each column is an acceleration axis.
# cognionics_emg_data = h5_file[device_name][stream_name]['data']
# cognionics_emg_data = np.array(cognionics_emg_data)
# # Get the timestamps for each row as seconds since epoch.
# cognionics_emg_time_s = h5_file[device_name][stream_name]['time_s']
# cognionics_emg_time_s = np.squeeze(np.array(cognionics_emg_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# # Get the timestamps for each row as human-readable strings.
# cognionics_emg_time_str = h5_file[device_name][stream_name]['time_str']
# cognionics_emg_time_str = np.squeeze(np.array(cognionics_emg_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list
#
# # Resample the cognionics EMG data to match the EMG timestamps.
# #  Note that the IMU streamed at about 50 Hz while the EMG streamed at about 200 Hz.
# fn_interpolate_acceleration = interpolate.interp1d(
#                                 cognionics_emg_time_s, # x values
#                                 cognionics_emg_data,   # y values
#                                 axis=0,              # axis of the data along which to interpolate
#                                 kind='linear',       # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
#                                 fill_value='extrapolate' # how to handle x values outside the original range
#                                 )
# cognionics_emg_s_resampled = gforce_lower_emg_time_s
# cognionics_emg_resampled = fn_interpolate_acceleration(cognionics_emg_s_resampled)
#
# print('gForce EMG Data:')
# print(' Shape', gforce_lower_emg_data.shape)
# print(' Sampling rate: %0.2f Hz' % ((gforce_lower_emg_data.shape[0]-1)/(max(gforce_lower_emg_time_s) - min(gforce_lower_emg_time_s))))
# print()
# print('Cognionics EMG Data Original:')
# print(' Shape', cognionics_emg_data.shape)
# print(' Sampling rate: %0.2f Hz' % ((cognionics_emg_data.shape[0]-1)/(max(cognionics_emg_time_s) - min(cognionics_emg_time_s))))
# print()
# print('Cognionics EMG Data Resampled to gForce EMG Timestamps:')
# print(' Shape', cognionics_emg_resampled.shape)
# print(' Sampling rate: %0.2f Hz' % ((cognionics_emg_resampled.shape[0]-1)/(max(cognionics_emg_s_resampled) - min(cognionics_emg_s_resampled))))
# print()












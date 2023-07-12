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
filepath = './Data_Archive/Sub00/2023-01-02_10-54-08_badminton-wearables_Sub00/2023-02-15_14-37-22_streamLog_badminton-wearables_Sub00.hdf5'

# Open the file.
h5_file = h5py.File(filepath, 'r')

####################################################
# Example of reading sensor data: read gForce Lower EMG data.
####################################################
print()
print('='*65)
print('Extracting gForce Lower EMG data from the HDF5 file')
print('='*65)

device_name = 'gforce-lowerarm-emg'
stream_name = 'emg-values'
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

device_name = 'gforce-upperarm-emg'
stream_name = 'emg-values'
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

device_name = 'cgx-aim-leg-emg'
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
# Get the data as an Nx2 matrix where each row is a timestamp and each column is an EMG channel.
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

device_name = 'moticon-insole'
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

device_name = 'moticon-insole'
stream_name = 'left-acceleration'
# Get the data as an Nx3 matrix where each row is a timestamp and each column is an EMG channel.
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
# Example of reading sensor data: read Perception Neuron Studio Joint Angular Velocity data.
####################################################
print()
print('='*65)
print('Extracting Perception Neuron Studio Joint Angular Velocity data from the HDF5 file')
print('='*65)

device_name = 'pns-joint'
stream_name = 'Euler-angle'
# Get the data as an Nx63 matrix where each row is a timestamp and each column is an Euler angle channel.
pns_euler_angle_data = h5_file[device_name][stream_name]['data']
pns_euler_angle_data = np.array(pns_euler_angle_data)
# Get the timestamps for each row as seconds since epoch.
pns_euler_angle_time_s = h5_file[device_name][stream_name]['time_s']
pns_euler_angle_time_s = np.squeeze(np.array(pns_euler_angle_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
# Get the timestamps for each row as human-readable strings.
pns_euler_angle_time_str = h5_file[device_name][stream_name]['time_str']
pns_euler_angle_time_str = np.squeeze(np.array(pns_euler_angle_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list

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
print(pns_euler_angle_time_str)
print()
print(' Sampling rate: %0.2f Hz' % ((pns_euler_angle_data.shape[0]-1)/(max(pns_euler_angle_time_s) - min(pns_euler_angle_time_s))))
print()
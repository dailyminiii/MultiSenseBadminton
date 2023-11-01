############
# MIT License
#
# Copyright (c) 2023 Minwoo Seong
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
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import h5py
import numpy as np
from scipy import interpolate  # for resampling
from scipy.signal import butter, lfilter  # for filtering
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from collections import OrderedDict
import os, glob
import pandas as pd

script_dir = os.path.dirname(os.path.realpath(__file__))

# from helpers import *
from utils.print_utils import *
from utils.dict_utils import *
from utils.time_utils import *

#######################################
############ CONFIGURATION ############
#######################################

# Define where outputs will be saved.
output_dir = os.path.join(script_dir, 'data_processed')
output_filepath = os.path.join(output_dir, 'data_processed_allStreams_60hz_21subj_allActs.hdf5') # output file name
annotation_data_filePath = '../Badminton ActionNet/Data_Archive/Annotation Data File.xlsx' # directory of annotation data xlsx file
# output_filepath = None

# Define the modalities to use.
# Each entry is (device_name, stream_name, extraction_function)
# where extraction_function can select a subset of the stream columns.
device_streams_for_features = [
    ('eye-gaze', 'gaze', lambda data: data),
    ('gforce-lowerarm-emg', 'emg-values', lambda data: data),
    ('gforce-upperarm-emg', 'emg-values', lambda data: data),
    ('cgx-aim-leg-emg', 'emg-values', lambda data: data),
    ('moticon-insole', 'left-pressure', lambda data: data),
    ('moticon-insole', 'right-pressure', lambda data: data),
    ('moticon-insole', 'cop', lambda data: data),
    ('pns-joint', 'Euler-angle', lambda data: data),
]

# Specify the input data.
# data_root_dir = os.path.join(script_dir, 'Data_Archive')
data_root_dir = '../Badminton ActionNet/Data_Archive/'

data_folders_bySubject = OrderedDict([
    ('Sub00', os.path.join(data_root_dir, 'Sub00')),
    ('Sub01', os.path.join(data_root_dir, 'Sub01')),
    ('Sub02', os.path.join(data_root_dir, 'Sub02')),
    ('Sub03', os.path.join(data_root_dir, 'Sub03')),
    ('Sub04', os.path.join(data_root_dir, 'Sub04')),
    # ('Sub05', os.path.join(data_root_dir, 'Sub05')),
    # ('Sub06', os.path.join(data_root_dir, 'Sub05')),
    ('Sub07', os.path.join(data_root_dir, 'Sub07')),
    # ('Sub08', os.path.join(data_root_dir, 'Sub08')),
    ('Sub09', os.path.join(data_root_dir, 'Sub09')),
    ('Sub10', os.path.join(data_root_dir, 'Sub10')),
    ('Sub11', os.path.join(data_root_dir, 'Sub11')),
    # ('Sub12', os.path.join(data_root_dir, 'Sub12')),
    ('Sub13', os.path.join(data_root_dir, 'Sub13')),
    ('Sub14', os.path.join(data_root_dir, 'Sub14')),
    ('Sub15', os.path.join(data_root_dir, 'Sub15')),
    ('Sub16', os.path.join(data_root_dir, 'Sub16')),
    ('Sub17', os.path.join(data_root_dir, 'Sub17')),
    ('Sub18', os.path.join(data_root_dir, 'Sub18')),
    ('Sub19', os.path.join(data_root_dir, 'Sub19')),
    ('Sub20', os.path.join(data_root_dir, 'Sub20')),
    ('Sub21', os.path.join(data_root_dir, 'Sub21')),
    ('Sub22', os.path.join(data_root_dir, 'Sub22')),
    ('Sub23', os.path.join(data_root_dir, 'Sub23')),
    ('Sub24', os.path.join(data_root_dir, 'Sub24')),
])

# Specify the labels to include.  These should match the labels in the HDF5 files.
baseline_label = 'None'
activities_to_classify = [  # Total Number is 3
    baseline_label,
    'Forehand Clear',
    'Backhand Driving',
]

baseline_index = activities_to_classify.index(baseline_label)
# Some older experiments may have had different labels.
#  Each entry below maps the new name to a list of possible old names.
activities_renamed = {
    'Forehand Clear': ['Forehand Clear'], # Change name to Forehand clear
    'Backhand Driving': ['Backhand Driving'],
}

# Define segmentation parameters.
resampled_Fs = 60  # define a resampling rate for all sensors to interpolate
num_segments_per_subject = 10
num_baseline_segments_per_subject = 10  # num_segments_per_subject*(max(1, len(activities_to_classify)-1))
segment_duration_s = 2.5
segment_length = int(round(resampled_Fs * segment_duration_s))
buffer_startActivity_s = 0.01
buffer_endActivity_s = 0.01

# Define filtering parameters.
filter_cutoff_emg_Hz = 15
filter_cutoff_emg_cognionics_Hz = 20
filter_cutoff_pressure_Hz = 10
filter_cutoff_gaze_Hz = 10

# Make the output folder if needed.
if output_dir is not None:
    os.makedirs(output_dir, exist_ok=True)
    print('\n')
    print('Saving outputs to')
    print(output_filepath)
    print('\n')

################################################
############ INTERPOLATE AND FILTER ############
################################################

# Will filter each column of the data.
def lowpass_filter(data, cutoff, Fs, order=4):
    nyq = 0.5 * Fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data.T).T
    return y

def convert_to_nan(arr, difff, time):
    for i in range(len(arr) - time):
        for j in range(len(arr[0])):
            diff = abs(arr[i, j] - arr[i + time, j])
            if diff > difff:
                arr[i, j] = np.nan
    return arr

# Load the original data.
data_bySubject = {}
for (subject_id, data_folder) in data_folders_bySubject.items():
    print()
    print('id : ', subject_id)
    print()
    print('Loading data for subject %s' % subject_id)
    data_bySubject[subject_id] = []
    hdf_filepaths = glob.glob(os.path.join(data_folder, '**/*.hdf5'), recursive=True)
    print(hdf_filepaths)
    for hdf_filepath in hdf_filepaths:
        data_bySubject[subject_id].append({})
        hdf_file = h5py.File(hdf_filepath, 'r')
        # Add the activity label information.
        have_all_streams = True
        # try:
        #     device_name = 'experiment-activities'
        #     stream_name = 'activities'
        #     data_bySubject[subject_id][-1].setdefault(device_name, {})
        #     data_bySubject[subject_id][-1][device_name].setdefault(stream_name, {})
        #     for key in ['time_s', 'data']:
        #         data_bySubject[subject_id][-1][device_name][stream_name][key] = hdf_file[device_name][stream_name][key][
        #                                                                         :]
        #     num_activity_entries = len(data_bySubject[subject_id][-1][device_name][stream_name]['time_s'])
        #     if num_activity_entries == 0:
        #         have_all_streams = False
        #     elif data_bySubject[subject_id][-1][device_name][stream_name]['time_s'][0] == 0:
        #         have_all_streams = False
        # except KeyError:
        #     have_all_streams = False
        # Load data for each of the streams that will be used as features.
        for (device_name, stream_name, _) in device_streams_for_features:
            data_bySubject[subject_id][-1].setdefault(device_name, {})
            data_bySubject[subject_id][-1][device_name].setdefault(stream_name, {})
            for key in ['time_s', 'data']:
                try:
                    data_bySubject[subject_id][-1][device_name][stream_name][key] = hdf_file[device_name][stream_name][
                                                                                        key][:]
                except KeyError:
                    have_all_streams = False
        if not have_all_streams:
            data_bySubject[subject_id].pop()
            print('  Ignoring HDF5 file:', hdf_filepath)
        hdf_file.close()

# print(data_bySubject)
# Filter data.
print()
for (subject_id, file_datas) in data_bySubject.items():
    print('Filtering data for subject %s' % subject_id)
    for (data_file_index, file_data) in enumerate(file_datas):
        # print('file_data : ', file_data)
        print(' Data file index', data_file_index)
        # Filter EMG data.
        for gforce_key in ['gforce-lowerarm-emg', 'gforce-upperarm-emg']:
            if gforce_key in file_data:
                t = file_data[gforce_key]['emg-values']['time_s']
                Fs = (t.size - 1) / (t[-1] - t[0])
                print(' Filtering %s with Fs %0.1f Hz to cutoff %f' % (gforce_key, Fs, filter_cutoff_emg_Hz))
                data_stream = file_data[gforce_key]['emg-values']['data'][:, :]
                y = np.abs(data_stream)
                y = lowpass_filter(y, filter_cutoff_emg_Hz, Fs)
                # for i in range(len(data_stream[0])):
                #     plt.plot(t-t[0], data_stream[:, i], label=gforce_key+'_raw')
                #     plt.plot(t-t[0], y[:, i], label=gforce_key+'_preprocessed')
                #     plt.legend()
                #     plt.show()
                #     plt.clf()
                #     plt.plot(t[500:900] - t[0], data_stream[500:900, i], label=gforce_key + '_raw')
                #     plt.plot(t[500:900] - t[0], y[500:900, i], label=gforce_key + '_preprocessed')
                #     plt.legend()
                #
                #     plt.show()
                #     plt.clf()
                file_data[gforce_key]['emg-values']['data'] = y
        for cognionics_key in ['cgx-aim-leg-emg']:
            if cognionics_key in file_data:
                t = file_data[cognionics_key]['emg-values']['time_s']
                Fs = (t.size - 1) / (t[-1] - t[0])
                print(' Filtering %s with Fs %0.1f Hz to cutoff %f' % (
                cognionics_key, Fs, filter_cutoff_emg_cognionics_Hz))
                data_stream = file_data[cognionics_key]['emg-values']['data'][:, :]
                data_stream = np.abs(data_stream)
                # Correcting the bounce value
                y = convert_to_nan(data_stream, difff=80, time=5)
                y[y > 26000] = np.nan
                # y[y < -26000] = np.nan
                # y[y < -26000] = np.nan
                df = pd.DataFrame(y)
                # print(df.isnull().sum())
                for ii in range(len(df.loc[0])):
                    df.loc[:, ii] = df.loc[:, ii].fillna(df.loc[:, ii].median())
                    # print(df.loc[:, ii].mean())
                # print(df.isnull().sum())
                y = df.to_numpy()
                y = lowpass_filter(y, filter_cutoff_emg_cognionics_Hz, Fs)
                # for i in range(len(data_stream[0])):
                #     # print('max', np.amax(data_stream[:, i]))
                #     # print('min', np.amin(data_stream[:, i]))
                #     plt.plot(t-t[0], data_stream[:, i], label=cognionics_key+'_raw_channel' + str(i+1))
                #     plt.plot(t - t[0], y[:, i], label=cognionics_key + '_preprocessed_channel'+ str(i+1))
                #     plt.legend()
                #     plt.show()
                #     plt.clf()
                #     plt.plot(t[50000:55000] - t[0], data_stream[50000:55000, i], label=cognionics_key + '_raw_channel' + str(i+1))
                #     plt.plot(t[50000:55000] - t[0], y[50000:55000, i], label=cognionics_key + '_preprocessed_channel' + str(i+1))
                #     plt.legend()
                #     plt.show()
                #     plt.clf()
                #     plt.plot(t[50000:55000] - t[0], y[50000:55000, i], label=cognionics_key + '_preprocessed_channel' + str(i+1))
                #     plt.legend()
                #     plt.show()
                #     plt.clf()
                file_data[cognionics_key]['emg-values']['data'] = y
        # Filter eye-gaze data.
        if 'eye-gaze' in file_data:
            t = file_data['eye-gaze']['gaze']['time_s']
            Fs = (t.size - 1) / (t[-1] - t[0])

            data_stream = file_data['eye-gaze']['gaze']['data'][:, :]
            y = data_stream

            # # Apply a ZOH to remove clipped values.
            # #  The gaze position is already normalized to video coordinates,
            # #   so anything outside [0,1] is outside the video.
            # print(' Holding clipped values in %s' % ('eye-gaze'))
            clip_low_x = 0 + 0.05
            clip_high_x = 1088 - 0.05
            clip_low_y = 0 + 0.05
            clip_high_y = 1080 - 0.05
            y[:, 0] = np.clip(y[:, 0], clip_low_x, clip_high_x)
            y[:, 1] = np.clip(y[:, 1], clip_low_y, clip_high_y)
            y[y == clip_low_x] = np.nan
            y[y == clip_high_x] = np.nan
            y[y == clip_low_y] = np.nan
            y[y == clip_high_y] = np.nan
            y = pd.DataFrame(y).interpolate(method='zero').to_numpy()
            # # Replace any remaining NaNs with a dummy value,
            # #  in case the first or last timestep was clipped (interpolate() does not extrapolate).
            y[np.isnan(y)] = 540
            # plt.plot(t-t[0], y[:,0], '*-')
            # plt.ylim(-2,2)
            # Filter to smooth.
            print('   Filtering %s with Fs %0.1f Hz to cutoff %f' % ('eye-gaze', Fs, filter_cutoff_gaze_Hz))
            y = lowpass_filter(y, filter_cutoff_gaze_Hz, Fs)
            # for i in range(len(data_stream[0])):
            #     plt.plot(t - t[0], data_stream[:, i], label='eye-gaze' + '_raw')
            #     plt.plot(t-t[0], y[:, i], label='eye-gaze'+'_preprocessed')
            #     plt.legend()
            #     plt.show()
            #     plt.clf()
            file_data['eye-gaze']['gaze']['data'] = y
        for moticon_key in ['left-pressure', 'right-pressure', 'cop']:
            if moticon_key in file_data:
                t = file_data['moticon-insole'][moticon_key]['time_s']
                Fs = (t.size - 1) / (t[-1] - t[0])
                print(' Filtering %s with Fs %0.1f Hz to cutoff %f' % ('moticon-insole', Fs, filter_cutoff_pressure_Hz))
                data_stream = file_data['moticon-insole'][moticon_key]['data'][:, :]
                y = np.abs(data_stream)
                y = lowpass_filter(y, filter_cutoff_pressure_Hz, Fs)
                # plt.plot(t-t[0], data_stream[:,0], label=moticon_key+'_raw')
                # plt.plot(t-t[0], y[:,0], label=moticon_key+'_preprocessed')
                # plt.legend()
                # plt.show()
                file_data['moticon-insole'][moticon_key]['data'] = y
        data_bySubject[subject_id][data_file_index] = file_data

# Normalize data.
print()
for (subject_id, file_datas) in data_bySubject.items():
    print('Normalizing data for subject %s' % subject_id)
    for (data_file_index, file_data) in enumerate(file_datas):
        # Normalize gForce Pro EMG data.
        for gforce_key in ['gforce-lowerarm-emg', 'gforce-upperarm-emg']:
            if gforce_key in file_data:
                data_stream = file_data[gforce_key]['emg-values']['data'][:, :]
                y = data_stream
                print(' Normalizing %s with min/max [%0.1f, %0.1f]' % (gforce_key, np.amin(y), np.amax(y)))
                # Normalize them jointly.
                y = y / ((np.amax(y) - np.amin(y)) / 2)
                # Jointly shift the baseline to -1 instead of 0.
                y = y - np.amin(y) - 1
                file_data[gforce_key]['emg-values']['data'] = y
                print('   Now has range [%0.1f, %0.1f]' % (np.amin(y), np.amax(y)))
                # plt.plot(y.reshape(y.shape[0], -1))
                # plt.show()

        # Normalize Cognionics EMG data.
        for cognionics_key in ['cgx-aim-leg-emg']:
            if cognionics_key in file_data:
                data_stream = file_data[cognionics_key]['emg-values']['data'][:, :]
                y = data_stream
                print(' Normalizing %s with min/max [%0.1f, %0.1f]' % (cognionics_key, np.amin(y), np.amax(y)))
                y = y / ((np.amax(y) - np.amin(y)) / 2)
                # Jointly shift the baseline to -1 instead of 0.
                y = y - np.amin(y) - 1
                file_data[cognionics_key]['emg-values']['data'] = y
                print('   Now has range [%0.1f, %0.1f]' % (np.amin(y), np.amax(y)))
                # plt.plot(y.reshape(y.shape[0], -1))
                # plt.show()

        # Normalize Perception Neuron Studio joints.
        if 'pns-joint-euler' in file_data:
            data_stream = file_data['pns-joint']['Euler-angle']['data'][:, :]
            y = data_stream
            min_val = -180
            max_val = 180
            print(' Normalizing %s with forced min/max [%0.1f, %0.1f]' % ('pns-joint-euler', min_val, max_val))
            # Normalize all at once since using fixed bounds anyway.
            # Preserve relative bends, such as left arm being bent more than the right.
            y = y / ((max_val - min_val) / 2)
            file_data['pns-joint']['Euler-angle']['data'] = y
            print('   Now has range [%0.1f, %0.1f]' % (np.amin(y), np.amax(y)))
            # plt.plot(y.reshape(y.shape[0], -1))
            # plt.show()

        # Normalize eyetracking gaze.
        if 'eye-gaze' in file_data:
            data_stream = file_data['eye-gaze']['gaze']['data'][:]
            t = file_data['eye-gaze']['gaze']['time_s'][:]
            y = data_stream
            min_x = 0
            max_x = 1088
            min_y = 0
            max_y = 1080

            print(' Normalizing %s with min/max [%0.1f, %0.1f] and min/max [%0.1f, %0.1f]' % (
            'eye-gaze', min_x, max_x, min_y, max_y))
            # # The gaze position is already normalized to video coordinates,
            # #  so anything outside [0,1] is outside the video.
            clip_low = -0.95
            clip_high = 0.95

            # y = np.clip(y, clip_low, clip_high)
            # Put in range [-1, 1] for extra resolution.
            # Normalize them jointly.
            y[:, 0] = y[:, 0] / ((max_x - min_x) / 2)
            y[:, 1] = y[:, 1] / ((max_y - min_y) / 2)
            # Jointly shift the baseline to -1 instead of 0.
            y = y - min_y - 1
            # y = (y - np.mean([clip_low, clip_high])) / ((clip_high - clip_low) / 2)
            # print(' Clipping %s to [%0.1f, %0.1f]' % ('eye-gaze', clip_low, clip_high))
            # plt.plot(t-t[0], y)
            # plt.show()
            file_data['eye-gaze']['gaze']['data'] = y
            print('   Now has range [%0.1f, %0.1f]' % (np.amin(y), np.amax(y)))
            # plt.plot(y.reshape(y.shape[0], -1))
            # plt.show()

        # Normalize Moticon Pressure.
        for moticon_key in ['left-pressure', 'right-pressure', 'cop']:
            if moticon_key in file_data:
                data_stream = file_data['moticon-insole'][moticon_key]['data'][:, :]
                y = data_stream
                print(' Normalizing %s with min/max [%0.1f, %0.1f]' % ('moticon-insole', np.amin(y), np.amax(y)))
                # Normalize them jointly.
                y = y / ((np.amax(y) - np.amin(y)) / 2)
                # Jointly shift the baseline to -1 instead of 0.
                y = y - np.amin(y) - 1
                file_data['moticon-insole'][moticon_key]['data'] = y
                print('   Now has range [%0.1f, %0.1f]' % (np.amin(y), np.amax(y)))
                # plt.plot(y.reshape(y.shape[0], -1))
                # plt.show()

        data_bySubject[subject_id][data_file_index] = file_data

# Aggregate data (and normalize if needed).
print()
for (subject_id, file_datas) in data_bySubject.items():
    print('Aggregating data for subject %s' % subject_id)
    for (data_file_index, file_data) in enumerate(file_datas):
        # Aggregate EMG data.
        for gforce_key in ['gforce-lowerarm-emg', 'gforce-upperarm-emg']:
            if gforce_key in file_data:
                pass

        # Aggregate eye-tracking gaze.
        if 'cgx-aim-leg-emg' in file_data:
            pass

        # Aggregate Perception Nueron Studio joints.
        if 'pns-joint' in file_data:
            pass

        # Aggregate eye-tracking gaze.
        if 'eye-gaze' in file_data:
            pass

        # Aggregate eye-tracking gaze.
        if 'moticon-insole' in file_data:
            pass

        data_bySubject[subject_id][data_file_index] = file_data

# Resample data.
print()
for (subject_id, file_datas) in data_bySubject.items():
    print('Resampling data for subject %s' % subject_id)
    for (data_file_index, file_data) in enumerate(file_datas):
        for (device_name, stream_name, _) in device_streams_for_features:
            data = np.squeeze(np.array(file_data[device_name][stream_name]['data']))
            time_s = np.squeeze(np.array(file_data[device_name][stream_name]['time_s']))
            target_time_s = np.linspace(time_s[0], time_s[-1],
                                        num=int(round(1 + resampled_Fs * (time_s[-1] - time_s[0]))),
                                        endpoint=True)
            fn_interpolate = interpolate.interp1d(
                time_s,  # x values
                data,  # y values
                axis=0,  # axis of the data along which to interpolate
                kind='linear',  # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
                fill_value='extrapolate'  # how to handle x values outside the original range
            )
            data_resampled = fn_interpolate(target_time_s)
            if np.any(np.isnan(data_resampled)):
                print('\n' * 5)
                print('=' * 50)
                print('=' * 50)
                print('FOUND NAN')
                print(subject_id, device_name, stream_name)
                timesteps_have_nan = np.any(np.isnan(data_resampled), axis=tuple(np.arange(1, np.ndim(data_resampled))))
                print('Timestep indexes with NaN:', np.where(timesteps_have_nan)[0])
                print_var(data_resampled)
                # input('Press enter to continue ')
                print('\n' * 5)
                time.sleep(10)
                data_resampled[np.isnan(data_resampled)] = 0
            file_data[device_name][stream_name]['time_s'] = target_time_s
            file_data[device_name][stream_name]['data'] = data_resampled
        data_bySubject[subject_id][data_file_index] = file_data


#########################################
############ CREATE FEATURES ############
#########################################

def get_feature_matrices(experiment_data, label_start_time_s, label_end_time_s, count=num_segments_per_subject):
    # Determine start/end times for each example segment.
    start_time_s = label_start_time_s + buffer_startActivity_s
    end_time_s = label_end_time_s - buffer_endActivity_s
    segment_start_times_s = np.linspace(start_time_s, end_time_s - segment_duration_s,
                                        num=count,
                                        endpoint=True)
    # Create a feature matrix by concatenating each desired sensor stream.
    feature_matrices = []
    for segment_start_time_s in segment_start_times_s:
        # print('Processing segment starting at %f' % segment_start_time_s)
        segment_end_time_s = segment_start_time_s + segment_duration_s
        feature_matrix = np.empty(shape=(segment_length, 0))
        for (device_name, stream_name, extraction_fn) in device_streams_for_features:

            print(' Adding data from [%s][%s]' % (device_name, stream_name))
            data = np.squeeze(np.array(experiment_data[device_name][stream_name]['data']))
            time_s = np.squeeze(np.array(experiment_data[device_name][stream_name]['time_s']))
            time_indexes = np.where((time_s >= segment_start_time_s) & (time_s <= segment_end_time_s))[0]
            # Expand if needed until the desired segment length is reached.
            time_indexes = list(time_indexes)
            while len(time_indexes) < segment_length:
                print(' Increasing segment length from %d to %d for %s %s for segment starting at %f' % (
                    len(time_indexes), segment_length, device_name, stream_name, segment_start_time_s))
                if time_indexes[0] > 0:
                    time_indexes = [time_indexes[0] - 1] + time_indexes
                elif time_indexes[-1] < len(time_s) - 1:
                    time_indexes.append(time_indexes[-1] + 1)
                else:
                    raise AssertionError
            while len(time_indexes) > segment_length:
                print(' Decreasing segment length from %d to %d for %s %s for segment starting at %f' % (
                    len(time_indexes), segment_length, device_name, stream_name, segment_start_time_s))
                time_indexes.pop()
            time_indexes = np.array(time_indexes)

            # Extract the data.
            time_s = time_s[time_indexes]
            data = data[time_indexes, :]
            data = extraction_fn(data)
            print('  Got data of shape', data.shape)
            # Add it to the feature matrix.
            data = np.reshape(data, (segment_length, -1))
            if np.any(np.isnan(data)):
                print('\n' * 5)
                print('=' * 50)
                print('=' * 50)
                print('FOUND NAN')
                print(device_name, stream_name, segment_start_time_s)
                timesteps_have_nan = np.any(np.isnan(data), axis=tuple(np.arange(1, np.ndim(data))))
                print('Timestep indexes with NaN:', np.where(timesteps_have_nan)[0])
                print_var(data)
                # input('Press enter to continue ')
                print('\n' * 5)
                time.sleep(10)
                data[np.isnan(data)] = 0
            feature_matrix = np.concatenate((feature_matrix, data), axis=1)
        feature_matrices.append(feature_matrix)
    # print(len(feature_matrices), feature_matrices[0].shape)
    return feature_matrices


#########################################
############ CREATE EXAMPLES ############
#########################################

example_labels = []
example_label_indexes = []
example_matrices_list = []
example_subject_ids = []
Forehand_time_list = []
Backhand_time_list = []

df = pd.read_excel(annotation_data_filePath)

for (subject_id, file_datas) in data_bySubject.items():
    print('Resampling data for subject %s' % subject_id)
    df_subject = df[df["Subject Number"] == subject_id]
    df_subject_forehand = df_subject[df_subject["Annotation Level 1\n(Stroke Type)"] == 'Forehand Clear']
    df_subject_backhand = df_subject[df_subject["Annotation Level 1\n(Stroke Type)"] == 'Backhand Driving']

    for (data_file_index, file_data) in enumerate(file_datas):

        sub_example_labels = []
        sub_example_label_indexes = []
        sub_example_subject_ids = []

        Forehand_start_time_list = df_subject_forehand['Annotation Start Time'].values.tolist()
        Forehand_stop_time_list = df_subject_forehand['Annotation Stop Time'].values.tolist()
        Backhand_start_time_list = df_subject_backhand['Annotation Start Time'].values.tolist()
        Backhand_stop_time_list = df_subject_backhand['Annotation Stop Time'].values.tolist()
        NoActivity_start_time_list = []
        NoActivity_stop_time_list = []
        NoActivity_start_time_list.extend(Forehand_stop_time_list[0:-1])
        NoActivity_start_time_list.extend(Backhand_stop_time_list[0:-1])
        NoActivity_stop_time_list.extend(Forehand_start_time_list[1:])
        NoActivity_stop_time_list.extend(Backhand_start_time_list[1:])

        example_matrices_device_eye_gaze = []
        example_matrices_device_gforce_lowerarm_emg = []
        example_matrices_device_gforce_upperarm_emg = []
        example_matrices_device_cgx_aim_emg = []
        example_matrices_device_moticon_insole_left_pressure = []
        example_matrices_device_moticon_insole_right_pressure = []
        example_matrices_device_moticon_insole_cop = []
        example_matrices_device_pns_joint_Euler = []

        print('Total Labeling Number :', len(Forehand_start_time_list) + len(Backhand_start_time_list))
        print('Saved Highclear Labeling Number :', len(Forehand_start_time_list), len(Forehand_stop_time_list))
        print('Saved Backhand Labeling Number :', len(Backhand_start_time_list), len(Backhand_stop_time_list))
        print('Saved NoActivity Labeling Number :', len(NoActivity_start_time_list), len(NoActivity_stop_time_list))

        device_num = 1
        for (device_name, stream_name, _) in device_streams_for_features:
            example_matrices_each_file = []
            print("Device Name :", device_name)
            data = np.squeeze(np.array(file_data[device_name][stream_name]['data']))
            time_s = np.squeeze(np.array(file_data[device_name][stream_name]['time_s']))
            label_indexes = [0] * len(time_s)

            # Initialize the Number of each stroke
            Num_base = 0
            Num_clear = 0
            Num_drive = 0

            # Save the Forehand Clear Data
            highNum = 0
            backNum = 0
            baseNum = 0

            for j in range(len(Forehand_start_time_list)):
                # Save the swing time of each stroke
                Forehand_time_list.append(Forehand_stop_time_list[j]-Forehand_start_time_list[j])
                # time indexing
                high_time_indexes = np.where((time_s >= Forehand_start_time_list[j]) & (time_s <= Forehand_stop_time_list[j]))

                if len(high_time_indexes[0]) > 0:
                    if len(data[high_time_indexes[0][0]:high_time_indexes[0][0] + segment_length, :]) == segment_length:
                        example_matrices_each_file.append(data[high_time_indexes[0][0] :high_time_indexes[0][0] + segment_length, :].tolist())
                        highNum += 1
                        if len(device_streams_for_features) == device_num:
                            sub_example_label_indexes.append(1)
                            sub_example_labels.append('Forehand Clear')
                            sub_example_subject_ids.append(subject_id)
                            Num_clear += 1
                for m in range(len(high_time_indexes[0])):
                    label_indexes[high_time_indexes[0][m]] = 1

            for j in range(len(Backhand_start_time_list)):
                Backhand_time_list.append(Backhand_stop_time_list[j] - Backhand_start_time_list[j])
                back_time_indexes = np.where((time_s >= Backhand_start_time_list[j]) & (time_s <= Backhand_stop_time_list[j]))
                if len(back_time_indexes[0]) > 0:
                    if len(data[back_time_indexes[0][0]: back_time_indexes[0][0] + segment_length, :]) == segment_length:
                        example_matrices_each_file.append(data[back_time_indexes[0][0]:back_time_indexes[0][0] + segment_length, :].tolist())
                        backNum += 1
                        if len(device_streams_for_features) == device_num:
                            sub_example_label_indexes.append(2)
                            sub_example_labels.append('Backhand Driving')
                            sub_example_subject_ids.append(subject_id)
                            Num_drive += 1
                for m in range(len(back_time_indexes[0])):
                    label_indexes[back_time_indexes[0][m]] = 2

            # Save the Baseline Data
            for j in range(len(NoActivity_start_time_list)):
                no_time_indexes = np.where(
                    (time_s >= NoActivity_start_time_list[j]) & (time_s <= NoActivity_stop_time_list[j]))
                if len(no_time_indexes[0]) > 0:
                    if len(data[no_time_indexes[0][0]:no_time_indexes[0][0] + segment_length, :]) == segment_length:
                        example_matrices_each_file.append(data[no_time_indexes[0][0]:no_time_indexes[0][0] + segment_length, :].tolist())
                        baseNum += 1
                        if len(device_streams_for_features) == device_num:
                            sub_example_label_indexes.append(0)
                            sub_example_labels.append('Baseline')
                            sub_example_subject_ids.append(subject_id)
                            Num_base += 1

            if device_name == "eye-gaze":
                example_matrices_device_eye_gaze.append(example_matrices_each_file)
            elif device_name == "gforce-lowerarm-emg":
                example_matrices_device_gforce_lowerarm_emg.append(example_matrices_each_file)
            elif device_name == "gforce-upperarm-emg":
                example_matrices_device_gforce_upperarm_emg.append(example_matrices_each_file)
            elif device_name == "cgx-aim-leg-emg":
                example_matrices_device_cgx_aim_emg.append(example_matrices_each_file)
            elif device_name == "moticon-insole" and stream_name == 'left-pressure':
                example_matrices_device_moticon_insole_left_pressure.append(example_matrices_each_file)
            elif device_name == "moticon-insole" and stream_name == 'right-pressure':
                example_matrices_device_moticon_insole_right_pressure.append(example_matrices_each_file)
            elif device_name == "moticon-insole" and stream_name == 'cop':
                example_matrices_device_moticon_insole_cop.append(example_matrices_each_file)
            elif device_name == "pns-joint":
                example_matrices_device_pns_joint_Euler.append(example_matrices_each_file)

            device_num += 1
            # print("highNum")
            # print(highNum)
            # print("backNum")
            # print(backNum)
            # print("baseNum")
            # print(baseNum)
            # print("totalNum")
            # print(highNum + backNum + baseNum)

        device_array1 = np.squeeze(np.array(example_matrices_device_eye_gaze), axis=0)
        device_array2 = np.squeeze(np.array(example_matrices_device_gforce_lowerarm_emg), axis=0)
        device_array3 = np.squeeze(np.array(example_matrices_device_gforce_upperarm_emg), axis=0)
        device_array4 = np.squeeze(np.array(example_matrices_device_cgx_aim_emg), axis=0)
        device_array5 = np.squeeze(np.array(example_matrices_device_moticon_insole_left_pressure), axis=0)
        device_array6 = np.squeeze(np.array(example_matrices_device_moticon_insole_right_pressure), axis=0)
        device_array7 = np.squeeze(np.array(example_matrices_device_moticon_insole_cop), axis=0)
        device_array8 = np.squeeze(np.array(example_matrices_device_pns_joint_Euler), axis=0)

        print("Feature shape of each device")
        print(device_array1.shape)
        print(device_array2.shape)
        print(device_array3.shape)
        print(device_array4.shape)
        print(device_array5.shape)
        print(device_array6.shape)
        print(device_array7.shape)
        print(device_array8.shape)

        if all(array.shape[0] == device_array1.shape[0] for array in [device_array2, device_array3, device_array4, device_array5, device_array6, device_array7, device_array8]):
            combined_array = np.concatenate((device_array1, device_array2, device_array3, device_array4, device_array5, device_array6, device_array7, device_array8), axis=2)
            print(combined_array.shape)
            print(np.array(sub_example_label_indexes).shape)
            print(np.array(sub_example_labels).shape)
            print(np.array(sub_example_subject_ids).shape)
            example_label_indexes.extend(sub_example_label_indexes)
            example_labels.extend(sub_example_labels)
            example_subject_ids.extend(sub_example_subject_ids)
        else:
            min_ = min(device_array1.shape[0], device_array2.shape[0], device_array3.shape[0], device_array4.shape[0], device_array5.shape[0], device_array6.shape[0], device_array7.shape[0], device_array8.shape[0])
            print(min_)
            example_label_indexes.extend(sub_example_label_indexes[:min_])
            example_labels.extend(sub_example_labels[:min_])
            example_subject_ids.extend(sub_example_subject_ids[:min_])
            combined_array = np.concatenate((device_array1[:min_, :, :], device_array2[:min_, :, :], device_array3[:min_, :, :], device_array4[:min_, :, :], device_array5[:min_, :, :], device_array6[:min_, :, :], device_array7[:min_, :, :], device_array8[:min_, :, :]), axis=2)
            print(combined_array.shape)
        # combined_array_squeezd = np.squeeze(combined_array, axis=0)
        example_matrices_list.append(combined_array)

example_matrices = np.concatenate([arr for arr in example_matrices_list], axis=0)
print('total feature shape')
print(example_matrices.shape)
print(np.array(example_labels).shape)
print(np.array(example_label_indexes).shape)
print(np.array(example_subject_ids).shape)

if output_filepath is not None:
    with h5py.File(output_filepath, 'w') as hdf_file:
        metadata = OrderedDict()
        metadata['output_dir'] = output_dir
        metadata['data_root_dir'] = data_root_dir
        metadata['data_folders_bySubject'] = data_folders_bySubject
        metadata['activities_to_classify'] = activities_to_classify
        metadata['device_streams_for_features'] = device_streams_for_features
        metadata['resampled_Fs'] = resampled_Fs
        metadata['segment_length'] = segment_length
        metadata['segment_duration_s'] = segment_duration_s
        metadata['filter_cutoff_emg_Hz'] = filter_cutoff_emg_Hz
        metadata['filter_cutoff_pressure_Hz'] = filter_cutoff_pressure_Hz
        metadata['filter_cutoff_gaze_Hz'] = filter_cutoff_gaze_Hz

        metadata = convert_dict_values_to_str(metadata, preserve_nested_dicts=False)

        hdf_file.create_dataset('example_labels', data=example_labels)
        hdf_file.create_dataset('example_label_indexes', data=example_label_indexes)
        hdf_file.create_dataset('example_matrices', data=example_matrices)
        hdf_file.create_dataset('example_subject_ids', data=example_subject_ids)

        hdf_file.attrs.update(metadata)

        print()
        print('Saved processed data to', output_filepath)
        print()

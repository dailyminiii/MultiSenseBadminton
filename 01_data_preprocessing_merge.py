import h5py
import numpy as np
from scipy import interpolate  # for resampling
from scipy.signal import butter, lfilter  # for filtering
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from collections import OrderedDict
import os, glob

script_dir = os.path.dirname(os.path.realpath(__file__))
# script_dir = 'D:\Github\ActionNet'

from helpers import *
from utils.print_utils import *
from utils.dict_utils import *
from utils.time_utils import *

#######################################
############ CONFIGURATION ############
#######################################

# Define where outputs will be saved.
output_dir = os.path.join(script_dir, 'data_processed')
output_filepath = os.path.join(output_dir, 'data_processed_allStreams_0205s_60hz_21subj_ex150-150_allActs_all_15_20_10_10.hdf5')
# output_filepath = None
#
# Define the modalities to use.
# Each entry is (device_name, stream_name, extraction_function)
# where extraction_function can select a subset of the stream columns.
device_streams_for_features = [
    ('eye-gaze', 'gaze', lambda data: data),
    # # ('eye-gaze', 'worn', lambda data: data),
    ('armband-gforce-lowerarm', 'pressure_values_N_cm2', lambda data: data),
    ('armband-gforce-upperarm', 'pressure_values_N_cm2', lambda data: data),
    ('EMG-DominantLeg-cognionics', 'emg-values', lambda data: data),
    ('insole-moticon-left-pressure', 'pressure_values_N_cm2', lambda data: data),
    ('insole-moticon-right-pressure', 'pressure_values_N_cm2', lambda data: data),
    ('insole-moticon-cop', 'cop', lambda data: data),
    ('pns-joint-euler', 'angle-values', lambda data: data),
    # ('pns-joint-angular-velocity', 'velocity-values', lambda data: data),
    # ('pns-joint-local-position', 'cm-values', lambda data: data),
]

# Specify the input data.
data_root_dir = os.path.join(script_dir, 'data_archive')

data_folders_bySubject = OrderedDict([
    ('S04', os.path.join(data_root_dir, 'S04')),
    ('S05_2', os.path.join(data_root_dir, 'S05_2')),
    ('S06_2', os.path.join(data_root_dir, 'S06_2')),
    ('S07', os.path.join(data_root_dir, 'S07')),
    ('S08', os.path.join(data_root_dir, 'S08')),
    # ('S09', os.path.join(data_root_dir, 'S09')),
    # ('S10', os.path.join(data_root_dir, 'S10')),
    ('S11_2', os.path.join(data_root_dir, 'S11_2')),
    # ('S12', os.path.join(data_root_dir, 'S12')),
    ('S13', os.path.join(data_root_dir, 'S13')),
    ('S14', os.path.join(data_root_dir, 'S14')),
    ('S15', os.path.join(data_root_dir, 'S15')),
    # ('S16', os.path.join(data_root_dir, 'S16')),
    ('S17', os.path.join(data_root_dir, 'S17')),
    ('S18', os.path.join(data_root_dir, 'S18')),
    ('S19', os.path.join(data_root_dir, 'S19')),
    ('S20', os.path.join(data_root_dir, 'S20')),
    ('S21', os.path.join(data_root_dir, 'S21')),
    ('S22', os.path.join(data_root_dir, 'S22')),
    ('S23', os.path.join(data_root_dir, 'S23')),
    ('S24', os.path.join(data_root_dir, 'S24')),
    ('S25', os.path.join(data_root_dir, 'S25')),
    ('S26', os.path.join(data_root_dir, 'S26')),
    ('S27', os.path.join(data_root_dir, 'S27')),
    ('S28', os.path.join(data_root_dir, 'S28')),
])

# Specify the labels to include.  These should match the labels in the HDF5 files.
baseline_label = 'None'
activities_to_classify = [ # Total Number is 3
    baseline_label,
    'Overhead Clear',
    'Backhand Driving',
]

baseline_index = activities_to_classify.index(baseline_label)
# Some older experiments may have had different labels.
#  Each entry below maps the new name to a list of possible old names.
activities_renamed = {
    'Overhead Clear': ['Overhead Clear'],
    'Backhand Driving': ['Backhand Driving'],
}

#Define segmentation parameters.
resampled_Fs = 60  # define a resampling rate for all sensors to interpolate
num_segments_per_subject = 150
num_baseline_segments_per_subject = 150  # num_segments_per_subject*(max(1, len(activities_to_classify)-1))
segment_duration_s = 2.5
segment_length = int(round(resampled_Fs * segment_duration_s))
buffer_startActivity_s = 0.01
buffer_endActivity_s = 0.01

# Define filtering parameters.
filter_cutoff_emg_Hz = 15
filter_cutoff_emg_cognionics_Hz = 20
filter_cutoff_pressure_Hz = 10
filter_cutoff_gaze_Hz = 10
num_tactile_rows_aggregated = 4
num_tactile_cols_aggregated = 4

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
    for i in range(len(arr)-time):
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
    for hdf_filepath in hdf_filepaths:
        if 'archived' in hdf_filepath:
            continue
        data_bySubject[subject_id].append({})
        hdf_file = h5py.File(hdf_filepath, 'r')
        print(hdf_filepath)
        # Add the activity label information.
        have_all_streams = True
        try:
            device_name = 'experiment-activities'
            stream_name = 'activities'
            data_bySubject[subject_id][-1].setdefault(device_name, {})
            data_bySubject[subject_id][-1][device_name].setdefault(stream_name, {})
            for key in ['time_s', 'data']:
                data_bySubject[subject_id][-1][device_name][stream_name][key] = hdf_file[device_name][stream_name][key][
                                                                                :]
            num_activity_entries = len(data_bySubject[subject_id][-1][device_name][stream_name]['time_s'])
            if num_activity_entries == 0:
                have_all_streams = False
            elif data_bySubject[subject_id][-1][device_name][stream_name]['time_s'][0] == 0:
                have_all_streams = False
        except KeyError:
            have_all_streams = False
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
#Filter data.
print()
for (subject_id, file_datas) in data_bySubject.items():
    print('Filtering data for subject %s' % subject_id)
    for (data_file_index, file_data) in enumerate(file_datas):
        # print('file_data : ', file_data)
        print(' Data file index', data_file_index)
        # Filter EMG data.
        for gforce_key in ['armband-gforce-lowerarm', 'armband-gforce-upperarm']:
            if gforce_key in file_data:
                t = file_data[gforce_key]['pressure_values_N_cm2']['time_s']
                Fs = (t.size - 1) / (t[-1] - t[0])
                print(' Filtering %s with Fs %0.1f Hz to cutoff %f' % (gforce_key, Fs, filter_cutoff_emg_Hz))
                data_stream = file_data[gforce_key]['pressure_values_N_cm2']['data'][:, :]
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
                file_data[gforce_key]['pressure_values_N_cm2']['data'] = y
        for cognionics_key in ['EMG-DominantLeg-cognionics']:
            if cognionics_key in file_data:
                t = file_data[cognionics_key]['emg-values']['time_s']
                Fs = (t.size - 1) / (t[-1] - t[0])
                print(' Filtering %s with Fs %0.1f Hz to cutoff %f' % (cognionics_key, Fs, filter_cutoff_emg_cognionics_Hz))
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
        for moticon_key in ['insole-moticon-left-pressure', 'insole-moticon-right-pressure']:
            if moticon_key in file_data:
                t = file_data[moticon_key]['pressure_values_N_cm2']['time_s']
                Fs = (t.size - 1) / (t[-1] - t[0])
                print(' Filtering %s with Fs %0.1f Hz to cutoff %f' % (moticon_key, Fs, filter_cutoff_pressure_Hz))
                data_stream = file_data[moticon_key]['pressure_values_N_cm2']['data'][:, :]
                y = np.abs(data_stream)
                y = lowpass_filter(y, filter_cutoff_pressure_Hz, Fs)
                # plt.plot(t-t[0], data_stream[:,0], label=moticon_key+'_raw')
                # plt.plot(t-t[0], y[:,0], label=moticon_key+'_preprocessed')
                # plt.legend()
                # plt.show()
                file_data[moticon_key]['pressure_values_N_cm2']['data'] = y
        if 'insole-moticon-cop' in file_data:
            t = file_data['insole-moticon-cop']['cop']['time_s']
            Fs = (t.size - 1) / (t[-1] - t[0])

            data_stream = file_data['insole-moticon-cop']['cop']['data'][:, :]
            y = data_stream

            # Filter to smooth.
            print('   Filtering %s with Fs %0.1f Hz to cutoff %f' % ('insole-moticon-cop', Fs, filter_cutoff_pressure_Hz))
            y = lowpass_filter(y, filter_cutoff_pressure_Hz, Fs)
            # plt.plot(t - t[0], data_stream[:, 0], label='insole-moticon-cop'+'_raw')
            # plt.plot(t-t[0], y[:,0], label='insole-moticon-cop'+'_preprocessed')
            # # plt.ylim(-2,2)
            # plt.legend()
            # plt.show()
            file_data['insole-moticon-cop']['cop']['data'] = y
        data_bySubject[subject_id][data_file_index] = file_data

# Normalize data.
print()
for (subject_id, file_datas) in data_bySubject.items():
    print('Normalizing data for subject %s' % subject_id)
    for (data_file_index, file_data) in enumerate(file_datas):
        # Normalize gForce Pro EMG data.
        for gforce_key in ['armband-gforce-lowerarm', 'armband-gforce-upperarm']:
            if gforce_key in file_data:
                data_stream = file_data[gforce_key]['pressure_values_N_cm2']['data'][:, :]
                y = data_stream
                print(' Normalizing %s with min/max [%0.1f, %0.1f]' % (gforce_key, np.amin(y), np.amax(y)))
                # Normalize them jointly.
                y = y / ((np.amax(y) - np.amin(y)) / 2)
                # Jointly shift the baseline to -1 instead of 0.
                y = y - np.amin(y) - 1
                file_data[gforce_key]['pressure_values_N_cm2']['data'] = y
                print('   Now has range [%0.1f, %0.1f]' % (np.amin(y), np.amax(y)))
                # plt.plot(y.reshape(y.shape[0], -1))
                # plt.show()

        # Normalize Cognionics EMG data.
        for cognionics_key in ['EMG-DominantLeg-cognionics']:
            if cognionics_key in file_data:
                data_stream = file_data[cognionics_key]['emg-values']['data'][:, :]
                y = data_stream
                print(' Normalizing %s with min/max [%0.1f, %0.1f]' % (cognionics_key, np.amin(y), np.amax(y)))
                # Normalize them jointly.
                # y[:, 0] = y[:, 0] / ((np.amax(y[:, 0]) - np.amin(y[:, 0])) / 2)
                # y[:, 1] = y[:, 1] / ((np.amax(y[:, 1]) - np.amin(y[:, 1])) / 2)
                # y[:, 2] = y[:, 2] / ((np.amax(y[:, 2]) - np.amin(y[:, 2])) / 2)
                # y[:, 3] = y[:, 3] / ((np.amax(y[:, 3]) - np.amin(y[:, 3])) / 2)
                # y[:, 0] = y[:, 0] - np.amin(y[:, 0]) - 1
                # y[:, 1] = y[:, 1] - np.amin(y[:, 1]) - 1
                # y[:, 2] = y[:, 2] - np.amin(y[:, 2]) - 1
                # y[:, 3] = y[:, 3] - np.amin(y[:, 3]) - 1
                y = y / ((np.amax(y) - np.amin(y)) / 2)
                # Jointly shift the baseline to -1 instead of 0.
                y = y - np.amin(y) - 1
                file_data[cognionics_key]['emg-values']['data'] = y
                print('   Now has range [%0.1f, %0.1f]' % (np.amin(y), np.amax(y)))
                # plt.plot(y.reshape(y.shape[0], -1))
                # plt.show()

        # Normalize Perception Neuron Studio joints.
        if 'pns-joint-euler' in file_data:
            data_stream = file_data['pns-joint-euler']['angle-values']['data'][:, :]
            y = data_stream
            min_val = -180
            max_val = 180
            print(' Normalizing %s with forced min/max [%0.1f, %0.1f]' % ('pns-joint-euler', min_val, max_val))
            # Normalize all at once since using fixed bounds anyway.
            # Preserve relative bends, such as left arm being bent more than the right.
            y = y / ((max_val - min_val) / 2)
            # for i in range(20):
            #   plt.plot(y[:,i])
            #   plt.ylim(-1,1)
            #   plt.show()
            file_data['pns-joint-euler']['angle-values']['data'] = y
            print('   Now has range [%0.1f, %0.1f]' % (np.amin(y), np.amax(y)))
            # plt.plot(y.reshape(y.shape[0], -1))
            # plt.show()

        if 'pns-joint-local-position' in file_data:
            data_stream = file_data['pns-joint-local-position']['cm-values']['data'][:, :]
            y = data_stream
            print(' Normalizing %s with min/max [%0.1f, %0.1f]' % ('pns-joint-local-position', np.amin(y), np.amax(y)))
            # Normalize them jointly.
            y = y / ((np.amax(y) - np.amin(y)) / 2)
            # Jointly shift the baseline to -1 instead of 0.
            y = y - np.amin(y) - 1
            file_data['pns-joint-local-position']['cm-values']['data'] = y
            print('   Now has range [%0.1f, %0.1f]' % (np.amin(y), np.amax(y)))
            # plt.plot(y.reshape(y.shape[0], -1))
            # plt.show()

        if 'pns-joint-angular-velocity' in file_data:
            data_stream = file_data['pns-joint-angular-velocity']['velocity-values']['data'][:, :]
            y = data_stream
            print(' Normalizing %s with min/max [%0.1f, %0.1f]' % ('pns-joint-angular-velocity', np.amin(y), np.amax(y)))

            min_val = -1000.0
            max_val = 1000.0

            # Normalize them jointly.
            y = y / ((max_val - min_val) / 2)
            file_data['pns-joint-angular-velocity']['velocity-values']['data'] = y
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

            print(' Normalizing %s with min/max [%0.1f, %0.1f] and min/max [%0.1f, %0.1f]' % ('eye-gaze', min_x, max_x, min_y, max_y))
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
        for moticon_key in ['insole-moticon-left-pressure', 'insole-moticon-right-pressure']:
            if moticon_key in file_data:
                data_stream = file_data[moticon_key]['pressure_values_N_cm2']['data'][:, :]
                y = data_stream
                print(' Normalizing %s with min/max [%0.1f, %0.1f]' % (moticon_key, np.amin(y), np.amax(y)))
                # Normalize them jointly.
                y = y / ((np.amax(y) - np.amin(y)) / 2)
                # Jointly shift the baseline to -1 instead of 0.
                y = y - np.amin(y) - 1
                file_data[moticon_key]['pressure_values_N_cm2']['data'] = y
                print('   Now has range [%0.1f, %0.1f]' % (np.amin(y), np.amax(y)))
                # plt.plot(y.reshape(y.shape[0], -1))
                # plt.show()

        if 'insole-moticon-cop' in file_data:
            data_stream = file_data['insole-moticon-cop']['cop']['data'][:]
            t = file_data['insole-moticon-cop']['cop']['time_s'][:]
            y = data_stream
            min_x = -0.5
            max_x = 0.5
            min_y = -0.574
            max_y = 0.426

            print(' Normalizing %s with min/max [%0.1f, %0.1f] and min/max [%0.1f, %0.1f]' % ('insole-moticon-cop', min_x, max_x, min_y, max_y))

            # Normalize them jointly.
            y[:, 0] = y[:, 0] / ((max_x - min_x) / 2)
            y[:, 1] = y[:, 1] / ((max_y - min_y) / 2)
            # Jointly shift the baseline to -1 instead of 0.
            y = y - min_y - 1
            # y = (y - np.mean([clip_low, clip_high])) / ((clip_high - clip_low) / 2)
            # print(' Clipping %s to [%0.1f, %0.1f]' % ('eye-gaze', clip_low, clip_high))
            # plt.plot(t-t[0], y)
            # plt.show()
            file_data['insole-moticon-cop']['cop']['data'] = y
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
        for gforce_key in ['armband-gforce-lowerarm', 'armband-gforce-upperarm']:
            if gforce_key in file_data:
                pass

        # Aggregate eye-tracking gaze.
        if 'EMG-DominantLeg-cognionics' in file_data:
            pass

        # Aggregate Perception Nueron Studio joints.
        if 'pns-joint-euler' in file_data:
            pass
        if 'pns-joint-local-position' in file_data:
            pass
        if 'pns-joint-angular-velocity' in file_data:
            pass
        # Aggregate eye-tracking gaze.
        if 'eye-tracking-gaze' in file_data:
            pass

        # Aggregate eye-tracking gaze.
        for moticon_key in ['insole-moticon-left-pressure', 'insole-moticon-right-pressure', 'insole-moticon-cop']:
            if moticon_key in file_data:
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

# Will store intermediate examples from each file.
example_matrices_byLabel = {}
# Then will create the following 'final' lists with the correct number of examples.
example_labels = []
example_label_indexes = []
example_matrices = []
example_subject_ids = []
print()
for (subject_id, file_datas) in data_bySubject.items():
    print()
    print('Processing data for subject %s' % subject_id)
    noActivity_matrices = []
    for (data_file_index, file_data) in enumerate(file_datas):
        # Get the timestamped label data.
        # As described in the HDF5 metadata, each row has entries for ['Activity', 'Start/Stop', 'Valid', 'Notes'].
        device_name = 'experiment-activities'
        stream_name = 'activities'
        activity_datas = file_data[device_name][stream_name]['data']
        activity_times_s = file_data[device_name][stream_name]['time_s']
        activity_times_s = np.squeeze(
            np.array(activity_times_s))  # squeeze (optional) converts from a list of single-element lists to a 1D list
        # Convert to strings for convenience.
        activity_datas = [[x.decode('utf-8') for x in datas] for datas in activity_datas]
        # Combine start/stop rows to single activity entries with start/stop times.
        #   Each row is either the start or stop of the label.
        #   The notes and ratings fields are the same for the start/stop rows of the label, so only need to check one.
        exclude_bad_labels = False  # some activities may have been marked as 'Bad' or 'Maybe' by the experimenter; submitted notes with the activity typically give more information
        activities_labels = []
        activities_start_times_s = []
        activities_end_times_s = []
        activities_ratings = []
        activities_notes = []
        for (row_index, time_s) in enumerate(activity_times_s):
            label = activity_datas[row_index][0]
            is_start = activity_datas[row_index][1] == 'Start'
            is_stop = activity_datas[row_index][1] == 'Stop'
            rating = activity_datas[row_index][2]
            notes = activity_datas[row_index][3]
            if rating is None:
                print('!!! Rating None !!!!')
            if exclude_bad_labels and rating in ['Bad', 'Normal']:
                continue
            # Record the start of a new activity.
            if is_start and rating is not None:
                activities_labels.append(label)
                activities_start_times_s.append(time_s)
                activities_ratings.append(rating)
                activities_notes.append(notes)
            # Record the end of the previous activity.
            if is_stop and rating is not None:
                activities_end_times_s.append(time_s)
        # Loop through each activity that is designated for classification.
        for (label_index, activity_label) in enumerate(activities_to_classify):
            if label_index == baseline_index:
                continue
            # Extract num_segments_per_subject examples from each instance of the activity.
            # Then later, will select num_segments_per_subject in total from all instances.
            file_label_indexes = [i for (i, label) in enumerate(activities_labels) if label == activity_label]
            if len(file_label_indexes) == 0 and activity_label in activities_renamed:
                for alternate_label in activities_renamed[activity_label]:
                    file_label_indexes = [i for (i, label) in enumerate(activities_labels) if label == alternate_label]
                    if len(file_label_indexes) > 0:
                        print('  Found renamed activity from "%s"' % alternate_label)
                        break
            print('  Found %d instances of %s' % (len(file_label_indexes), activity_label))

            for file_label_index in file_label_indexes:
                try:
                    start_time_s = activities_start_times_s[file_label_index]
                    end_time_s = activities_end_times_s[file_label_index]
                    duration_s = end_time_s - start_time_s
                    # Extract example segments and generate a feature matrix for each one.
                    # num_examples = int(num_segments_per_subject/len(file_label_indexes))
                    # if file_label_index == file_label_indexes[-1]:
                    #   num_examples = num_segments_per_subject - num_examples*(len(file_label_indexes)-1)
                    num_examples = num_segments_per_subject
                    print('  Extracting %d examples from activity "%s" with duration %0.2fs' % (
                    num_examples, activity_label, duration_s))
                    # print('file data : ', file_data)
                    feature_matrices = get_feature_matrices(file_data,
                                                            start_time_s, end_time_s,
                                                            count=num_examples)
                    example_matrices_byLabel.setdefault(activity_label, [])
                    example_matrices_byLabel[activity_label].extend(feature_matrices)
                except IndexError:
                    print('list index out of range')

        # Generate matrices for not doing any activity.
        # Will generate one matrix for each inter-activity portion,
        #  then later select num_baseline_segments_per_subject of them.
        for (label_index, activity_label) in enumerate(activities_labels):
            if label_index == len(activities_labels) - 1:
                continue
            print('  Getting baseline examples between activity "%s"' % (activity_label))
            noActivity_start_time_s = activities_end_times_s[label_index]
            noActivity_end_time_s = activities_start_times_s[label_index + 1]
            duration_s = noActivity_end_time_s - noActivity_start_time_s
            if duration_s < segment_duration_s:
                continue
            # Extract example segments and generate a feature matrix for each one.
            feature_matrices = get_feature_matrices(file_data,
                                                    noActivity_start_time_s,
                                                    noActivity_end_time_s,
                                                    count=10)
            noActivity_matrices.extend(feature_matrices)

    # Choose a subset of the examples of each label, so the correct number is retained.
    # Will evenly distribute the selected indexes over all possibilities.
    for (activity_label_index, activity_label) in enumerate(activities_to_classify):
        if activity_label_index == baseline_index:
            continue
        print(' Selecting %d examples for subject %s of activity "%s"' % (
        num_segments_per_subject, subject_id, activity_label))
        if activity_label not in example_matrices_byLabel:
            print('\n' * 5)
            print('=' * 50)
            print('=' * 50)
            print('  No examples found!')
            # print('  Press enter to continue ')
            print('\n' * 5)
            time.sleep(10)
            continue
        feature_matrices = example_matrices_byLabel[activity_label]
        example_indexes = np.round(np.linspace(0, len(feature_matrices) - 1,
                                               endpoint=True,
                                               num=num_segments_per_subject,
                                               dtype=int))
        for example_index in example_indexes:
            example_labels.append(activity_label)
            example_label_indexes.append(activity_label_index)
            example_matrices.append(feature_matrices[example_index])
            example_subject_ids.append(subject_id)

    # Choose a subset of the baseline examples.
    print(' Selecting %d examples for subject %s of activity "%s"' % (
    num_baseline_segments_per_subject, subject_id, baseline_label))
    noActivity_indexes = np.round(np.linspace(0, len(noActivity_matrices) - 1,
                                              endpoint=True,
                                              num=num_baseline_segments_per_subject,
                                              dtype=int))
    for noActivity_index in noActivity_indexes:
        # try:
        example_labels.append(baseline_label)
        example_label_indexes.append(baseline_index)
        example_matrices.append(noActivity_matrices[noActivity_index])
        example_subject_ids.append(subject_id)
        # except IndexError:
        #     print('INDEX ERROR')

print()

#########################################
############# SAVE RESULTS  #############
#########################################

if output_filepath is not None:
    with h5py.File(output_filepath, 'w') as hdf_file:
        metadata = OrderedDict()
        metadata['output_dir'] = output_dir
        metadata['data_root_dir'] = data_root_dir
        metadata['data_folders_bySubject'] = data_folders_bySubject
        metadata['activities_to_classify'] = activities_to_classify
        metadata['device_streams_for_features'] = device_streams_for_features
        metadata['segment_duration_s'] = segment_duration_s
        metadata['segment_length'] = segment_length
        metadata['num_segments_per_subject'] = num_segments_per_subject
        metadata['num_baseline_segments_per_subject'] = num_baseline_segments_per_subject
        metadata['buffer_startActivity_s'] = buffer_startActivity_s
        metadata['buffer_endActivity_s'] = buffer_endActivity_s
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





# ActionNet - Badminton Stroke Data Collection

For dataset, you can refer to the site: XXX

The sports industry is witnessing an increasing trend of utilizing multiple synchronized sensors to collect player data, enabling the creation of personalized training systems with real-time feedback from multiple perspectives. Badminton could benefit from these various sensors, but there is a notable lack of comprehensive badminton action datasets for analysis and training feedback. To address this gap, this paper introduces a multi-sensor-based badminton action dataset for forehand clear and backhand drive strokes. This includes 7,763 badminton swing data from 25 players. It provides eye tracking, body tracking, muscle signals, foot pressure, detailed annotation data on stroke type, skill level, hitting sound, ball landing, hitting location, survey data, and interview data. The dataset was designed based on interviews with badminton coaches to ensure usability. The dataset includes a range of skills consisting of 12 novices, 8 intermediates, and 5 experts, providing resources for understanding biomechanics across skill levels. We validate the potential usefulness of our dataset by applying a proof-of-concept machine learning model to classify stroke type and level of expertise.

## Sensors to use

![Sensors to use](https://user-images.githubusercontent.com/79134282/233352475-a961fe8a-ba6c-4d77-a83b-8449ddeea52e.jpg)

## Data Collection Framework

![Data Collection Framework](https://user-images.githubusercontent.com/79134282/233351724-3bba3af4-1cbf-442c-a77f-a836ac986298.jpg)

## Sensor Data Description 

The sensor data is stored in HDF5 format for each subject, with each dataset containing specific information. The details of the datasets are as follows:

**1) cgx-aim-leg-emg**

This dataset contains electromyography (EMG) values for the dominant leg, measured in millivolts (mV). It consists of four channels, and each channel represents specific muscle information. For detailed muscle descriptions, please refer to the MultiSenseBadminton paper.

**2) experiment-activities**

This dataset records the start and end times of each motion activity. It has three channels:

The first channel indicates the stroke type.
The second channel indicates whether the motion has started or stopped.
The third channel indicates whether the corresponding data is saved or blank. "Saved" indicates that the stroke data is correctly captured, while "blank" indicates the absence of stroke data.

**3) experiment-calibration**

The calibration data consists of three channels and is specific to gforce and leg EMG. The details of the calibration dataset are as follows:

The calibration dataset has three channels:

1) Start/Stop Status: This channel indicates the start and stop status of the calibration process.
2) Calibration Data Storage: The second channel indicates whether the calibration data has been stored properly. If the calibration data is saved correctly, it is marked as "Good". Otherwise, it is left blank.
3) Calibration Type: The third channel specifies the type of calibration performed.
Calibration was conducted for two types of data: gforce and leg EMG. The calibration process involved performing a total of three motions for gforce and two motions for leg EMG. The motions performed are as follows:

For gforce:

Lower Arm Inward Motion
Lower Arm Outward Motion
Upper Arm Inward Motion

For leg EMG:

Leg Force Motion
Squat Motion

The purpose of calibration was to extract the maximum EMG value from the recorded data. The calibration data allows for proper normalization and calibration of subsequent EMG measurements.

**4) Eye-gaze**

This dataset captures eye-tracking data using pupil invisible glasses. It consists of two channels:
The first channel represents the X-value of gaze positions, ranging from 0 to 1088.
The second channel represents the Y-value of gaze positions, ranging from 0 to 1080.
Additionally, the dataset includes a "worn" column which indicates whether the corresponding data was predicted while wearing glasses. The value 1 represents wearing glasses, while 0 represents not wearing glasses.

**5) gforce-lowerarm-emg**

This dataset provides EMG values for the lower arm, measured in normalized units ranging from 0 to 250. It contains data for eight channels.

**6) gforce-upperarm-emg**

Similar to the previous dataset, this dataset contains EMG values for the upper arm. It also consists of eight channels, and the data is normalized between 0 and 250.

**7) moticon-insole**

This dataset contains multiple types of data, including:

Center of Pressure (COP): It consists of four channels representing the x and y coordinates of COP for the left and right foot. [Left_X, Left_Y, Right_X, Right_Y] 4 Channels
Acceleration: This data captures linear acceleration in the x, y, and z directions, measured in g (gravity). [ACC_X, ACC_Y, ACC_Z] 3 Channels
Angular Velocity: It records the angular velocity in the x, y, and z axes, measured in (degree/s). [Vel_X, Vel_Y, Vel_Z] 3 Channels
Pressure: The pressure data is measured in Newton per square centimeter (N/cm²). [Pressure_1 ~ Pressure_16] 16 Channels
Total Force: It represents the total force and is reported in Newton (N) units. [Total Force] 1 Channels

**8) pns-joint**: This dataset contains information related to joint positions, velocities, and angles. It includes the following parameters for each joint (total of 21 joints):

Local Position: The position of the joint in centimeters (cm).
Global Position: The global position of the joint (cm).
Angular Velocity: The angular velocity of the joint in degrees per second (degree/s).
Euler Angle: The Euler angle of the joint in degrees. (degree)
Quaternion: The quaternion representation of the joint.

The order of the joints in the dataset is as follows: Hip, Right Up Leg, Right Leg, Right Foot, Left Up Leg, Left Leg, Left Foot, Spine, Spine 1, Spine 2, Neck, Neck 1, Head, Right Shoulder, Right Arm, Right Forearm, Right Hand, Left Shoulder, Left Arm, Left Forearm, Left Hand. (**total 21 joints**)

For further details and specific usage instructions, please refer to the MultiSenseBadminton paper associated with the dataset.

## Environment

![Environment](https://user-images.githubusercontent.com/79134282/233352857-31ca2d5e-73ab-4e29-b44b-ae304c2011ab.jpg)

## Data Annotation

![Annotation Level](https://github.com/dailyminiii/MultiSenseBadminton/assets/79134282/b3b6351a-8048-4b62-bddb-960eb698129d)

## Data Preprocessing

![Data Preprocessing](https://user-images.githubusercontent.com/79134282/233353008-060aae23-26a0-4684-8337-97ab22ac88e7.jpg)

## Network Architecture

![Network Architecture](https://user-images.githubusercontent.com/79134282/233352749-cfac7fec-f370-4b53-a91e-d581793011a0.jpg)



## Contact

If you have any questions regarding the dataset, please feel free to contact me at seongminwoo@gm.gist.ac.kr.










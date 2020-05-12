# Collaborative Whiteboard: An Exploration of Utilizing Kinect Depth Data

Collaborative Whiteboard is a prototype implementation of a multi-modal interactive whiteboard. Our implementation allows users to manipulate a virtual whiteboard interface using both voice commands and finger movements on arbitrary projection surfaces.

# Project Components

## RGB-D Data Collection

The RGB-D data collection codebase is located in the folder *./rgbd_scan*.
The main script can be run as follows: **ALLLLLLLLLAN**

An example data sequence extracted using this script is available at:
**Link to Sample Data Sequence**

## Finger Position Tracking

The Finger tracking codebase is located in the folder *./finger_tracking*.
To run the main script, which extracts and saves finger positions from a data sequence, first change the line defining the variable **PROJECT_ROOT_DIR** to the root dir of the project on the local machine.

Then, run:
```python finger_tracking/main.py --sequence_dir $DIR$```
where $DIR$ is the subdirectory within **PROJECT_ROOT_DIR**/sequences where the data sequence is located.

## User Interface

The UI codebase is located in *./frontend*.
To run an example of the UI, do: **IISSSSSSSRAEL**

# Dependencies
- Python >= 3.7
- opencv-python >= 4.1.1.26
- numpy >= 1.18.1 

# Authors
Alan Cheng, Alex Kimn, and Israel Macias

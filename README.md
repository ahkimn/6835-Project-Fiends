# Collaborative Whiteboard: An Exploration of Utilizing Kinect Depth Data

Collaborative Whiteboard is a prototype implementation of a multi-modal interactive whiteboard. Our implementation allows users to manipulate a virtual whiteboard interface using both voice commands and finger movements on arbitrary projection surfaces.

# Project Components

## RGB-D Data Collection

The RGB-D data collection codebase is located in the folder *./rgbd_scan*.
It can be compiled on run using Visual Studios (we tested on the 2017 version).
We have only included the source and header files since copying an entire vsproj doesn't work well

OpenCV 3.x.x and Kinect SDK V2 must be included in the project. 

(OpenCV) https://docs.opencv.org/2.4/doc/tutorials/introduction/windows_visual_studio_Opencv/windows_visual_studio_Opencv.html

(Kinect SDK) https://docs.microsoft.com/en-us/previous-versions/windows/kinect/dn799271(v=ieb.10)

As Visual Studio setup is a hassle, we have also included example data sequences captured from our setup.
The following Dropbox link contains the example datasets 
https://www.dropbox.com/sh/y7ip34b92y8yzru/AAARZCA63v2hk80DA-LTsSLAa?dl=0

NOTE: these files are large (around 2GB for each sequence)
We have saved the output contour overlay for the *stonks* sequence in the *contours* folder.

## Finger Position Tracking

The Finger tracking codebase is located in the folder *./finger_tracking*.

Download any sequences from the Dropbox link above, and place inside the *./sequence/* folder (at the same level as the example folder)

To run the main script, which extracts and saves finger positions from a data sequence, first change the line defining the variable **PROJECT_ROOT_DIR** to the root dir of the project on the local machine.

Then, run:
```python finger_tracking/main.py --sequence_dir $DIR$```
where $DIR$ is the subdirectory within **PROJECT_ROOT_DIR**/sequences where the data sequence is located.

## User Interface

The UI codebase is located in *./frontend*.
To run an example of the UI: 

- Run index.html which opens the interface in a browser. This is confirmed to work on the latest Google Chrome
- This works best when the UI is full screen, thus it is advised to put your browser in fullscreen mode.
- In demo.py, there is a replace **DATA_DIR** with the directory that contains the data sequence from finger tracking. Demo.py will load the data coordinates and execute all of the sequenced coordinates onto the UI. 

# Dependencies
- Python >= 3.7
- opencv-python >= 3.4.2
- numpy >= 1.18.1 
- pyautogui >= 0.9.50

# Authors
Alan Cheng, Alex Kimn, and Israel Macias

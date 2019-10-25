# Tracking
The following code is an uncompleted version of calculating hand trajectory using Realsense D435, yolov3(F-RCNN) and opencv tracker 'CSRT' for further tracking the hand.

## Requirements
opencv(==3.4.6) built with tracker repos from opencv-contrib.
libtorch with cuda 9 or 9.2
Also, please provide the realsense bag files and enter there path in the main.cpp file.


### Running the code
To run the code, first you need to provide the libtorch path in src/CMakeLists.txt.

```
mkdir build
cd build
cmake ..

```

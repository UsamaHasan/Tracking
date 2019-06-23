# Tracking
The following code is an uncompleted version of tracking hand trajectory using yolov3 and opencv tracker 'CSRT'.

## Requirements
opencv(==3.4.6) built with tracker repos from opencv-contrib.
libtorch with cuda 9 pr 9.2

### Running the code
To run the code, first you need to provide the libtorch path in src/CMakeLists.txt.

```
mkdir build
cd build
cmake ..

```

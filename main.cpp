// -*- lsst-c++ -*-
/**
 *Author: Usama Hasan
*/
#include"pipeline.hpp"
int main()
{
    Pipeline pipeline;
    //Insert path to your bag file here
    std::string bag_file = "/home/usamahasan/tracking/C++/bag_files/82921207023520190501_142849.bag";
    //Insert path to your YOLO cfg here
    char* cfg_file = "/home/usamahasan/tracking/yolofiles/yolov3.cfg";
    //Inset path to weight file here
    char* weight_file = "/home/usamahasan/tracking/yolofiles/final.weight";
    //Read the documentaion of the function
    pipeline.pipe(cfg_file , weight_file , bag_file);

}
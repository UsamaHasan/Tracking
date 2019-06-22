// -*- lsst-c++ -*-
/**
 * Author : Usama Hasan
*/
/**
 * @brief Container class for objects
 * This class is a container for the detected objects from the models output.
 * The objects that were detected will have a certain class,
 * @param[in] name will represent their class,
 * @param[out] id will be allocated as multiple items of same class will be present in 
 * a use case,  
 * @param[in] ground_truth will contain the detected output of YOLO,
 * @param[out] will contain the trackers output 
 * 
*/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include "helpers.hpp"
class Object
{
    private:
        std::string name;
        int id ;
        region_of_interest  *ground_truth;
        region_of_interest  *result_roi;
        cv::Ptr<cv::Tracker> tracker;       
        
    /**
     * setter function for name
     * @param[in] name_ to initialize name of object
    */
    void set_name (std::string name_)
    {
        this->name = name_;
    }
    /**
     *getter function for name
     *@returns name
    */    
    std::string get_name()
    {   
        return this->name;
    }
    /**
     * setter function for ground_truth output of YOLO mode
     * @param[in] pointer to array to initialize ground_truth for object
    */
    void set_ground_truth(region_of_interest roi)
    {
        this->ground_truth =& roi;
    }
    /**
     * getter function for ground_truth of object.
     * @returns pointer to array of ground_truth of object.
    */
    region_of_interest * get_ground_truth()
    {
        return this->ground_truth;
    }
    /**
     * setter function to intialize result roi obtained from object tracker's output
     * @param result_roi poitnter to array to initialize output of tracker.
    */
    void set_result_roi(region_of_interest roi)
    {
        this->result_roi =& roi;
    }
    /**
     * getter function for to return roi tracked by opencv tracker
     * @returns roi pointer to array result_roi 
    */
    region_of_interest * get_result_roi()
    {
        return this->result_roi;
    }
    public:
    /**
     * @breif intialize cv::tracker on the detected roi
    */
    void initialize_trakcer()
    {
        //Our by default choice is CSRT, due to his high performance on our dataset.
        this->tracker = cv::TrackerCSRT::create();
    }
    /**
     * 
    */
    void update_tracker();
};
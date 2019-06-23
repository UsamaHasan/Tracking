// -*- lsst-c++ -*-
/**
 * Author : Usama Hasan
*/
/**
 * The following class will Initialize Yolov3 model in Libtorch framework,
 * @param[in] name contains the name of model
 * @param[in] num_classes the number of classes model will detect
 * @param[in] detection_threshold the accuracy threshold of model
 * @param[in] nms_threshold threshold to remove noise ouputs
 * @param[in] input_image_size 
*/
#include "NeuralNet.h"
#include<opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include<iostream>
#include "helpers.hpp"
class Model
{
    private:
        std::string name;
        float detection_threshold;
        float nms_threshold;
        int input_image_size; 
        int num_classes ;
        NeuralNet *nn;
        torch::Device *device;
    /**
     * Default constructor
    */
   public:
    /**
     * Overloaded Constructor
     *@param[in] name_ Name of model
     *@param[in] num_class Number of classes of model
     *@param[in] detection_thresh Accuracy for object detection
     *@param[in] nms threshold to remove noise
    */
    Model(std::string name_, int num_class  ,float detection_thresh , float nms )
    {
        this->name = name_;
        this->num_classes = num_class;
        this->detection_threshold = detection_thresh;
        this->nms_threshold = nms;
        this->nn = NULL; 
    }

    /**
     * The function will initialize Torch backend based on the compile binaries provided 
     * by the user
     * @returns torch::device() with available backend
    */
    void device_init()
    {   
        torch::DeviceType device_type;

        if (torch::cuda::is_available() ) 
        {        
            device_type = torch::kCUDA;
        } 
        else
        {
        device_type = torch::kCPU;
        }
        
        this->device  = new torch::Device(device_type);
    }

    /**
     * Initialize a YOLO model with torch backend
     * @param[in] cfg should contain the path of yolo configuration file
     * @param[in] weight_file should contain the path of model weight file to be loaded
     * @param[in] device should contain torch::Device object with prefreable backend set.
    */
    void initialization(char* cfg, char* weight_file)
    {
        // input image size for YOLO model
        this->input_image_size = 416;
        
        //YOLO model implementation in torch object
        
        this->nn = new NeuralNet(cfg,this->device);

        map<string, string> *info = nn->get_net_info();

        info->operator[]("height") = std::to_string(input_image_size);

        this->nn->load_weights(weight_file);

        this->nn->to(*this->device);

        this->nn->eval();
    }
    /**
     * Convert input image to model required format
     * @param[in] image input to model
     * @returns resized float image
    */
    cv::Mat preprocess_input(cv::Mat image)
    {
        cv::Mat resized_image;
        cv::cvtColor(image, resized_image,  cv::COLOR_RGB2BGR);
        cv::resize(resized_image, resized_image, cv::Size(this->input_image_size, this->input_image_size));
        cv::Mat img_float;
        resized_image.convertTo(img_float, CV_32F, 1.0/255);
        return img_float;
    }
    /**
     * Model will run a forward pass and returns output
     * @param[in] input Input tensor to be feeded to the network
     * @returns results Output of model
    */
    
    torch::Tensor forward_pass(cv::Mat image )
    {

        auto device = torch::Device(torch::kCPU);
        torch::Tensor img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(image.data, {1, this->input_image_size, this->input_image_size, 3});
        img_tensor = img_tensor.permute({0,3,1,2});
        auto  image_tensor = torch::autograd::make_variable(img_tensor, false).to(*this->device);
    
        auto output = this->nn->forward(image_tensor);
        auto result = this->nn->write_results(output, this->num_classes, this->detection_threshold, this->nms_threshold);
        return result;
    }
    /**
     * Process model output to return roi of objects
     * @param[out] image Input Image on which results will be drawn
     * @param[in]  result torch::Tensor output of model
     * @returns the function will return the attributes of all the objects detected
    */
    std::list <object_attributes> output_processing(cv::Mat &image  , torch::Tensor result)
    {
        std::list <object_attributes> object_attribute_list;
        if (result.dim() == 1)
        {
            return object_attribute_list;
            //Case where no object is found.
        }
        else
        {   
            //Remaining Task section
            //Extract Class id Of object from here 
            //Change the function to return results
            //Further split the process were we create objects of class 'Object'
            //with respect to the id's of detected ids
            //Our core focus is to obtain hand and the overlapping object
            //As soon as the object is detected initialize a tracker on it
            //We an overlap is detected , write a heuristic which will further 
            //try to understand the trajectory of tracking

            float width_scale = float(image.cols) / this->input_image_size;
            float height_scale = float(image.rows) / this->input_image_size;

            result.select(1,1).mul_(width_scale);
            result.select(1,2).mul_(height_scale);
            result.select(1,3).mul_(width_scale);
            result.select(1,4).mul_(height_scale);
            
            torch::TensorAccessor <float ,2>results = result.accessor<float, 2>();
            
            draw_rectangles(image , results);
            return list_of_obj_attributes(results);
        }
        
    }
    ~Model(){
        delete this->nn;
        delete this->device;
    }
};
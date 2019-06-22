// -*- lsst-c++ -*-
/**
 * Author : Usama Hasan
*/
#include "model.hpp"
#include "helpers.hpp"
#include "object.hpp"
#include <librealsense2/rs.hpp>
#include <opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
/**
 * @brief A container class to run all the modules.
*/
class Pipeline{
    
   private:

    Model *model;
    /**
     *initialize torch model by providing cfg file and pre trained weights
     @param[in] cfg Yolo configuration file
     @param[in] weight_file Pretrained Files 
    */
    void initialize_model(char* cfg , char* weight_file)
    {
        std::string model_name = "YOLO";
        this->model = new Model(model_name , 2 , 0.6 , 0.6);
        this->model->device_init();
        this->model->initialization(cfg , weight_file);  
    }
    /**
     * Pass image tensor to model to get results
     * @param[in] image to feed to the model
    */
    std::list<object_attributes> model_inference(cv::Mat &image)
    {
        cv::Mat processed_img = this->model->preprocess_input(image);
        auto results = this->model->forward_pass(processed_img);
        auto list = this->model->output_processing(image , results);
        return list;
    }
    /**
     * @brief check if hand is detected.
     * @returns boolean value if hand is detected.  
    */
    bool check_hand(std::list<object_attributes> attribute_list)
    {
        std::list<object_attributes>::iterator iter ;
        for (iter = attribute_list.begin(); iter != attribute_list.end(); iter++)
        {   
            if (iter->get_class_id() == 0)
            {
                return 1;
            }
        }
        return 0;
    }
   public:
   /**
    * Deafault Constructor of class Pipeline
   */
  Pipeline()
    {
        model = NULL;
    }
    /**
     * Function will start realsense pipeline to read frames from either the connected 
     * camera stream or the provided bag file and further perform inference to get the
     * required output
     * @param[in] yolo_cfg input your yolo model cfg here
     * @param[in] weight_file provide your weight file here
     * @param[in] bag_file input your path to recorded realsense frames  
    */
    void pipe(char* yolo_cfg , char* weight_file ,std::string bag_file )
    {
        rs2::config cfg;
        //Initialize stream using Realsense bag file containing recorded frames
        cfg.enable_device_from_file(bag_file);
        rs2::pipeline pipe;
        //Initialize pipeline
        pipe.start(cfg);

        initialize_model(yolo_cfg , weight_file);

        while(true)
        {
            //Receive frames from pipeline
            auto frames  = pipe.wait_for_frames();
            //RGB Frame
            rs2::frame color_frame = frames.get_color_frame();
            //Convert rs2::frame RGB frme into cv::Mat 
            cv::Mat color_image = frame_to_mat(color_frame);
            //list contains attributes of detected class like their location and class id.
            std::list<object_attributes>list_of_attrs = model_inference(color_image);
            
            //Check if for the output
            if(list_of_attrs.size() == 0)
            {
                //Move to next frame 
                continue;
            }
            else
            {    
                //hand detection flag 
               bool flag = this->check_hand(list_of_attrs);
               if(flag)
               {
                    //Create Hand object to start tracking          

               }
            }
            
            //And then check for the overlap item
            //Start tracking on both hand and item
            
            cv::imshow("Color_frame" , color_image);
            cv::waitKey(1);
        }
    
    }
    
};
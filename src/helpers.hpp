// -*- lsst-c++ -*-
/**
 *Author: Usama Hasan
*/
/**
 * 
*/

#pragma once
#include<torch/torch.h>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include <librealsense2/rs.hpp>
/**
 * @brief to store region of interest
*/
struct region_of_interest{
    private:
        int x_1;
        int y_1;
        int x_2;
        int y_2;
    public:
        region_of_interest()
        {
            this->x_1 = 0;
            this->y_1 = 0;
            this->x_2 = 0;
            this->y_2 = 0;
        }
        region_of_interest(int x1, int y1, int x2, int y2)
        {
            this->x_1 = x1;
            this->x_2 = x2;
            this->y_1 = y1;
            this->y_2 = y2;
        }
};
/**
 * @brief To store the region of interest and class id
*/
struct object_attributes{
    private:
    int class_id ;
    region_of_interest roi;
    public:
    void set_roi(region_of_interest r){
     this->roi = r;   
    }
    region_of_interest get_roi(){
    return this->roi;
    }
    void set_class_id(int id){
        this->class_id = id;
    }
    int get_class_id(){
        return this->class_id;
    }
};


/**
 * To draw rectangle where objects are detected.
 * @param[out] image input mat on which rectangles are drawns
*/
static inline void draw_rectangles(cv::Mat &image , torch::TensorAccessor <float, 2> results)
{
    for (int i = 0; i < results.size(0) ; i++)
    {
        cv::rectangle(image, cv::Point(results[i][1], results[i][2]), cv::Point(results[i][3], results[i][4]), cv::Scalar(0, 0, 255), 1, 1, 0);
        string text = std::to_string( results[i][7]);
        cv::putText(image , text , cv::Point(results[i][1],results[i][2]), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200,200,250), 1, cv::LINE_AA );
    }
}
/**
 * @breif returns list object attributes
*/
static inline std::list<object_attributes> list_of_obj_attributes( torch::TensorAccessor <float, 2> results)
{   
    std::list <object_attributes> object_attributes_list;
    std::list<object_attributes>::iterator iter;
    iter = object_attributes_list.begin();
    for (int i = 0; i < results.size(0) ; i++)
    {
        string id = std::to_string(results[i][7]);
        int value = std::stoi(id); 
        region_of_interest roi(results[i][1] , results[i][2] , results[i][3] , results[i][4]);
        object_attributes obj;
        obj.set_class_id(value);
        obj.set_roi(roi);
        object_attributes_list.insert(iter,obj);
        iter++;
    }
    return object_attributes_list;
}
/**
 * Convert realsense frame to cv::Mat of corresponding type
 * @param[in] frame realsense frames which has to be converted into Mat
 * @returns cv::Mat of Type CV_8UC3 for RS2_FORMAT_BGR8
 * @returns cv::Mat of Type CV_8UC3 for RS2_FORMAT_BGR8
 * @returns cv::Mat of Type CV_16UC1 for RS2_FORMAT_Z16
 * @returns cv::Mat of Type CV_8UC1 for RS2_FORMAT_Y8
 * @throws runtime exception if the rs frame format doesnot matches the above mentioned formats.
*/

static inline cv::Mat frame_to_mat(const rs2::frame& frame)
{
   
    auto video_frame = frame.as<rs2::video_frame>();

    const int w = video_frame.get_width();
    const int h = video_frame.get_height();

    if (frame.get_profile().format() == RS2_FORMAT_BGR8)
    {
        return cv::Mat( cv::Size(w, h), CV_8UC3, (void*)frame.get_data(), cv::Mat::AUTO_STEP);
    }
    else if (frame.get_profile().format() == RS2_FORMAT_RGB8)
    {
        auto r = cv::Mat( cv::Size(w, h), CV_8UC3, (void*)frame.get_data(), cv::Mat::AUTO_STEP);
        cvtColor(r, r, cv::COLOR_RGB2BGR);
        return r;
    }
    else if (frame.get_profile().format() == RS2_FORMAT_Z16)
    {
        return cv::Mat(cv::Size(w, h), CV_16UC1, (void*)frame.get_data(), cv::Mat::AUTO_STEP);
    }
    else if (frame.get_profile().format() == RS2_FORMAT_Y8)
    {
        return cv::Mat(cv::Size(w, h), CV_8UC1, (void*)frame.get_data(), cv::Mat::AUTO_STEP);
    }
//    throw std::runtime_error("");
}
/**
 * Intersection over union is a formula to calculate overlap between two rectangles
 * we will use this to measure the accuracy of our trackers,the funtion will return 
 * float between (0-1) which will represent the accuracy , if the intersection area is 
 * zero it will simply return zero.
 * @param[in] bbox_1 coordinates of first region of interest
 * @param[in] bbox_2 coordinates of second region of interest
 * @returns inter ratio of overlap between two regions of interest
 */
static inline float intersection_over_union(int* bbox_1, int* bbox_2)
{
    int xA = max(bbox_1[0] , bbox_2[0]);
    int yA = max(bbox_1[1] , bbox_1[1]);
    int xB = min(bbox_1[2] , bbox_2[2]);
    int yB = min(bbox_1[3] , bbox_2[3]);
    //Intersection Area
    int intersection_Area = max(0 , xB - xA + 1) * max(0 , yB - yA + 1);

    float box_1_area = (bbox_1[2] - bbox_1[0] + 1) * (bbox_1[3] - bbox_1[1] + 1);
    float box_2_area = (bbox_2[2] - bbox_2[0] + 1) * (bbox_2[3] - bbox_2[1] + 1); 
    //Resultant intersection over union formula output this will be in range of (0-1)
    float inter = intersection_Area / float(box_1_area + box_2_area - intersection_Area);

    return inter;
}


// trim from start (in place)
static inline void ltrim(std::string &s) 
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch)
    {
        return !std::isspace(ch);
    })
    );
}

// trim from end (in place)
static inline void rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch)
    {
        return !std::isspace(ch);
    }).base(), s.end());
}

/**
 * 
*/
static inline void trim(std::string &s)
{
    ltrim(s);
    rtrim(s);
}
/**
 * 
*/
static inline int split(const string& str, std::vector<string>& ret_, string sep = ",")
{
    if (str.empty())
    {
        return 0;
    }

    string tmp;
    string::size_type pos_begin = str.find_first_not_of(sep);
    string::size_type comma_pos = 0;

    while (pos_begin != string::npos)
    {
        comma_pos = str.find(sep, pos_begin);
        if (comma_pos != string::npos)
        {
            tmp = str.substr(pos_begin, comma_pos - pos_begin);
            pos_begin = comma_pos + sep.length();
        }
        else
        {
            tmp = str.substr(pos_begin);
            pos_begin = comma_pos;
        }

        if (!tmp.empty())
        {
        	trim(tmp);
            ret_.push_back(tmp);
            tmp.clear();
        }
    }
    return 0;
}
/**
 * 
*/
static inline int split(const string& str, std::vector<int>& ret_, string sep = ",")
{
	std::vector<string> tmp;
	split(str, tmp, sep);

	for(int i = 0; i < tmp.size(); i++)
	{
		ret_.push_back(std::stoi(tmp[i]));
	}
}

/**
 * 
 * 
 * 
 */ 
static inline torch::Tensor get_bbox_iou(torch::Tensor box1, torch::Tensor box2)
{
	// Get the coordinates of bounding boxes
    torch::Tensor b1_x1, b1_y1, b1_x2, b1_y2; 
    b1_x1 = box1.select(1, 0);
    b1_y1 = box1.select(1, 1);
    b1_x2 = box1.select(1, 2);
    b1_y2 = box1.select(1, 3);
    torch::Tensor b2_x1, b2_y1, b2_x2, b2_y2;
    b2_x1 = box2.select(1, 0);
    b2_y1 = box2.select(1, 1);
    b2_x2 = box2.select(1, 2);
    b2_y2 = box2.select(1, 3);
    
    // et the corrdinates of the intersection rectangle
    torch::Tensor inter_rect_x1 =  torch::max(b1_x1, b2_x1);
    torch::Tensor inter_rect_y1 =  torch::max(b1_y1, b2_y1);
    torch::Tensor inter_rect_x2 =  torch::min(b1_x2, b2_x2);
    torch::Tensor inter_rect_y2 =  torch::min(b1_y2, b2_y2);
    
    // Intersection area
    torch::Tensor inter_area = torch::max(inter_rect_x2 - inter_rect_x1 + 1,torch::zeros(inter_rect_x2.sizes()))*torch::max(inter_rect_y2 - inter_rect_y1 + 1, torch::zeros(inter_rect_x2.sizes()));
    
    // Union Area
    torch::Tensor b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1);
    torch::Tensor b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1);
    
    torch::Tensor iou = inter_area / (b1_area + b2_area - inter_area);
    
    return iou;
}    
/**
 *
 *  
 */   
static inline void non_maximum_suppression()
{

}
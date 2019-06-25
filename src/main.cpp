// -*- lsst-c++ -*-
/**
 *Author: Usama Hasan
*/
#include "pipeline.hpp"
#include <mqtt/topic.h>
#include <mqtt/async_client.h>
#include "mqtt-client.hpp"
//Enter Address here
const std::string DFLT_SERVER_ADDRESS	{ "tcp://localhost:1883" };
//
const std::string DFLT_CLIENT_ID		{ "async_publish" };
//TOPIC
const string TOPIC { "" };
//Payload To publish message when item is inserted 
const char* PAYLOAD1 = "Item Added";
//Payload To publist message when item is removed
const char* PAYLOAD2 = "Item Removed";
//Publish message on connect
const char* PAYLOAD = "Connected" ;

const int  QOS = 1;

const auto TIMEOUT = std::chrono::seconds(10);

int main()

{
        
    Pipeline pipeline;
    //Insert path to your bag file here
    std::string bag_file = "/home/usamahasan/tracking/C++/bag_files/82921207023520190501_142849.bag";
    //Insert path to your YOLO cfg here
    char* cfg_file = "/home/usamahasan/tracking/yolofiles/yolov3.cfg";
    //Inset path to weight file here
    char* weight_file = "/home/usamahasan/tracking/yolofiles/final.weight";
    
    mqtt::async_client client(DFLT_SERVER_ADDRESS  ,DFLT_CLIENT_ID);
    
    mqtt::connect_options connect_opts;

    callback call_back;
    //Set call_back when connection fails.
    client.set_callback(call_back);
    //Publish message on connection with client
    mqtt::message on_connet_msg (TOPIC , PAYLOAD , 1 , true);

    mqtt::will_options will(on_connet_msg);

    connect_opts.set_will(will);

    while(True)
    {
        try{

            mqtt::token_ptr connect_token = client.connect(connect_opts);
        
            connect_token->wait();
        
            pipeline.pipe(cfg_file , weight_file , bag_file);

        }
    }
    //Read the documentaion of the function

}
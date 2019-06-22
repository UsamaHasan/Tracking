// -*- lsst-c++ -*-
/**
 * Author : Usama Hasan
*/
#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>

using namespace std;
/**
 * Declare A struct Of type nn::Module
*/
struct NeuralNet : torch::nn::Module {

public:
	NeuralNet();
	/**
	 * Overloaded constructor for class NeuralNet 
	*/
	NeuralNet(const char *conf_file, torch::Device *device);
	/**
	 * 
	*/
	map<string, string>* get_net_info();
	/**
	 * 
	*/
	void load_weights(const char *weight_file);
	/**
	 * 
	*/
	torch::Tensor forward(torch::Tensor x);
	/**
	 * 
	*/
	torch::Tensor write_results(torch::Tensor prediction, int num_classes, float confidence, float nms_conf );

private:

	torch::Device *_device;

	//This contains all the layers name in model!
	vector<map<string, string>> blocks;
	//
	torch::nn::Sequential features;
	
	//To store layers info
	vector<torch::nn::Sequential> module_list;

	/**
	 * the function will parse yolo.cfg file and store the layers name in vector this->blocks.
	*/
    void load_cfg(const char *cfg_file);
	/**
	 * This will create the respective YOLO model layers in libTorch and will store 
	 * them in the NeuralNet::module_list<torch::nn::sequential> container
	*/
    void create_modules();
	/**
	 *
	*/
    int get_integer_from_cfg(map<string, string> block, string key, int default_value);
	/**
	 * 
	*/
    string get_string_from_cfg(map<string, string> block, string key, string default_value);
};

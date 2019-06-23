// -*- lsst-c++ -*-

/**
 * This is an opensource implementation of Yolov3 in Libtorch, most of it has not been 
 * changed for the desired use.
 * 
*/
/**
 * 
*/
#include "NeuralNet.h"
#include <stdio.h>
#include <iostream>
#include <typeinfo>
#include "helpers.hpp"


int NeuralNet::get_integer_from_cfg(map<string, string> block, string key, int default_value)
{
	if ( block.find(key) != block.end() ) 
	{
		return std::stoi(block.at(key));
	}
	return default_value;
}

string NeuralNet::get_string_from_cfg(map<string, string> block, string key, string default_value)
{
	if ( block.find(key) != block.end() ) 
	{
		return block.at(key);
	}
	return default_value;
}

torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                          int64_t stride, int64_t padding, int64_t groups, bool with_bias=false){
    torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
    conv_options.stride_ = stride;
    conv_options.padding_ = padding;
    conv_options.groups_ = groups;
    conv_options.with_bias_ = with_bias;
    return conv_options;
}

torch::nn::BatchNormOptions bn_options(int64_t features){
    torch::nn::BatchNormOptions bn_options = torch::nn::BatchNormOptions(features);
    bn_options.affine_ = true;
    bn_options.stateful_ = true;
    return bn_options;
}

struct EmptyLayer : torch::nn::Module
{
    EmptyLayer(){
        
    }

    torch::Tensor forward(torch::Tensor x) {
        return x; 
    }
};

struct UpsampleLayer : torch::nn::Module
{
	int _stride;
    UpsampleLayer(int stride){
        _stride = stride;
    }

    torch::Tensor forward(torch::Tensor x) {

    	torch::IntList sizes = x.sizes();

    	int64_t w, h;

    	if (sizes.size() == 4)
    	{
    		w = sizes[2] * _stride;
    		h = sizes[3] * _stride;

			x = torch::upsample_nearest2d(x, {w, h});
    	}
    	else if (sizes.size() == 3)
    	{
			w = sizes[2] * _stride;
			x = torch::upsample_nearest1d(x, {w});
    	}   	
        return x; 
    }
};

struct MaxPoolLayer2D : torch::nn::Module
{
	int _kernel_size;
	int _stride;
    MaxPoolLayer2D(int kernel_size, int stride){
        _kernel_size = kernel_size;
        _stride = stride;
    }

    torch::Tensor forward(torch::Tensor x) {	
    	if (_stride != 1)
    	{
    		x = torch::max_pool2d(x, {_kernel_size, _kernel_size}, {_stride, _stride});
    	}
    	else
    	{
    		int pad = _kernel_size - 1;

       		torch::Tensor padded_x = torch::replication_pad2d(x, {0, pad, 0, pad});
    		x = torch::max_pool2d(padded_x, {_kernel_size, _kernel_size}, {_stride, _stride});
    	}       

        return x;
    }
};

struct DetectionLayer : torch::nn::Module
{
	vector<float> _anchors;

    DetectionLayer(vector<float> anchors)
    {
        _anchors = anchors;
    }
    
    torch::Tensor forward(torch::Tensor predictions, int inp_dim, int num_classes, torch::Device device)
    {
    	return predict_transform(predictions, inp_dim, _anchors, num_classes, device);
    }

    torch::Tensor predict_transform(torch::Tensor predictions, int inp_dim, vector<float> anchors, int num_classes, torch::Device device)
    {
    	int no_of_detected_objects = predictions.size(0);
    	int stride = floor(inp_dim / predictions.size(2));
    	int grid_size = floor(inp_dim / stride);
    	int bregion_of_interestttrs = 5 + num_classes;
    	int num_anchors = anchors.size()/2;

    	for (int i = 0; i < anchors.size(); i++)
    	{
    		anchors[i] = anchors[i]/stride;
    	}
    	torch::Tensor result = predictions.view({no_of_detected_objects, bregion_of_interestttrs * num_anchors, grid_size * grid_size});
    	result = result.transpose(1,2).contiguous();
    	result = result.view({no_of_detected_objects, grid_size*grid_size*num_anchors, bregion_of_interestttrs});
    	
    	result.select(2, 0).sigmoid_();
        result.select(2, 1).sigmoid_();
        result.select(2, 4).sigmoid_();

        auto grid_len = torch::arange(grid_size);

        std::vector<torch::Tensor> args = torch::meshgrid({grid_len, grid_len});

        torch::Tensor x_offset = args[1].contiguous().view({-1, 1});
        torch::Tensor y_offset = args[0].contiguous().view({-1, 1});

        // std::cout << "x_offset:" << x_offset << endl;
        // std::cout << "y_offset:" << y_offset << endl;

        x_offset = x_offset.to(device);
        y_offset = y_offset.to(device);

        auto x_y_offset = torch::cat({x_offset, y_offset}, 1).repeat({1, num_anchors}).view({-1, 2}).unsqueeze(0);
        result.slice(2, 0, 2).add_(x_y_offset);

        torch::Tensor anchors_tensor = torch::from_blob(anchors.data(), {num_anchors, 2});
        //if (device != nullptr)
        	anchors_tensor = anchors_tensor.to(device);
        anchors_tensor = anchors_tensor.repeat({grid_size*grid_size, 1}).unsqueeze(0);

        result.slice(2, 2, 4).exp_().mul_(anchors_tensor);
        result.slice(2, 5, 5 + num_classes).sigmoid_();
   		result.slice(2, 0, 4).mul_(stride);

    	return result;
    }
};

NeuralNet::NeuralNet(const char *cfg_file, torch::Device *device) {

	load_cfg(cfg_file);

	_device = device;

	create_modules();
}

void NeuralNet::load_cfg(const char *cfg_file)
{
	ifstream fs(cfg_file);
	string line;
 
	if(!fs) 
	{
		std::cout << "Fail to load cfg file:" << cfg_file << endl;
		return;
	}

	while (getline (fs, line))
	{ 
		trim(line);

		if (line.empty())
		{
			continue;
		}		

		if ( line.substr (0,1)  == "[")
		{
			map<string, string> block;			

			string key = line.substr(1, line.length() -2);
			block["type"] = key;  
			this->blocks.push_back(block);
		}
		else
		{
			map<string, string> *block = &this->blocks[ this->blocks.size() -1];

			vector<string> op_info;

			split(line, op_info, "=");

			if (op_info.size() == 2)
			{
				string p_key = op_info[0];
				string p_value = op_info[1];
				block->operator[](p_key) = p_value;
			}			
		}				
	}
	fs.close();
}

void NeuralNet::create_modules()
{
	int prev_filters = 3;

	std::vector<int> output_filters;

	int index = 0;

	int filters = 0;

	for (int i = 0, len = this->blocks.size(); i < len; i++)
	{
		map<string, string> block = this->blocks[i];
		
		//Contains the current layer name
		string layer_type = block["type"];

		torch::nn::Sequential module;

		if (layer_type == "net")
			continue;

		if (layer_type == "convolutional")
		{

			/**
			 * Load all the information of the respective convolutional
			 * Activation layer type
			 * batch Normailize or not
			 * No of Filters
			 * Kernel_size
			 * Stride
			 * zero padding or not 
			 * bias 
			 */
			string activation = get_string_from_cfg(block, "activation", "");
			int batch_normalize = get_integer_from_cfg(block, "batch_normalize", 0);
			filters = get_integer_from_cfg(block, "filters", 0);
			int padding = get_integer_from_cfg(block, "pad", 0);
			int kernel_size = get_integer_from_cfg(block, "size", 0);
			int stride = get_integer_from_cfg(block, "stride", 1);

			int pad = padding > 0?  (kernel_size -1)/2: 0;
			bool with_bias = batch_normalize > 0? false : true;

			torch::nn::Conv2d conv = torch::nn::Conv2d(conv_options(prev_filters, filters, kernel_size, stride, pad, 1, with_bias));
			module->push_back(conv);

			if (batch_normalize > 0)
			{
				torch::nn::BatchNorm bn = torch::nn::BatchNorm(bn_options(filters));
                module->push_back(bn);
			}

			if (activation == "leaky")
			{
				module->push_back(torch::nn::Functional(torch::leaky_relu, /*slope=*/0.1));
			}			
		}
		else if (layer_type == "upsample")
		{
			/**
			 * [route] layer - is the same as Concat-layer in the Caffelayers =- 1, -4
			 *  means that will be concatenated two layers, with relative indexies -1 and -4
			*/

			int stride = get_integer_from_cfg(block, "stride", 1);

			UpsampleLayer uplayer(stride);
			module->push_back(uplayer);
		}
		else if (layer_type == "maxpool")
		{
			///
			int stride = get_integer_from_cfg(block, "stride", 1);
			int size = get_integer_from_cfg(block, "size", 1);

			MaxPoolLayer2D poolLayer(size, stride);
			module->push_back(poolLayer);
		}
		else if (layer_type == "shortcut")
		{
			//
			int from = get_integer_from_cfg(block, "from", 0);
			block["from"] = std::to_string(from);

			blocks[i] = block;

			// placeholder
			EmptyLayer layer;
			module->push_back(layer);
		}
		else if (layer_type == "route")
		{
			//
			string layers_info = get_string_from_cfg(block, "layers", "");

			std::vector<string> layers;
			split(layers_info, layers, ",");

			std::string::size_type sz; 
			signed int start = std::stoi(layers[0], &sz);
			signed int end = 0;

			if (layers.size() > 1)
			{
				end = std::stoi(layers[1], &sz);
			}

			if (start > 0)	start = start - index;

			if (end > 0) end = end - index;

			block["start"] = std::to_string(start);
			block["end"] = std::to_string(end);

			blocks[i] = block;

			// placeholder
			EmptyLayer layer;
			module->push_back(layer);

			if (end < 0)
			{
				filters = output_filters[index + start] + output_filters[index + end];
			}
			else
			{
				filters = output_filters[index + start];
			}
		}
		else if (layer_type == "yolo")
		{
			string mask_info = get_string_from_cfg(block, "mask", "");
			std::vector<int> masks;
			split(mask_info, masks, ",");

			string anchor_info = get_string_from_cfg(block, "anchors", "");
			std::vector<int> anchors;
			split(anchor_info, anchors, ",");

			std::vector<float> anchor_points;
			int pos;
			for (int i = 0; i< masks.size(); i++)
			{
				pos = masks[i];
				anchor_points.push_back(anchors[pos * 2]);
				anchor_points.push_back(anchors[pos * 2+1]);
			}

			DetectionLayer layer(anchor_points);
			module->push_back(layer);
		}
		else
		{
			cout << "unsupported operator:" << layer_type << endl;
		}

		prev_filters = filters;
        output_filters.push_back(filters);
        module_list.push_back(module);

        char *module_key = new char[strlen("layer_") + sizeof(index) + 1];

        sprintf(module_key, "%s%d", "layer_", index);

        register_module(module_key, module);

        index += 1;
	}
}

map<string, string>* NeuralNet::get_net_info()
{
	if (blocks.size() > 0)
	{
		return &blocks[0];
	}
}

void NeuralNet::load_weights(const char *weight_file)
{
	ifstream fs(weight_file, ios::binary);

	// header info: 5 * int32_t
	int32_t header_size = sizeof(int32_t)*5;

	int64_t index_weight = 0;

	fs.seekg (0, fs.end);
    int64_t length = fs.tellg();
    // skip header
    length = length - header_size;

    fs.seekg (header_size, fs.beg);
    float *weights_src = (float *)malloc(length);
    fs.read(reinterpret_cast<char*>(weights_src), length);

    fs.close();

    at::TensorOptions options= torch::TensorOptions()
        .dtype(torch::kFloat32)
        .is_variable(true);
    at::Tensor weights = torch::CPU(torch::kFloat32).tensorFromBlob(weights_src, {length/4});

	for (int i = 0; i < module_list.size(); i++)
	{
		map<string, string> module_info = blocks[i + 1];

		string module_type = module_info["type"];

		// only conv layer need to load weight
		if (module_type != "convolutional")	continue;
		
		torch::nn::Sequential seq_module = module_list[i];

		auto conv_module = seq_module.ptr()->ptr(0);
		torch::nn::Conv2dImpl *conv_imp = dynamic_cast<torch::nn::Conv2dImpl *>(conv_module.get());

		int batch_normalize = get_integer_from_cfg(module_info, "batch_normalize", 0);

		if (batch_normalize > 0)
		{
			// second module
			auto bn_module = seq_module.ptr()->ptr(1);

			torch::nn::BatchNormImpl *bn_imp = dynamic_cast<torch::nn::BatchNormImpl *>(bn_module.get());

			int num_bn_biases = bn_imp->bias.numel();

			at::Tensor bn_bias = weights.slice(0, index_weight, index_weight + num_bn_biases);
			index_weight += num_bn_biases;
	
			at::Tensor bn_weights = weights.slice(0, index_weight, index_weight + num_bn_biases);
			index_weight += num_bn_biases;

			at::Tensor bn_running_mean = weights.slice(0, index_weight, index_weight + num_bn_biases);
			index_weight += num_bn_biases;

			at::Tensor bn_running_var = weights.slice(0, index_weight, index_weight + num_bn_biases);
			index_weight += num_bn_biases;

			bn_bias = bn_bias.view_as(bn_imp->bias);
			bn_weights = bn_weights.view_as(bn_imp->weight);
			bn_running_mean = bn_running_mean.view_as(bn_imp->running_mean);
			bn_running_var = bn_running_var.view_as(bn_imp->running_variance);

			bn_imp->bias.set_data(bn_bias);
			bn_imp->weight.set_data(bn_weights);
			bn_imp->running_mean.set_data(bn_running_mean);
			bn_imp->running_variance.set_data(bn_running_var);
		}
		else
		{
			int num_conv_biases = conv_imp->bias.numel();

			at::Tensor conv_bias = weights.slice(0, index_weight, index_weight + num_conv_biases);
			index_weight += num_conv_biases;

			conv_bias = conv_bias.view_as(conv_imp->bias);
			conv_imp->bias.set_data(conv_bias);
		}		

		int num_weights = conv_imp->weight.numel();
	
		at::Tensor conv_weights = weights.slice(0, index_weight, index_weight + num_weights);
		index_weight += num_weights;	

		conv_weights = conv_weights.view_as(conv_imp->weight);
		conv_imp->weight.set_data(conv_weights);
	}
}

torch::Tensor NeuralNet::forward(torch::Tensor x) 
{
	int module_count = module_list.size();

	//Create a vector which will store outputs of all layers
	std::vector<torch::Tensor> outputs(module_count);

	//Tensor to store predictions
	torch::Tensor result;

	int write = 0;

	for (int i = 0; i < module_count; i++)
	{
		
		map<string, string> block = this->blocks[i+1];
		string layer_type = block["type"];

		if (layer_type == "net")
			continue;

		if (layer_type == "convolutional" || layer_type == "upsample" || layer_type == "maxpool")
		{
			torch::nn::SequentialImpl *seq_imp = dynamic_cast<torch::nn::SequentialImpl *>(module_list[i].ptr().get());
			
			x = seq_imp->forward(x);
			outputs[i] = x;
		}
		else if (layer_type == "route")
		{
			int start = std::stoi(block["start"]);
			int end = std::stoi(block["end"]);

			if (start > 0)
			{ 
				start = start - i;
			}

			if (end == 0)
			{
				x = outputs[i + start];
			}
			else
			{
				if (end > 0) end = end - i;

				torch::Tensor map_1 = outputs[i + start];
				torch::Tensor map_2 = outputs[i + end];

				x = torch::cat({map_1, map_2}, 1);
			}

			outputs[i] = x;
		}
		else if (layer_type == "shortcut")
		{
			int from = std::stoi(block["from"]);
			x = outputs[i-1] + outputs[i+from];
            outputs[i] = x;
		}
		else if (layer_type == "yolo")
		{
			torch::nn::SequentialImpl *seq_imp = dynamic_cast<torch::nn::SequentialImpl *>(module_list[i].ptr().get());

			map<string, string> net_info = blocks[0];
			int inp_dim = get_integer_from_cfg(net_info, "height", 0);
			int num_classes = get_integer_from_cfg(block, "classes", 0);

			x = seq_imp->forward(x, inp_dim, num_classes, *_device);

			if (write == 0)
			{
				result = x;
				write = 1;
			}
			else
			{
				result = torch::cat({result,x}, 1);
			}

			outputs[i] = outputs[i-1];
		}
	}
	return result;
}

torch::Tensor NeuralNet::write_results(torch::Tensor predictions, int num_classes, float confidence, float nms_conf)
{
	//A tensor containing zeros on entries where result has confidence below the given threshold
	auto confidence_check_mask = (predictions.select(2,4) > confidence).to(torch::kFloat32).unsqueeze(2);
	
	//By multiplying the confidence_check_mask with the prediction tensor will create zeros on the results below the confidence threshold
	predictions.mul_(confidence_check_mask);
	
	//This will contain index of all non-zero entires in our prediction tensor
	auto filtered_index_list = torch::nonzero(predictions.select(2, 4)).transpose(0, 1).contiguous();	

	//if the filtered_index_list is empty return a tensor with 0 output.
	if (filtered_index_list.size(0) == 0) 
	{
        return torch::zeros({0});
    }
	// Create a tensor to store the values of region of interest
	torch::Tensor region_of_interest = torch::ones(predictions.sizes(), predictions.options());
	
	/**
	 * at::tensor.select function documentation
	 * Returns a new Tensor which is a tensor slice at the given index in the dimension dim.
	 * The returned tensor has one less dimension: the dimension dim is removed.
	 * As a result, it is not possible to select() on a 1D tensor. 
	*/


	// top left x = centerX - w/2
	region_of_interest.select(2, 0) = predictions.select(2, 0) - predictions.select(2, 2).div(2);
	region_of_interest.select(2, 1) = predictions.select(2, 1) - predictions.select(2, 3).div(2);
	region_of_interest.select(2, 2) = predictions.select(2, 0) + predictions.select(2, 2).div(2);
	region_of_interest.select(2, 3) = predictions.select(2, 1) + predictions.select(2, 3).div(2);


    predictions.slice(2, 0, 4) = region_of_interest.slice(2, 0, 4);

	//No of objects detected by the model
    int no_of_detected_objects = predictions.size(0);
    
	//Number of items in each result i.e object class , x_1 , y_1 , x_2 , y_2
	int elements = 5;

	//@returns tensor container to return output
    torch::Tensor output = torch::ones({1, predictions.size(2) + 1});
    
	bool write = false;

    int num = 0;

	//The loop will have iteration to the no_of_detected_objects  
    for (int i = 0; i < no_of_detected_objects; i++)
    {
    	auto object_result = predictions[i];
    	// get the max classes score at each result
    	std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(object_result.slice(1, elements, elements + num_classes), 1);

    	// class score
    	auto max_conf = std::get<0>(max_classes);

    	// index
    	auto max_conf_score = std::get<1>(max_classes);
    	
		max_conf = max_conf.to(torch::kFloat32).unsqueeze(1);
    	
		max_conf_score = max_conf_score.to(torch::kFloat32).unsqueeze(1);

    	// shape: n * 7, left x, left y, right x, right y, object confidence, class_score, class_id
    	object_result = torch::cat({object_result.slice(1, 0, 5), max_conf, max_conf_score}, 1);
    	
    	// remove item which object confidence == 0
        auto non_zero_index =  torch::nonzero(object_result.select(1,4));
        
		auto object_result_data = object_result.index_select(0, non_zero_index.squeeze()).view({-1, 7});

        // get unique classes 
        std::vector<torch::Tensor> img_classes;

	    for (int m = 0, len = object_result_data.size(0); m < len; m++) 
	    {
	    	bool found = false;	        
	        for (int n = 0; n < img_classes.size(); n++)
	        {
	        	auto ret = (object_result_data[m][6] == img_classes[n]);
	        	if (torch::nonzero(ret).size(0) > 0)
	        	{
	        		found = true;
	        		break;
	        	}
	        }
	        if (!found) img_classes.push_back(object_result_data[m][6]);
	    }

        for (int k = 0; k < img_classes.size(); k++)
        {
        	auto cls = img_classes[k];

        	auto cls_mask = object_result_data * (object_result_data.select(1, 6) == cls).to(torch::kFloat32).unsqueeze(1);
        	auto class_mask_index =  torch::nonzero(cls_mask.select(1, 5)).squeeze();

        	auto image_pred_class = object_result_data.index_select(0, class_mask_index).view({-1,7});
        	// ascend by confidence
        	// seems that inverse method not work
        	std::tuple<torch::Tensor,torch::Tensor> sort_ret = torch::sort(image_pred_class.select(1,4));

        	auto conf_sort_index = std::get<1>(sort_ret);
        	
        	// seems that there is something wrong with inverse method
        	// conf_sort_index = conf_sort_index.inverse();

        	image_pred_class = image_pred_class.index_select(0, conf_sort_index.squeeze()).cpu();

           	for(int w = 0; w < image_pred_class.size(0)-1; w++)
        	{
        		int mi = image_pred_class.size(0) - 1 - w;

        		if (mi <= 0)
        		{
        			break;
        		}

        		auto ious = get_bbox_iou(image_pred_class[mi].unsqueeze(0), image_pred_class.slice(0, 0, mi));

        		auto iou_mask = (ious < nms_conf).to(torch::kFloat32).unsqueeze(1);
        		image_pred_class.slice(0, 0, mi) = image_pred_class.slice(0, 0, mi) * iou_mask;

        		// remove from list
        		auto non_zero_index = torch::nonzero(image_pred_class.select(1,4)).squeeze();
        		image_pred_class = image_pred_class.index_select(0, non_zero_index).view({-1,7});
			}
			
        	torch::Tensor batch_index = torch::ones({image_pred_class.size(0), 1}).fill_(i);
			std::cout << "Batch_Index:" << batch_index << std::endl;
			std::cout << "Image Prediction class" << image_pred_class << std::endl;
        	if (!write)
        	{
        		output = torch::cat({batch_index, image_pred_class}, 1);
				write = true;
        	}
        	else
        	{
        		auto out = torch::cat({batch_index, image_pred_class}, 1);
        		output = torch::cat({output,out}, 0);
        	}
			num += 1;
        }
    }

    if (num == 0)
    {
    	return torch::zeros({0});
    }

    return output;
}

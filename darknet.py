from __future__ import division
from util import * 

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

def get_test_input():
    img = cv2.imread("imgs/dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

def parse_cfg(cfgfile):
	"""
	Takes a configuration file

	Returns a list of blocks. Each blocks describes a block in the neural
	network to be built. Block is represented as a dictionary in the list

	"""
	file = open(cfgfile, 'r')
	lines = file.read().split('\n')                        # store the lines in a list
	lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
	lines = [x for x in lines if x[0] != '#']              # get rid of comments
	lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces

	block = {}
	blocks = []

	for line in lines:
	    if line[0] == "[":               # This marks the start of a new block
	        if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
	            blocks.append(block)     # add it the blocks list
	            block = {}               # re-init the block
	        block["type"] = line[1:-1].rstrip()     
	    else:
	        key,value = line.split("=") 
	        block[key.rstrip()] = value.lstrip()
	blocks.append(block)

	return blocks

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

class MaxPoolStride1(nn.Module):
	def __init__(self, kernel_size):
		super(MaxPoolStride1, self).__init__()
		self.kernel_size = kernel_size
		self.pad = kernel_size - 1

	def forward(self, x):
		print("\n\nBefore PAD x.shape = ", x.shape)
		padded_x = F.pad(x, (0,self.pad,0,self.pad), mode="replicate")
		print("\nAFter PAD x.shape = ", padded_x.shape)
		pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
		return pooled_x

def create_modules(blocks):
	net_info = blocks[0]     #Captures the information about the input and pre-processing    
	module_list = nn.ModuleList()
	prev_filters = 3 # to keep track of prev feature map depth. Initially is 3 (RGB)
	output_filters = [] # to know the feature map depth in the route layer (concatenate outputs of different layers)

	for index, x in enumerate(blocks[1:]):
		module = nn.Sequential()

	    #check the type of block
	    #create a new module for the block
	    #append to module_list

		if(x['type'] == 'convolutional'):
			activation = x['activation']
			try:
				batch_norm = int(x['batch_normalize'])
				bias = False
			except:
				batch_norm = 0
				bias = True

			filters = int(x['filters'])
			padding = int(x['pad'])
			kernel_size = int(x['size'])
			stride = int(x['stride'])

			if (padding):
				pad = (kernel_size - 1) // 2
			else:
				pad = 0

			conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
			module.add_module('conv{0}'.format(index), conv)

			if (batch_norm):
				bn = nn.BatchNorm2d(filters)
				module.add_module('batch_norm_{0}'.format(index), bn)

			if (activation == 'leaky'):
				activ = nn.LeakyReLU(0.1, inplace = True)
				module.add_module('leaky_{0}'.format(index), activ)
			# Tiny model
			elif (activation == 'linear'):
				activ = nn.ReLU(inplace = True)
				module.add_module('ReLU_{0}'.format(index), activ)
		# Tiny model
		elif (x['type'] == 'maxpool'):
			kernel = int(x['size'])
			stride = int(x['stride'])
			if (stride != 1):
				maxpool = nn.MaxPool2d(kernel, stride = stride)
			else:
				maxpool = MaxPoolStride1(kernel)
			module.add_module('maxpool_{0}'.format(index), maxpool)

		elif (x['type'] == 'upsample'):
			#upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')
			upsample = EmptyLayer()
			module.add_module('upsample_{0}'.format(index), upsample)

		elif (x['type'] == 'route'):
			x['layers'] = x['layers'].split(',')
			start = int(x['layers'][0])
			try:
				end = int(x['layers'][1])
			except:
				end = 0
			if(start > 0):
				start = start - index
			if (end > 0):
				end = end - index

			route = EmptyLayer()
			module.add_module('route_{0}'.format(index), route)

			if (end < 0):
				filters = output_filters[start + index] + output_filters[end + index]
			else: #if there is only one parameter for the route module in the .cfg
				filters = output_filters[start + index]

		elif (x['type'] == 'shortcut'):
			shortcut = EmptyLayer()
			module.add_module('shortcut_{0}'.format(index), shortcut)

		elif (x['type'] == 'yolo'):
			mask = x['mask'].split(',')
			mask = [int(m) for m in mask]

			anchors = x['anchors'].split(',')
			anchors = [int(a) for a in anchors]

			anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
			anchors = [anchors[i] for i in mask]

			detection = DetectionLayer(anchors)
			module.add_module('Detection_{}'.format(index), detection)

		module_list.append(module)
		prev_filters = filters
		output_filters.append(filters)

	return (net_info, module_list)


class Darknet(nn.Module):
	def __init__(self, cfgfile):
		super(Darknet, self).__init__()
		self.blocks = parse_cfg(cfgfile)
		self.net_info, self.module_list = create_modules(self.blocks)

	def forward(self, x, CUDA):
		modules = self.blocks[1:]
		outputs = {} # for the route layers

		# use it to indicate that at least one prediction is ready so we can concatenate
		# output feature maps of different spatial scale (13x13 + 26x26 + 52x52)
		write = 0

		for i, module in enumerate(modules):
			module_type = (module['type'])

			#print(i, module_type, '\n')

			if (module_type == 'convolutional' or module_type == 'maxpool'):
				print("module # ", i, module_type)
				print("Before x.shape ", x.shape)
				x = self.module_list[i](x)
				print("After x.shape ", x.shape)

			elif (module_type == 'upsample'):
				print("module # ", i, module_type)
				print("Before Upsamle: x.shape = ", x.shape)
				x = F.interpolate(x, scale_factor = 2, mode='bilinear', align_corners=True)
				print("After Upsamle: x.shape = ", x.shape)

			elif (module_type == 'route'):
				print("module # ", i, module_type)
				layers = [int(layer) for layer in module['layers']]
				print("layers: ", layers)

				if (layers[0] > 0):
					layers[0] = layers[0] - i

				if (len(layers) == 1):
					x = outputs[i + layers[0]]
					print("x.shape = ", x.shape)

				else:
					if (layers[1] > 0):
						layers[1] = layers[1] - i

					map1 = outputs[i + layers[0]]
					map2 = outputs[i + layers[1]]

					print("map1.shape  ", map1.shape, '\n', "map2.shape = ", map2.shape)

					x = torch.cat((map1, map2), 1)

			elif (module_type == 'shortcut'):
				from_ = int(module['from'])
				x = outputs[i - 1] + outputs[i + from_]

			elif (module_type == 'yolo'):
				print("module # ", i, module_type)
				anchors = self.module_list[i][0].anchors
				input_dim = int(self.net_info['height'])

				num_classes = int(module['classes'])

				x = x.data

				print("YOLO. x.shape = ", i, x.shape)
				x = predict_transform(x, input_dim, anchors, num_classes, CUDA)

				if not write:
					detections = x
					write = 1
				else:
					detections = torch.cat((detections, x), 1)

			outputs[i] = x

		return detections

	def load_weights(self, weight_file):
		# Only conv layers have weights.
		# For Conv + batch norm: bn_bias, bn_weight, bn_running_mean, bn_running_var, conv_weight
		# For just Conv: conv_bias, conv_weight
		# 
		fp = open(weight_file, "rb")

		# First 5 values (160 bit) contain header info
		# 1 - Major version number
		# 2 - Minor version number
		# 3 - Subversion number
		# 4,5 - Images seen by the network (during training)

		header = np.fromfile(fp, dtype=np.int32, count = 5)
		self.header = torch.from_numpy(header)
		self.seen = self.header[3]

		# the rest of the file are weights (float32)

		weights = np.fromfile(fp, dtype = np.float32)
		# To keep track of where we are in the weights file
		ptr = 0

		for i in range(len(self.module_list)):
			module_type = self.blocks[i + 1]['type']

			if (module_type == 'convolutional'):
				module = self.module_list[i]
				try:
					batch_norm = int(self.blocks[i + 1]['batch_normalize'])
				except:
					batch_norm = 0

				conv = module[0]

				# first load bn data
				if (batch_norm):
					bn = module[1]

					num_bn_biases = bn.bias.numel()

					bn_biases = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
					ptr += num_bn_biases

					bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
					ptr += num_bn_biases

					bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
					ptr += num_bn_biases

					bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
					ptr += num_bn_biases

					# Cast the loaded weights into dims of the module weights
					bn_biases = bn_biases.view_as(bn.bias.data)
					bn_weigths = bn_weights.view_as(bn.weight.data)
					bn_running_mean = bn_running_mean.view_as(bn.running_mean)
					bn_running_var = bn_running_var.view_as(bn.running_var)

					# Copy the data to the module
					bn.bias.data.copy_(bn_biases)
					bn.weight.data.copy_(bn_weights)
					bn.running_mean.copy_(bn_running_mean)
					bn.running_var.copy_(bn_running_var)

				# Conv biases are present if there is no BatchNorm
				else:
					num_conv_biases = conv.bias.numel()

					conv_biases = torch.from_numpy(weights[ptr: ptr + num_conv_biases])
					ptr += num_conv_biases

					conv_biases = conv_biases.view_as(conv.bias.data)

					conv.bias.data.copy_(conv_biases)

				# Load conv weights
				num_conv_weights = conv.weight.numel()

				conv_weights = torch.from_numpy(weights[ptr: ptr + num_conv_weights])
				ptr += num_conv_weights

				conv_weights = conv_weights.view_as(conv.weight.data)

				conv.weight.data.copy_(conv_weights)



"""model = Darknet("cfg/yolov3.cfg")
model.load_weights("model/yolov3.weights")
inp = get_test_input()
#print(model)
pred = model(inp, torch.cuda.is_available())
print(pred.shape, pred)

print("Pred shape before NMS: ", pred.shape)
pred = write_result(pred, 0.6, 80, 0.5)
if pred is not 0:
	print("Pred shape after NMS: ", pred.shape)
else:
	print("There is no detections")"""

model = Darknet("cfg/yolov3-tiny.cfg")
print(model)
model.load_weights("model/yolov3-tiny.weights")
inp = get_test_input()
#print(model)
pred = model(inp, torch.cuda.is_available())
print(pred.shape, pred)

print("Pred shape before NMS: ", pred.shape)
pred = write_result(pred, 0.6, 80, 0.5)
if pred is not 0:
	print("Pred shape after NMS: ", pred.shape)
else:
	print("There is no detections")
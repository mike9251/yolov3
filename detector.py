from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import random

def arg_parse():
	"""
	Parse arguements to the detect module

	"""

	parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

	parser.add_argument("--images", dest = 'images', help = 
	                    "Image / Directory containing images to perform detection upon",
	                    default = "imgs", type = str)
	parser.add_argument("--det", dest = 'det', help = 
	                    "Image / Directory to store detections to",
	                    default = "det", type = str)
	parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
	parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
	parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
	parser.add_argument("--cfg", dest = 'cfgfile', help = 
	                    "Config file",
	                    default = "cfg/yolov3.cfg", type = str)
	parser.add_argument("--weights", dest = 'weightsfile', help = 
	                    "weightsfile",
	                    default = "model/yolov3.weights", type = str)
	parser.add_argument("--reso", dest = 'reso', help = 
	                    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
	                    default = "416", type = str)

	return parser.parse_args()
    
args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes('data/coco.names')
#print(classes)

print("Loading the model...")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("SUCCESS\n")

model.net_info['height'] = args.reso
inp_dim = int(model.net_info['height'])
assert(inp_dim % 32 == 0)
assert(inp_dim > 0)

if CUDA:
	model.cuda()

# Set the model in evaluation mode
model.eval()


read_dir = time.time()

try:
	img_list = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
	img_list = []
	img_list.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
	print ("No file or directory with the name {}".format(images))
	exit()

# Create a folder for detections (if it doesn't exist)
if not os.path.exists(args.det):
	os.makedirs(args.det)

load_batch = time.time()
loaded_img = [cv2.imread(x) for x in img_list]
# Call prep_image for each image in loaded_img with inp_dim
img_batches = list(map(prep_image, loaded_img, [inp_dim for x in range(len(img_list))]))

#List containing dimensions of original images
img_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_img]
img_dim_list = torch.FloatTensor(img_dim_list).repeat(1, 2)

if CUDA:
	img_dim_list = img_dim_list.cuda()

leftover = 0
if (len(img_dim_list) % batch_size):
	leftover = 1
if (batch_size != 1):
	num_batches = len(img_list) // batch_size
	img_batches = [torch.cat((img_batches[i * batch_size: min((i +  1)*batch_size,
                       len(im_batches))]))  for i in range(num_batches)]

write = 0
start_det_loop = time.time()
for i, batch in enumerate(img_batches):
	start = time.time()
	if CUDA:
		batch = batch.cuda()

	#prediction = model(Variable(batch, volatile = True), CUDA)
	with torch.no_grad():
		prediction = model(Variable(batch), CUDA)

	prediction = write_result(prediction, confidence, num_classes, nms_thesh)

	end = time.time()
	# if there weren't any detections
	if type(prediction) == int:
		for img_num, image in enumerate(img_list[i * batch_size: min((i + 1) * batch_size, len(img_list))]):
			img_id = i * batch_size + img_num
			print("{0:20s} predicted in {1:6.3f} seconds".format(image.split('/')[-1], (end - start)/batch_size))
			print("{0:20s} {1:s}".format("Objects deteted: ", ""))
			print("-----------------------------------------------------------")
		continue
	# Map image indexes in the batch to indexes in img_list 
	prediction[:, 0] += i * batch_size

	if not write:
		output = prediction
		write = 1
	else:
		output = torch.cat((output, prediction))

	for img_num, image in enumerate(img_list[i * batch_size: min((i + 1) * batch_size, len(img_list))]):
		img_id = i * batch_size + img_num
		objects = [classes[int(x[-1])] for x in output if int(x[0]) == img_id]
		print("{0:20s} predicted in {1:6.3f} seconds".format(image.split('/')[-1], (end - start)/batch_size))
		print("{0:20s} {1:s}".format("Objects deteted: ", " ".join(objects)))
		print("-----------------------------------------------------------")

	if CUDA:
		torch.cuda.synchronize()


try:
	output
except:
	print("No detections!")
	exit()

output_recast = time.time()

# Get a tensor with elements from img_dim_list and indexes from output[:, 0]
img_dim_list = torch.index_select(img_dim_list, 0, output[:, 0].long())
print("Output: ", output)
print(img_dim_list.shape, '\n', img_dim_list)
# returns min value at each row (detection) in the dim. Use only first returned value [0], [1] - is index of min value
scaling_factor = torch.min(inp_dim/img_dim_list, 1)[0].view(-1, 1)
print("scale = ", scaling_factor)
print("x = ", output[:, 1], 'w = ', output[:, 3], 'y = ', output[:, 2], 'h = ', output[:, 4])
output[:, [1, 3]] -= (inp_dim - scaling_factor * img_dim_list[:, 0].view(-1, 1))/2
output[:, [2, 4]] -= (inp_dim - scaling_factor * img_dim_list[:, 1].view(-1, 1))/2



output[:, 1:5] /= scaling_factor
print("x = ", output[:, 1], 'w = ', output[:, 3], 'y = ', output[:, 2], 'h = ', output[:, 4])

# Clamp boxes outside the image to its edge
for i in range(output.shape[0]):
	# x is bouded by 0.0, w by image width
	output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, img_dim_list[i, 0])
	output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, img_dim_list[i, 1])

class_load = time.time()
colors = pkl.load(open('pallete', 'rb'))

draw = time.time()

def draw_boxes(x, results):
	tl = tuple(x[1:3].int())
	br = tuple(x[3:5].int())

	img = results[int(x[0])]

	class_id = int(x[-1])
	label = "{0}".format(classes[class_id])
	color = random.choice(colors)
	cv2.rectangle(img, tl, br, color, 1)
	text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
	br = tl[0] + text_size[0] + 3, tl[1] + text_size[1] + 4

	cv2.rectangle(img, tl, br, color, -1)
	cv2.putText(img, label, (tl[0], br[1]), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
	return img

list(map(lambda x: draw_boxes(x, loaded_img), output))

#det_names = pd.Series(img_list).apply(lambda x: "{}/det_{}".format(args.det, x.split('/')[-1]))
#det_names = 
#print("det_names = ", osp.realpath('.')+'\\'+args.det+img_list[0])
det_names = []
for i, name in enumerate(img_list):
	det_name = '/det/det_' + name.split('/')[-1]

	det_names.append('/'.join(name.split('/')[:-1]) + det_name)
	det_name = '/'.join(name.split('/')[:-2]) + det_name

	cv2.imwrite(det_name, loaded_img[i])

print(det_names[0])
#list(max(cv2.imwrite, det_names, loaded_img))
end = time.time()

print("SUMMARY")
print("------------------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time taken (in seconds)"))
print("{:25s}: {:2.3f}".format("Reading addresses:", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch:", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(img_list)) + " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output processing:", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing boxes:", end - draw))
print("{:25s}: {:2.3f}".format("Average time per image:", (end - load_batch)/len(img_list)))

torch.cuda.empty_cache()


#inp = get_test_input()
#print(model)
#pred = model(inp, torch.cuda.is_available())
#print(pred.shape, pred)

#print("Pred shape before NMS: ", pred.shape)
#pred = write_result(pred, 0.6, 80, 0.5)
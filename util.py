from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2

# recalculate predictions + reshape the output tensor so we can concatenate outputs from different layers
def predict_transform(prediction, input_dim, anchors, num_classes, CUDA = True):

	batch_size = prediction.size(0)
	stride = input_dim / prediction.size(2)
	grid_size = input_dim / stride

	stride = int(stride)
	grid_size = int(grid_size)
	
	bbox_attrs = 5 + num_classes
	num_anchors = len(anchors)

	prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
	prediction = prediction.transpose(1,2).contiguous()
	prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
	# anchors' width and hight are relative to the input image size, which is in 'stride' time is larger than the output feature map
	anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

	# calculate (x, y) - the center of the bb and the objectness score
	# for entire batch for each bb
	prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
	prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
	prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

	grid = np.arange(grid_size)
	a, b = np.meshgrid(grid, grid)

	x_offset = torch.FloatTensor(a).view(-1, 1)
	y_offset = torch.FloatTensor(b).view(-1, 1)

	if CUDA:
		x_offset = x_offset.cuda()
		y_offset = y_offset.cuda()

	x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
	# apply the offset of the bb's center
	prediction[:, :, :2] += x_y_offset

	# apply log space transform for the height and width of the bb
	anchors = torch.FloatTensor(anchors)

	if CUDA:
		anchors = anchors.cuda()
	# just repeat bb's width/hight for each cell
	anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
	prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

	# apply sigmoid to the class scores
	prediction[:, :, 5: 5 + num_classes] = torch.sigmoid(prediction[:, :, 5: 5 + num_classes])

	# resize the parameters of the bbs to the input image size
	prediction[:,:,:4] *= stride

	return prediction

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_iou(box1, box2):
	b1_tlx, b1_tly, b1_brx, b1_bry = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
	b2_tlx, b2_tly, b2_brx, b2_bry = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

	# Coordinate of the IoU
	iou_tlx = torch.max(b1_tlx, b2_tlx)
	iou_tly = torch.max(b1_tly, b2_tly)
	iou_brx = torch.min(b1_brx, b2_brx)
	iou_bry = torch.min(b1_bry, b2_bry)

	iou_area = torch.clamp(iou_brx - iou_tlx + 1, min = 0) * torch.clamp(iou_bry - iou_tly + 1, min = 0)

	# Box areas

	b1_area = (b1_brx - b1_tlx) * (b1_bry - b1_tly)
	b2_area = (b2_brx - b2_tlx) * (b2_bry - b2_tly)

	iou = iou_area / (b1_area + b2_area - iou_area)

	return iou


def write_result(prediction, conf, num_classes, nms_conf = 0.4):
	# conf - objectness score threshold
	# nms_conf - IoU threshold
	print("Before maask: ", (prediction[:, :, 4] > conf).shape)
	conf_mask = (prediction[:, :, 4] > conf).float().unsqueeze(2)
	print("After mask: ", conf_mask.shape)
	prediction = prediction * conf_mask

	# Convert x,y,h,w parameters of bb to tlx, tly, brx, bry
	box_corners = prediction.new(prediction.shape)
	# tlx
	box_corners[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2]/2
	# tly
	box_corners[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3]/2
	# brx
	box_corners[:, :, 2] = box_corners[:, :, 0] + prediction[:, :, 2]
	#bry
	box_corners[:, :, 3] = box_corners[:, :, 1] + prediction[:, :, 3]
	"""n = box_corners.size(1)
	for i in range(n):
		if prediction[0, i, 0] > 1:
			print("x = ", prediction[0, i, 0])
			print("y = ", prediction[0, i, 1])
			print("w = ", prediction[0, i, 2])
			print("h = ", prediction[0, i, 3])

			print("tlx = ", box_corners[0, i, 0])
			print("tly = ", box_corners[0, i, 1])
			print("brx = ", box_corners[0, i, 2])
			print("bry = ", box_corners[0, i, 3])
			print("W = ", box_corners[0, i, 2] - box_corners[0, i, 0])
			print("H = ", box_corners[0, i, 3] - box_corners[0, i, 1])"""

	prediction[:, :, :4] = box_corners[:, :, :4]

	# Perform NMS for each sample in the batch
	batch_size = prediction.size(0)

	write = False

	for ind in range(batch_size):
		pred = prediction[ind]

		print("pred.shape = ", pred.shape)

		# For this step we replace class scores (80) with max scores over each box
		#max_conf - the biggest class score in the box
		#max_conf_ind - the index of the max_conf (number of box and index of the class) 
		max_conf, max_conf_ind = torch.max(pred[:, 5: 5 + num_classes], 1)
		max_conf = max_conf.float().unsqueeze(1)
		max_conf_ind = max_conf_ind.float().unsqueeze(1)

		print("max_conf.shape = ", max_conf.shape)

		# First 5 parameters: bb and objectness score, last 2: max class score + index of the class
		seq = (pred[:, :5], max_conf, max_conf_ind)
		pred = torch.cat(seq, 1)

		print("pred.shape = ", pred.shape)

		# Get rid of zeroed boxes (previously, if objectness score < conf)
		non_zero_ind = torch.nonzero(pred[:, 4])

		print(non_zero_ind)

		print("Pred: \n", pred[non_zero_ind[0]])
		print("Pred")
		# To handle situations when there are no detections use try-except
		try:
			pred_ = pred[non_zero_ind.unsqueeze(0), :].view(-1, 7)
			print("From try!")
		except:
			continue

		if (pred_.shape[0] == 0):
			continue

		#Get the various classes detected in the image
		# -1 index holds the class index
		img_classes = unique(pred_[:,-1])

		print("# of classes: ", img_classes)

		# loop over detected classes in the image
		for class_ind in img_classes:
			print("# class: ", class_ind)
			# perform NMS
			# Get the detections with a particular class
			class_mask = pred_ * (pred_[:, -1] == class_ind).float().unsqueeze(1)
			# indexes of the boxes for which max conf is for class_ind class
			class_mask_ind = torch.nonzero(class_mask[:, -2]).squeeze()

			pred_class = pred_[class_mask_ind].view(-1, 7)

			# Sort the detections in the descending order of the objectness score
			conf_sort_ind = torch.sort(pred_class[:, 4], descending = True)[1]

			pred_class = pred_class[conf_sort_ind]
			idx = pred_class.size(0)
			# Loop over all predictions (boxes)
			for i in range(idx):
				# After each iteration the number of boxes may be smaller (some may get suppressed)
				# so some indexes may be invalid. In this case 'except' will triggered 
				try:
					ious = bbox_iou(pred_class[i].unsqueeze(0), pred_class[i + 1:])
				except ValueError:
					break
				except IndexError:
					break

				# Zero out detections with IoU > threshold
				iou_mask = (ious < nms_conf).unsqueeze(1)
				pred_class[i + 1:] *= iou_mask.float()

				# Remove zero entries
				non_zero_ind = torch.nonzero(pred_class[:, 4]).squeeze()
				pred_class = pred_class[non_zero_ind].view(-1, 7)

			# Create an array of size=# of 
			batch_ind = pred_class.new(pred_class.size(0), 1).fill_(ind)
			seq = batch_ind, pred_class

			if not write:
				output = torch.cat(seq, 1)
				write = True
			else:
				out = torch.cat(seq, 1)
				output = torch.cat((output, out))

	try:
		return output
	except:
		return 0

def load_classes(file):
	f = open(file, 'r')
	names = f.read().split('\n')[:-1] # -1 to exclude the last line ''
	return names

def letterbox_image(img, ind_dim):
	img_w, img_h = img.shape[1], img.shape[0]
	w, h = inp_dim
	new_w = int(img_w * min(w/img_w, h/img_h))
	new_h = int(img_h * min(w/img_w, h/img_h))

	resized_img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)

	result = np.fill((w, h, 3), 128)

	result[(h-new_h)//2:(h-new_h)//2 + new_h, (w-new_w)//2:(w-new_w)//2 + new_w, :] = resized_img

	return result

def prep_image(img, inp_dim):
	"""
	Prepare image for inputting to the neural network. 

	Returns a Variable 
	"""
	img = cv2.resize(img, (inp_dim, inp_dim))
	# From BGR image form a C x H x W tensor
	img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
	# Create Variable with 1 x C x H x W dims
	img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)

	return img
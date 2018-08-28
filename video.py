import time
import torch 
#import torch.nn as nn
#from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
#import os 
#import os.path as osp
from darknet import Darknet
import pickle as pkl
#import pandas as pd
import random
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
	"""
	Parse arguements to the detect module

	"""

	parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
	parser.add_argument("--video", dest = 'video', help = 
	                    "Video file",
	                    default = "challenge.mp4", type = str)
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

def process_video():
	args = arg_parse()
	video_file = args.video
	confidence = float(args.confidence)
	nms_thesh = float(args.nms_thresh)
	input_dim = int(args.reso)
	num_classes = 80
	classes = load_classes('data/coco.names')
	CUDA = torch.cuda.is_available()

	nframe = -1
	nbox = 0

	#out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (1280, 720))

	print("Loading the model...")
	model = Darknet(args.cfgfile)
	model.load_weights(args.weightsfile)
	print("SUCCESS\n")

	model.net_info['height'] = int(args.reso)
	colors = pkl.load(open('pallete', 'rb'))

	if CUDA:
		model.cuda()

	# Set the model in evaluation mode
	model.eval()

	cap = cv2.VideoCapture(video_file)
	cv2.namedWindow("result", cv2.WINDOW_NORMAL)
	start = time.time()
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		im_dim = frame.shape[1], frame.shape[0]
		im_dim = torch.FloatTensor(im_dim).repeat(1,2)
		print("im_dim shape = ", im_dim.shape)

		img = prep_image(frame, input_dim)
		#cv2.imshow("result", frame)

		with torch.no_grad():
			prediction = model(Variable(img), CUDA)

		output = write_result(prediction, confidence, num_classes, nms_thesh)

		if (type(output) == int):
			cv2.imshow("result", frame)
			frame += 1
			print("FPS = {:5.2f}".format(nframe / (time.time() - start)))
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			continue

		im_dim = im_dim.repeat(output.size(0), 1)
		scaling_factor = torch.min(input_dim/im_dim, 1)[0].view(-1, 1)

		print("Scale = ", scaling_factor)

		output[:, [1, 3]] -= (input_dim - scaling_factor * im_dim[:, 0].view(-1, 1))/2
		output[:, [2, 4]] -= (input_dim - scaling_factor * im_dim[:, 1].view(-1, 1))/2

		output[:, 1:5] /= scaling_factor

		for i in range(output.shape[0]):
			output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
			output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

		def draw_boxes(x, image,):
			i, pred = x
			tl = tuple(pred[1:3].int())
			br = tuple(pred[3:5].int())
			print("tl = ", tl)
			print("br = ", br)

			print("pred.shape = ", pred.shape)
			#print("Pred[0] = ", pred[0])
			#print("Pred[1] = ", pred[1])

			img = image#[int(x[0])]

			class_id = int(pred[-1])
			label = "{0}".format(classes[class_id])
			color = colors[i]#random.choice(colors)
			cv2.rectangle(img, tl, br, color, 3)
			text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
			br = tl[0] + text_size[0] + 3, tl[1] + text_size[1] + 4

			cv2.rectangle(img, tl, br, color, -1)
			cv2.putText(img, label, (tl[0], br[1]), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
			return img

		result = frame
		list(map(lambda x: draw_boxes(x, result), enumerate(output)))

		cv2.imshow("result", result)

		#out.write(result)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		frame += 1
		print("FPS = {:5.2f}".format(nframe / (time.time() - start)))

	cap.release()
	cv2.destroyAllWindows()

process_video()
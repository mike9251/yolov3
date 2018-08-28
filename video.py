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
	                    default = "video", type = str)
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

	scaling_factor = torch.min(input_dim/(1280, 720), 1)[0].view(-1, 1)

	out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (1280, 720))
	#clf, feature_scaler = load_model()
	#sliding_window(cv2.imread('test_images/test4.jpg'), clf, feature_scaler)
	print("Loading the model...")
	model = Darknet(args.cfgfile)
	model.load_weights(args.weightsfile)
	print("SUCCESS\n")

	model.net_info['height'] = int(args.reso)

	if CUDA:
		model.cuda()

	# Set the model in evaluation mode
	model.eval()

	cap = cv2.VideoCapture(video_file)
	cv2.namedWindow("result", cv2.WINDOW_NORMAL)
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		img = prep_image(frame, input_dim)

		with torch.no_grad():
			prediction = model(Variable(img), CUDA)

		prediction = write_result(prediction, confidence, num_classes, nms_thesh)

		prediction[:, [1, 3]] -= (input_dim - scaling_factor * 1280.view(-1, 1))/2
		prediction[:, [2, 4]] -= (input_dim - scaling_factor * 720.view(-1, 1))/2

		cv2.imshow("result", result)

		out.write(result)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
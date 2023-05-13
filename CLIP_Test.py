import time
import cv2
import pyrealsense2 as rs
import numpy as np
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from planeRansac import Plane
from scipy.ndimage import maximum_filter
# from graspable import circleRsac, plot3DCircle
import os
import threading
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from matplotlib.path import Path
import torch
import clip
import VisionNode
from PIL import Image
import torch.nn as nn
from Finetune_CLIP import CLIPEmbed, convert_models_to_fp32
# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


intr_1 = np.array([326.096, 242.595, 615.588, 615.923])
intr_2 = np.array([316.201, 252.538, 615.668, 615.678])
intr_3 = np.array([317.315, 242.768, 614.85, 615.065])

cov_1 = np.eye(3)*np.array([1,1,5])
cov_2 = np.eye(3)*np.array([1,1,5])
cov_3 = np.eye(3)*np.array([1,1,5])



purpleLow = np.array([135,40,40])
purpleHigh = np.array([150,256,256])

# pinkLow = np.array([150,100,100])
# pinkHigh = np.array([170,256,256])

greenLow = np.array([80,120,120])
greenHigh = np.array([90,256,256])

blueLow = np.array([90,120,120])
blueHigh = np.array([130,256,256])

redLow = np.array([150,80,80])
redHigh = np.array([185,256,256])


def theta2SO2(theta):
	R = np.array([[np.cos(theta),-np.sin(theta)],
				  [np.sin(theta),np.cos(theta)]])
	return(R)

def vec2SE2(vec):
	T = np.eye(3)
	T[:2,:2] = theta2SO2(vec[2])
	T[:2,2] = vec[:2]
	return(T)

def invertSE2(T):
	R = T[:2,:2]
	p = T[:2,2]
	outT = np.eye(3)
	outT[:2,:2] = R.T
	outT[:2,2] = np.dot(R,p)*-1
	return(outT)

def SE22Vec(T):
	vec = np.zeros(3)
	vec[:2] = T[:2,2]
	vec[2] = np.arctan2(T[1,0],T[1,1])
	return(vec)


def run_vision(vis_node):
	try:
		while not vis_node.stopped:
			prev_time = time.time()
			vis_node.update_frames(render_depth_colormap = True)
			# vis_node.detect_aruco()
			# vis_node.render_aruco()
			# vis_node.render_aruco_axis()
			# vis_node.update_bolt_pose(VisionNode.vec2SE3(arm.get_position()[1]),3)
			# vis_node.visual_servo_aruco(VisionNode.vec2SE3(arm.get_position()[1]),3,method='combo', suppress_rotation=True, cache_pose = 0)
			vis_node.render_camera_views()
			time.sleep(.05)


			# print("Avg fps: ",1/(time.time()-prev_time))
	except IndexError as e:
		vis_node.stopped = True
		print(e)

def preprocess_image(n_px):
	return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def main():
	extrinsics_3 = np.eye(4)


	# rvec_1 = np.load("rvec_1.npy")
	rvec_1 = np.zeros(3)
	theta_1 = np.sqrt(np.sum(rvec_1**2))
	omega_1 = rvec_1/theta_1

	# rvec_2 = np.load("rvec_2.npy")
	rvec_2 = np.zeros(3)

	theta_2 = np.sqrt(np.sum(rvec_2**2))
	omega_2 = rvec_2/theta_2
	vis_node = VisionNode.VisionNode([extrinsics_3],
						  [intr_3],
						  [],
						  2,
						  [rvec_1,rvec_2],
						  [cov_1,cov_2,cov_3],
						  load_detic=True)
	vis_node.start_cameras(['152122075556'],exposure=0)
	time.sleep(1)
	
	vision_thread = threading.Thread(target = run_vision, args = (vis_node,))
	vision_thread.start()
	time.sleep(1)
	print("initialized cameras")

	device = "cuda" if torch.cuda.is_available() else "cpu"
	model, preprocess = clip.load("ViT-B/32", device=device)

	convert_models_to_fp32(model)

	# embed_model = CLIPEmbed(512)
	# embed_model.to(device)
	checkpoint = torch.load("model_checkpoint/cropped_clip_model0700.pt")
	# embed_model.load_state_dict(checkpoint['model_state_dict'])

	# Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
	# checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
	# checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
	# checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 

	model.load_state_dict(checkpoint['model_state_dict'])

	text_options = ["water bottle in a roll of tape","water bottle next to a roll of tape", "water bottle","roll of tape" ]
	text_options = ["spatual in pan","spatuala in pot","pan","pot"]
	text_options = ["open","closed"]
	# text_options = []

	try:
		while(True):
			im = vis_node.get_semantic_cropped_img(0)
			# # im = vis_node.color_images[0]
			# # print(np.mean(im))
			# # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

			image = Image.fromarray(im)
			# image = Image.open('cropped_training_images/open_oven_035.jpg')
			# image = torch.from_numpy(im)
			# print(image.shape)
			image_input = preprocess(image).unsqueeze(0).to(device)
			text_inputs = torch.cat([clip.tokenize(f"a photo of a {text}") for text in text_options]).to(device)

			with torch.no_grad():
			    image_features = model.encode_image(image_input)
			    text_features = model.encode_text(text_inputs)
			    # new_image_features = embed_model(image_features)

			# print(torch.mean(image_features))
			# print(torch.mean(new_image_features))
			# # print("SHAPES")
			# print(image_features.shape)
			# print(text_features.shape)
			image_features /= image_features.norm(dim=-1, keepdim=True)
			text_features /= text_features.norm(dim=-1, keepdim=True)
			similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
			print(similarity.shape)
			values, indices = similarity[0].topk(len(text_options))

			print("\nTop predictions:\n")
			for value, index in zip(values, indices):
			    print(f"{text_options[index]:>16s}: {100 * value.item():.2f}%")

			# 1/0
	finally:
		vis_node.stop_cameras()


if __name__ == '__main__':
	main()
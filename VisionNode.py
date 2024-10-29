# from interbotix_xs_modules.arm import InterbotixManipulatorXS
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

import argparse
import glob
import multiprocessing as mp
import tempfile
import warnings
import tqdm
import sys
import mss

# from detectron2.config import get_cfg
# from detectron2.data.detection_utils import read_image
# from detectron2.utils.logger import setup_logger

# sys.path.insert(0, 'third_party/CenterNet2/')
# from centernet.config import add_centernet_config
# from detic.config import add_detic_config

# from detic.predictor import VisualizationDemo
# from PIL import Image
# import detic_module

import detic_interface




intr_1 = np.array([326.096, 242.595, 615.588, 615.923])
intr_2 = np.array([316.201, 252.538, 615.668, 615.678])
intr_3 = np.array([317.315, 242.768, 614.85, 615.065])


cov_1 = np.eye(3)*np.array([1,1,5])
cov_2 = np.eye(3)*np.array([1,1,5])
cov_3 = np.eye(3)*np.array([1,1,5])


def get_coords(intr,pixel_coords,depth_image,mask=np.array([]),mode = "closest"):
	# # print(depth_image.shape)
	# if pixel_coords == (0,0):
	#     return(np.array([0,0,0]))
	# # print(pixel_coords,depth_image[pixel_coords[0]-1:pixel_coords[0]+2,pixel_coords[1]-1:pixel_coords[1]+2])
	# depth = np.max(depth_image[pixel_coords[0]-1:pixel_coords[0]+2,pixel_coords[1]-1:pixel_coords[1]+2])
	
	if mask.any():
		depths = depth_image[mask[:,0].flatten(),mask[:,1].flatten()]
		if(depths[np.nonzero(depths)].shape[0] > 0):
			if mode == "closest":
				depth = np.min(depths[np.nonzero(depths)])
			elif mode == "average":
				depth = np.mean(depths[np.nonzero(depths)])
			elif mode == "farthest":
				depth = np.max(depths[np.nonzero(depths)])
			else:
				print("invalid depth mode")
				depth = 0

		else:
			print("invalid depth")
			# print(depths)
			depth = 0

	else:
		depth = depth_image[pixel_coords]

	x = (pixel_coords[1] - intr[0])/intr[2] *depth
	y = (pixel_coords[0] - intr[1])/intr[3] *depth
	return(np.array([x,y,depth]))

def batch_coords(intr,pixel_coords,depth_image):
	out_coords =np.zeros((pixel_coords.shape[0],3))
	out_coords[:,2] = depth_image[pixel_coords[:,0].flatten(),pixel_coords[:,1].flatten()]
	out_coords[:,0] = (pixel_coords[:,1] - intr[0])/intr[2] *out_coords[:,2]
	out_coords[:,1] = (pixel_coords[:,0] - intr[1])/intr[3] *out_coords[:,2]
	return(out_coords)

def rodriguesrot(x,omega,theta):
	xrot = x*np.cos(theta) + np.cross(omega,x)*np.sin(theta)+omega*np.dot(omega,x)*(1-np.cos(theta))
	return(xrot)

def rodriguesVec(vec, n0, n1):
	# Get vector of rotation k and angle theta
	n0 = n0/np.linalg.norm(n0)
	n1 = n1/np.linalg.norm(n1)
	k = np.cross(n0,n1)
	k = k/np.linalg.norm(k)
	theta = np.arccos(np.dot(n0,n1))
	return(rodriguesrot(vec,k,theta))

def fitPlane(pointcloud, max_points = 20):
	if pointcloud.shape[0] > max_points:
		U,E,V = np.linalg.svd(pointcloud[np.linspace(0,pointcloud.shape[0]-1,max_points).astype(int),:])
	else:
		U,E,V = np.linalg.svd(pointcloud)
	normal_vec = V.T[:,0]
	center = np.mean(pointcloud,axis=0)
	return(normal_vec,center)


def coordTransform(coord, extrinsics):
	homogenous = np.append(coord,1.).reshape((4,1))
	return(np.dot(extrinsics,homogenous)[:3])

def Euler2SO3(angles):
	x, y, z = angles[0], angles[1], angles[2]
	Rx = np.array([[1,0,0],
				   [0,np.cos(x),-np.sin(x)],
				   [0,np.sin(x),np.cos(x)]])
	Ry = np.array([[np.cos(y),0,np.sin(y)],
				   [0,1,0],
				   [-np.sin(y),0,np.cos(y)]])
	Rz = np.array([[np.cos(z),-np.sin(z),0],
				   [np.sin(z),np.cos(z),0],
				   [0,0,1]])
	return(np.dot(Rx,Ry).dot(Rz))

def vec2SE3(vec,intrinsic=True):
	T = np.eye(4)
	if intrinsic:
		T[:3,:3] = Euler2SO3(vec[3:])
	else:
		r = R.from_euler('XYZ',vec[3:])
		T[:3,:3] = r.as_matrix()
	T[:3,3] = vec[:3]
	return(T)

def tvecrvec2SE3(tvec,rvec):
	T = np.eye(4)
	T[:3,:3] = Euler2SO3(rvec)
	T[:3,3] = rvec
	return(T)

def invertSE3(T):
	R = T[:3,:3]
	p = T[:3,3]
	outT = np.eye(4)
	outT[:3,:3] = R.T
	outT[:3,3] = np.dot(R,p)*-1
	return(outT)

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def SO32Euler(R) :

	assert(isRotationMatrix(R))

	sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

	singular = sy < 1e-6

	if  not singular :
		x = np.arctan2(R[2,1] , R[2,2])
		y = np.arctan2(-R[2,0], sy)
		z = np.arctan2(R[1,0], R[0,0])
	else :
		print("singular")
		x = np.arctan2(-R[1,2], R[1,1])
		y = np.arctan2(-R[2,0], sy)
		z = 0

	return np.array([x, y, z])

def SE32vec(T):
	vec = np.zeros(6)
	vec[:3] = T[:3,3]
	vec[3:] = SO32Euler(T[:3,:3])
	return(vec)

def SO32so3(R):
	theta = np.arccos((np.trace(R)-1)/2)
	omega = 1/(2*np.sin(theta))*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
	return(omega*theta)

def SE32AA(T):
	ret = np.zeros(6)
	ret[:3] = T[:3,3]
	ret[3:] = SO32so3(T[:3,:3])
	return(ret)

def AA2SE3(T):
	ret = np.eye(4)
	ret[:3,3] = T[:3]
	ret[:3,:3] = so32SO3(T[3:])
	return(ret)

def vec2SE2(vec):
	T = np.eye(3)
	T[:2,:2] = Euler2SO2(vec[2])
	T[:2,2] = vec[:2]
	return(T)

def Euler2SO2(theta):
	return(np.array([[np.cos(theta),-np.sin(theta)],
					 [np.sin(theta),np.cos(theta)]]))

def invertSE2(T):
	R = T[:2,:2]
	p = T[:2,2]
	outT = np.eye(3)
	outT[:2,:2] = R.T
	outT[:2,2] = np.dot(R,p)*-1
	return(outT)

def interpolateTransforms(T1,T2,u):
	matstack = np.vstack((T1[:3,:3].reshape(1,3,3),T2[:3,:3].reshape(1,3,3)))
	rots = R.from_matrix(matstack)
	slerp = Slerp([0,1],rots)
	Tret = np.eye(4)
	Tret[:3,:3] = slerp([u])[0].as_matrix()
	Tret[:3,3] = (T2[:3,3] - T1[:3,3])*u + T1[:3,3]
	return(Tret)

def hatmap(vec):
	return(np.array([[0,-vec[2],vec[1]],
					 [vec[2],0,-vec[0]],
					 [-vec[1],vec[0],0]]))
def so32SO3(omega):
	theta = np.linalg.norm(omega)
	omega /= theta
	K = hatmap(omega)
	# print(K)
	# print(theta)
	# print(np.sin(theta)*K)
	# print()
	return(np.eye(3)+np.sin(theta)*K+(1-np.cos(theta))*np.dot(K,K))

def computeVectorRot(v1,v2):
	omega = np.cross(v1,v2)
	omega /= np.linalg.norm(omega)
	theta = np.arccos(np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2))
	return(omega*theta)

def fix_rotation(ang):
	return((ang + np.pi)%(np.pi*2)-np.pi)



class FramePlot:
	def __init__(self):
		self.xlim = [0,700]
		self.ylim = [-350,350]
		self.zlim = [0,700]

		self.fig= plt.figure()
		self.ax = self.fig.add_subplot(111,projection='3d')
		self.ax.set_xlim(self.xlim)
		self.ax.set_ylim(self.ylim)
		self.ax.set_zlim(self.zlim)
		self.ax.set_xlabel("X")
		self.ax.set_ylabel("Y")
		self.ax.set_zlabel("Z")

	def clearAxis(self):
		plt.cla()
		self.ax.set_xlim(self.xlim)
		self.ax.set_ylim(self.ylim)
		self.ax.set_zlim(self.zlim)
		self.ax.set_xlabel("X")
		self.ax.set_ylabel("Y")
		self.ax.set_zlabel("Z")

	def plotFrame(self,T,vec_length = 50, label = ""):
		pos = T[:3,3]
		R = T[:3,:3]
		xvec = np.dot(R,np.array([[vec_length,0,0]]).T).flatten()
		yvec = np.dot(R,np.array([[0,vec_length,0]]).T).flatten()
		zvec = np.dot(R,np.array([[0,0,vec_length]]).T).flatten()
		self.ax.quiver(pos[0],pos[1],pos[2],xvec[0],xvec[1],xvec[2],color='r')
		self.ax.quiver(pos[0],pos[1],pos[2],yvec[0],yvec[1],yvec[2],color='g')
		self.ax.quiver(pos[0],pos[1],pos[2],zvec[0],zvec[1],zvec[2],color='b')
		self.ax.text(pos[0],pos[1],pos[2],label,'y')

	def drawPlot(self,t =.05):
		plt.draw()
		plt.show(block=False)
		plt.pause(t)
		
	def showPlot(self):
		plt.show()



class VisionNode:
	def __init__(self,extrinsics,intrinsics,fixed_cameras,robot_camera,rvecs,camera_covariances,save_path = "", model_config="",model_weights="",load_detic=False, detic_lang_tags = None):
		self.pipelines = []
		self.cam_configs = []
		self.aligns = []
		self.extrinsics = extrinsics
		self.rvecs = rvecs
		self.thetas = []
		self.omegas = []
		for rvec in rvecs:
			theta = np.sqrt(np.sum(rvec**2))
			omega = rvec/theta
			self.thetas.append(theta)
			self.omegas.append(omega)

		self.intrinsics = intrinsics
		self.intrinsics_mat = []
		for intr in intrinsics:
			mat = np.eye(3)
			mat[0,0] = intr[2]
			mat[1,1] = intr[3]
			mat[0,2] = intr[0]
			mat[1,2] = intr[1]
			self.intrinsics_mat.append(mat)
		self.fixed_cameras = fixed_cameras
		self.robot_camera = robot_camera
		self.inv_rob_covs = []
		# inv_rob_cov_sum = np.zeros((3,3))
		# for i in self.fixed_cameras:
		# 	inv_rob_cov = np.linalg.inv(np.dot(extrinsics[i][:3,:3],camera_covariances[i]).dot(extrinsics[i][:3,:3].T))
		# 	self.inv_rob_covs.append(inv_rob_cov)
		# 	inv_rob_cov_sum = inv_rob_cov_sum + inv_rob_cov
		# self.inv_agg_cov = np.linalg.inv(inv_rob_cov_sum)
		self.color_images = [None]*len(intrinsics)
		self.depth_images = [None]*len(intrinsics)
		self.depth_colormaps = [None]*len(intrinsics)
		self.detic_vis = [None]*len(intrinsics)
		self.camera_rotations = [cv2.ROTATE_180, None]
		self.stopped = False
		if load_detic:
			if not detic_lang_tags:
				self.detic_lang_tags = ['spatula']
			else:
				self.detic_lang_tags = detic_lang_tags
			self.detic = detic_interface.Predictor()
			self.detic.setup(self.detic_lang_tags)

		else:
			self.detic = None

		self.save_path = save_path
		self.save_counter = 0
		if self.save_path != "" and not os.path.exists(self.save_path):
			os.makedirs(self.save_path)
		
		self._vision_thread = threading.Thread(target=self._vision_thread)
		self._mutex = threading.Lock()

	def _vision_thread(self):
		while not self.stopped:
			
			self.update_frames(render_depth_colormap = True)
			
			self.render_camera_views()
			self._mutex.acquire()
			if self.save_path != "":
				self.save_frames()
			self._mutex.release()
			time.sleep(.05)
	
	def save_frames(self):
		for i in range(len(self.color_images)):
			cv2.imwrite(os.path.join(self.save_path,"color_image_"+str(i).zfill(2)+"_"+str(self.save_counter).zfill(4)+".png"),self.color_images[i])
		self.save_counter += 1

	def start_recording(self,save_path):
		self._mutex.acquire()
		self.save_path = save_path
		self.save_counter = 0
		if not os.path.exists(self.save_path):
			os.makedirs(self.save_path)
		# delete any image files in the directory
		for file in os.listdir(self.save_path):
			if file.endswith(".png"):
				os.remove(os.path.join(self.save_path,file))
		self._mutex.release()

	def stop_recording(self):
		self._mutex.acquire()
		self.save_path = ""
		self.save_counter = 0
		self._mutex.release()


	def start_cameras(self,camera_ids,exposure=0,resolution=(640,480)):
		for cid in camera_ids:
			cam_pipeline = rs.pipeline()
			cam_config = rs.config()
			cam_config.enable_device(cid)
			cam_config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, 30)
			cam_config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, 30)
			align = rs.align(rs.stream.color)
			cfg = cam_pipeline.start(cam_config)
			if exposure > 0:
				color_sensor = cfg.get_device().query_sensors()[1]
				color_sensor.set_option(rs.option.exposure, exposure)
			self.pipelines.append(cam_pipeline)
			self.cam_configs.append(cam_config)
			self.aligns.append(align)

		self._vision_thread.start()



	def stop_cameras(self):
		self.stopped = True
		if self._vision_thread.is_alive():
			self._vision_thread.join()
		for pipeline in self.pipelines:
			pipeline.stop()

	def update_frames(self,render_depth_colormap = False):
		for i in range(len(self.pipelines)):
			while True:
				frames = self.pipelines[i].wait_for_frames()
				depth_frame = frames.get_depth_frame()
				color_frame = frames.get_color_frame()
				if depth_frame and color_frame:
					break
			color_image = np.asanyarray(color_frame.get_data())
			frameset = self.aligns[i].process(frames)
			aligned_depth_frame = frameset.get_depth_frame()
			aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
			self.color_images[i] = color_image
			self.depth_images[i] = aligned_depth_image
			if self.camera_rotations[i] != None:
				self.color_images[i] = cv2.rotate(self.color_images[i],self.camera_rotations[i])
				self.depth_images[i] = cv2.rotate(self.depth_images[i],self.camera_rotations[i])

			if render_depth_colormap:
				depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_images[i], alpha=0.5), cv2.COLORMAP_JET)
				self.depth_colormaps[i] = depth_colormap

	def update_frames_threaded(self,render_depth_colormap = False,fps = 30.):
		while not self.stopped:
			self.update_frames(render_depth_colormap = render_depth_colormap)
			time.sleep(1./fps)

	def render_camera_views(self):
		cols = []
		for cam in range(len(self.intrinsics)):
			# print(self.color_images[cam].shape, self.depth_colormaps[cam].shape)
			cols.append(np.vstack((self.color_images[cam], self.depth_colormaps[cam])))
		images = np.hstack(cols)
		# print(images.shape,images.dtype)
		cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
		cv2.imshow('RealSense', images)
		cv2.waitKey(1)

	def get_camera_images(self):
		while not np.any(self.color_images) or not np.any(self.depth_images):
			time.sleep(.001)
		return(self.color_images,self.depth_images)

	def render_camera_views_cropped(self):
		images = self.detic.get_image_crop(self.color_images[0])
		cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
		cv2.imshow('RealSense', images)
		cv2.waitKey(1)

	def render_camera_views_threaded(self,fps = 30.):
		while not self.stopped:
			self.render_camera_views()
			time.sleep(1./fps)


	def world_frame_point_cloud(self,mask,cam,robot_pose):
		points = batch_coords(self.intrinsics[cam],mask,self.depth_images[cam])
		print(points.shape)
		homogenous_points = np.vstack((points.T,np.ones((1,points.shape[0]))))
		cam_pose = np.dot(robot_pose, self.extrinsics[self.robot_camera])
		world_points = np.dot(cam_pose,homogenous_points).T
		return(world_points[:,:-1])


	def get_point_cloud(self,cam,depth_max = 1000):
		depth_image = self.depth_images[cam]
		u = np.arange(depth_image.shape[0])
		v = np.arange(depth_image.shape[1])
		uu,vv = np.meshgrid(u,v)
		coords = np.hstack((uu.reshape((-1,1)),vv.reshape((-1,1))))
		points = batch_coords(self.intrinsics[cam],coords,depth_image)
		return(points)

	def get_semantic_cropped_img(self,cam):
		img = self.color_images[cam]
		cropped_im = self.detic.get_image_crop(img)
		return(cropped_im)

	def run_detection(self,cam,visualize=False,vis_length=0):
		img = self.color_images[cam]
		predictions = self.detic.evaluate(img)
		if visualize:
			vis = self.detic.gen_vis(img,predictions)
		return(predictions)




def run_vision(vis_node):
	try:
		while not vis_node.stopped:
			vis_node.update_frames(render_depth_colormap = True)
			vis_node.render_camera_views()
			time.sleep(.05)


			# print("Avg fps: ",1/(time.time()-prev_time))
	except IndexError as e:
		vis_node.stopped = True
		print(e)
	finally:
		vis_node.stop_cameras()

def create_vision_node():
	camera_list = ['746612070227','242322071433']
	extrinsics_3 = np.eye(4)
	extrinsics_3[:3,3] = np.array([71.,-15.,30.])
	intrinsics = {}
	intrinsics['746612070227'] = intr_2
	intrinsics['827312070621'] = intr_3
	intrinsics['152122075556'] = intr_3
	intrinsics['242322071433'] = intr_3

	extrinsics = {}
	extrinsics['746612070227'] = extrinsics_3
	extrinsics['827312070621'] = extrinsics_3
	extrinsics['152122075556'] = extrinsics_3
	extrinsics['242322071433'] = extrinsics_3

	rvec_1 = np.zeros(3)
	theta_1 = np.sqrt(np.sum(rvec_1**2))
	omega_1 = rvec_1/theta_1

	# rvec_2 = np.load("rvec_2.npy")
	rvec_2 = np.zeros(3)
	theta_2 = np.sqrt(np.sum(rvec_2**2))
	omega_2 = rvec_2/theta_2
	vis_node = VisionNode([extrinsics[x] for x in camera_list],
						  [intrinsics[x] for x in camera_list],
						  [], # which camera is fixed
						  2,
						  [rvec_1, rvec_2],
						  [cov_1,cov_2,cov_3],
						  detic_lang_tags=['skillet','spatula'],
						  save_path = "") # place the name of the objective here

	vis_node.start_cameras(camera_list,exposure=200)
	return(vis_node)



def main():

	vis_node = create_vision_node()
	_ = input("Press enter to stop")
	vis_node.stop_cameras()
	sys.exit(0)


	# v1 = np.array([0,0,1])
	# v2 = np.array([0,-1/np.sqrt(2),-1/np.sqrt(2)])
	# print(SO32Euler(so32SO3(computeVectorRot(v1,v2))))
	# 1/0
	camera_list = ['746612070227','827312070621','152122075556']
	camera_list = ['827312070621','746612070227']
	camera_list = ['242322071433']
	camera_list = ['746612070227']

	# extrinsics_1 = np.load("extrinsics_1.npy")
	# extrinsics_2 = np.load("extrinsics_2.npy")
	extrinsics_3 = np.eye(4)
	# extrinsics_3[3:,3] = np.array([15.,71.,30.,])
	extrinsics_3[:3,3] = np.array([71.,-15.,30.])

	intrinsics = {}
	intrinsics['746612070227'] = intr_2
	intrinsics['827312070621'] = intr_3
	intrinsics['152122075556'] = intr_3

	extrinsics = {}
	extrinsics['746612070227'] = extrinsics_3
	extrinsics['827312070621'] = extrinsics_3
	extrinsics['152122075556'] = extrinsics_3




	# rvec_1 = np.load("rvec_1.npy")
	rvec_1 = np.zeros(3)
	theta_1 = np.sqrt(np.sum(rvec_1**2))
	omega_1 = rvec_1/theta_1

	# rvec_2 = np.load("rvec_2.npy")
	rvec_2 = np.zeros(3)
	theta_2 = np.sqrt(np.sum(rvec_2**2))
	omega_2 = rvec_2/theta_2
	vis_node = VisionNode([extrinsics[x] for x in camera_list],
						  [intrinsics[x] for x in camera_list],
						  [], # which camera is fixed
						  2,
						  [rvec_2],
						  [cov_1,cov_2,cov_3],
						  detic_lang_tags=['skillet','spatula']) # place the name of the objective here
	# vis_node.start_cameras(['827112072509','746612070227','827312070621'],exposure=0)
	vis_node.start_cameras(camera_list,exposure=0)

	# time.sleep(1)
		
	# vision_thread = threading.Thread(target = run_vision, args = (vis_node,))
	# vision_thread.start()
	# time.sleep(5)

	# try:
	# 	language_tags = []
	# 	counter = 0
	# 	for it in range(30):
	# 		i = input("Language tag")
	# 		language_tags.append(i)
	# 		for c in range(3):
	# 			im = Image.fromarray(vis_node.color_images[c])
	# 			im.save("training_images/"+str(i.replace(' ','_'))+"_"+"%03d" % (counter,)+""+".jpg")
	# 			counter += 1
	# 	np.save("language_tags",np.array(language_tags))
	# 	vis_node.stop_cameras()
	# finally:
	# 	vis_node.stop_cameras()
	# 	vision_thread.join()

	# 1/0

	try:
		# camThread = threading.Thread(target = vis_node.update_frames_threaded, args = (True,30), daemon=True)
		# camThread.start()
		# renderThread = threading.Thread(target = vis_node.render_camera_views_threaded, args = (30), daemon = True)
		# renderThread.start()
		# fp = FramePlot()
		# coords = []
		language_tags = []

		while True:
			prev_time = time.time()
			vis_node.update_frames(render_depth_colormap = True)
			# vis_node.localize_bolt()
			# print(vis_node.bolt_location)
			# print(vis_node.bolt_ang)
			# vis_node.detect_aruco()
			# vis_node.render_aruco()
			# vis_node.render_aruco_axis()ls tra

			# vis_node.save_state(save_frames=True, force_dump=False)
			# print(vis_node.corners[2])
			# print(vis_node.ids[2])

			# print(vis_node.visual_servo_aruco(np.eye(4),4))
			# print(vis_node.get_marker_pos_depth(0,0))
			# print(vis_node.get_marker_pos_size(0,0))spo

			# print(vis_node.get_marker_pos_depth(1,0))
			# print(vis_node.get_marker_pos_size(1,0))
			# tik = time.time()
			# tvec, rvec, occluded = vis_node.get_marker_pose_combo(2,3)
			# # print(time.time()-tik)
			# # tik = time.time()
			# T = tvecrvec2SE3(tvec,rvec)
			# # print(time.time()-tik)
			# print(T)

			# fp.clearAxis()
			# fp.plotFrame(T)
			# fp.drawPlot()

			# print(vis_node.detect_marker_rotation(2,3))
			# print(vis_node.get_tag_pixel_coords(2,3))
			# coord, occluded  = vis_node.estimate_coords_tag(2)
			# if not occluded:
			# 	coords.append(coord.T)


			# vis_node.visual_servo(vec2SE3(robot_pose),0)
			# vis_node.estimate_coords(0)
			# vis_node.estimate_coords(0)
			vis_node.render_camera_views()
			# predictions = vis_node.run_detection(0,visualize=True,vis_length=10)
			# spatula_bbox = predictions['instances'].pred_boxes.tensor.cpu().numpy()[0]
			# u = np.arange(spatula_bbox[1],spatula_bbox[3],dtype=int)
			# v = np.arange(spatula_bbox[0],spatula_bbox[2],dtype=int)
			# uu,vv = np.meshgrid(u,v)
			# coords = np.vstack((uu.flatten(),vv.flatten())).T
			# print(coords.shape)
			# pc = batch_coords(vis_node.intrinsics[0],coords,vis_node.depth_images[0])
			# np.save("spatula_pc.npy",pc)
			# cv2.imwrite("Pan_Spatula.jpg",vis_node.color_images[0])
			# np.save("RGB.npy",vis_node.color_images[0])
			# np.save("Depth.npy",vis_node.depth_images[0])

			# vis_node.save_single_state(save_frames=True)
			# print("Avg fps: ",1/(time.time()-prev_time))
			# pc, occluded = findPointCloud(vis_node.color_images[2],vis_node.depth_images[2], intr_3, blueLow, blueHigh)
			# np.save("DoorHandlePC.npy",pc)
			# pc = vis_node.get_point_cloud(0)
			# pc/=1000
			# print(pc.shape)
			# img = vis_node.color_images[0]
			# print(img.shape)
			# print(img.reshape((-1,3)).shape)
			# np.savez("pc.npz", xyz = pc, xyz_color = img.transpose(1,0,2).reshape((-1,3)))
			# im = Image.fromarray(vis_node.color_images[0])
			# im.save("test.jpg")
			


	finally:

		vis_node.stop_cameras()
		# vis_node.save_single_state(save_frames=True)
		# vis_node.save_state(save_frames=True, force_dump=True)
		# coords = np.vstack(coords)
		# np.save(vis_node.save_folder+"coords"+".npy",coords)


if __name__ == '__main__':
	# import cProfile
	# cProfile.run('main()')
	main()


	
	


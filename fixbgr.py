from PIL import Image
import cv2
import numpy as np
import os

def fixbgr(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img

def main():
	directory = 'val_images/'
	#load all images from directory, pass to fixbgr, and then save the output
	list_image_names = os.listdir(directory) 
	list_image_path = [directory+x for x in list_image_names]
	for image_path in list_image_path:
		img = cv2.imread(image_path)
		img = fixbgr(img)
		cv2.imwrite(image_path,img)

if __name__ == '__main__':
	main()
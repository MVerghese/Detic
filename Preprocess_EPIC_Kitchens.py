import detic_module
import numpy as np
import os
from detectron2.data.detection_utils import read_image
import cv2
from PIL import Image

def preprocess(epic_path, save_path, detic_interface, past_tense_list, epic_nouns):
	# Preprocess the EPIC-Kitchens dataset
	# epic_path is the path to the EPIC-Kitchens dataset
	# This function will save the cropped images in a new folder called "cropped_images"
	# The cropped images will be saved in the same folder structure as the original images

	# Create a new folder for the cropped images
	cropped_path = os.path.join(save_path, 'cropped_images')
	if not os.path.exists(cropped_path):
		os.makedirs(cropped_path)
	# Create a new folder for the annotations
	annotation_path = os.path.join(save_path, 'annotations')
	if not os.path.exists(annotation_path):
		os.makedirs(annotation_path)

	# Open past_tesne_list and split lines into array
	with open(past_tense_list, 'r') as f:
		read_past_tense_list = f.read().splitlines()

	for line in read_past_tense_list[24000:]:
		entry = line[2:-2].split("' '")
		if len(entry) < 7:
			continue
		annotation = entry[2]
		#replace 'opened' with 'open' in annotation
		annotation = annotation.replace('opened', 'open')
		#replace 'closed' with 'close' in annotation
		annotation = annotation.replace('closed', 'close')
		p_id = entry[3]
		v_id = entry[4]
		endframe = entry[6]

		print(entry[0],annotation)

		#format endframe with 10 digits and leading zeros

		image_path = epic_path + '/' + p_id + '/rgb_frames/' + v_id + '/' + 'frame_' + str(endframe.zfill(10)) + '.jpg'
		try:
			img = read_image(image_path, format="BGR")
		except FileNotFoundError:
			print("FileNotFoundError")
			continue

		predictions  = detic_interface.run_detection(img,visualize=False)
		lang_classes = []
		for noun in epic_nouns:
			if noun in annotation:
				lang_classes.append(epic_nouns.index(noun))

		pred_classes = predictions['instances'].pred_classes.cpu().numpy()

		bbox_idxs = np.in1d(pred_classes,lang_classes).nonzero()[0]

		# print(predictions['instances'].scores.cpu().numpy())
		bboxes = predictions['instances'].pred_boxes.tensor.cpu().numpy()
		# print(bboxes)
		lang_bboxes = bboxes[bbox_idxs,:]
		if lang_bboxes.shape[0] == 0:
			print("Object not found in image")
			# Image.fromarray(img).save(cropped_path + '/' + p_id + '/' + v_id + '/' + v_id+'_'+str(endframe.zfill(10))+'_'+annotation.replace(' ','_') + '.jpg')

			continue
		# print(lang_bboxes)
		total_bbox = np.array([np.concatenate((np.min(lang_bboxes[:,:2],axis=0),np.max(lang_bboxes[:,2:],axis=0)))])
		total_bbox = detic_module.expand_bbox(total_bbox,1.5)
		total_bbox = total_bbox.reshape(4).astype(int)
		# print(total_bbox)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		cropped_image = img[total_bbox[1]:total_bbox[3],total_bbox[0]:total_bbox[2],:]
		im = Image.fromarray(cropped_image)
		if not os.path.exists(cropped_path + '/' + p_id + '/' + v_id):
			os.makedirs(cropped_path + '/' + p_id + '/' + v_id)

		# Image.fromarray(img).save(cropped_path + '/' + p_id + '/' + v_id + '/' + v_id+'_'+str(endframe.zfill(10))+'_'+annotation.replace(' ','_') + '.jpg')
		im.save(cropped_path + '/' + p_id + '/' + v_id + '/' + v_id+'_'+str(endframe.zfill(10))+'_'+annotation.replace(' ','_') + '.jpg')


def main():
	# Path to EPIC-Kitchens dataset
	epic_path = '/media/mverghese/Mass Storage/EPIC-KITCHENS'
	# Path to save the cropped images
	save_path = '/media/mverghese/Mass Storage/EPIC-KITCHENS_Processed'
	# Path to past_tense_list
	past_tense_list = '/home/mverghese/PrimitiveDecomp/src/results_fixed.txt'
	#Load epic kitchens noun classes
	with open('/media/mverghese/Mass Storage/epic-kitchens-100-annotations/EPIC_100_noun_classes_v2.csv', 'r') as f:
		read_noun_classes = f.read().splitlines()
	epic_nouns = []
	for n in read_noun_classes[1:]:
		noun = n.split(',')[1][1:-1]
		if ':' in noun:
			parts = noun.split(':')
			noun = parts[1]+' '+parts[0]
		epic_nouns.append(noun)
	# Join epic_nouns list in to one string separated by commas
	# epic_nouns_list = ','.join(epic_nouns)
	# print(epic_nouns_list)
	# Initialize DeticModule
	detic_interface = detic_module.DeticModule(epic_nouns)
	# detic_interface = None
	# Preprocess the EPIC-Kitchens dataset
	preprocess(epic_path, save_path, detic_interface, past_tense_list, epic_nouns)

if __name__ == '__main__':
	main()
	
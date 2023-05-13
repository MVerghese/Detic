import numpy as np 
from detic_module import DeticModule
from detectron2.data.detection_utils import read_image
from PIL import Image
from matplotlib import pyplot as plt

def point_in_box(point,box):
	# Check if a point is inside a box
	# point: 2D point
	# box: 2D box
	# return: True if the point is inside the box, False otherwise
	if point[0] < box[0] or point[0] > box[2] or point[1] < box[1] or point[1] > box[3]:
		return False
	return True


# Find the 2D point in the query image that matches the reference image
def match(detic_interface,ref_image,ref_point,query_image):
	# Find the 2D point in the query image that matches the reference image
	# ref_image: reference image
	# ref_point: 2D point in the reference image
	# query_image: query image
	# return: 2D point in the query image that matches the reference image
	#		 None if no match is found
	ref_predictions = detic_interface.run_detection(ref_image,visualize=False)
	query_predictions = detic_interface.run_detection(query_image,visualize=False)
	ref_rois = ref_predictions['instances'].pred_boxes.tensor.cpu().numpy()
	query_rois = query_predictions['instances'].pred_boxes.tensor.cpu().numpy()
	ref_embeddings = []
	for i in range(ref_rois.shape[0]):
		bbox = ref_rois[i,:]
		if point_in_box(ref_point,bbox):
			cropped_image = ref_image[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),:]
			# Convert image to PIL Image
			cropped_image = Image.fromarray(cropped_image)
			ref_embeddings.append(detic_interface.get_clip_embeddings(cropped_image).cpu().numpy().flatten())
			# Display bbox on ref_image and visualize
			ref_image_copy = ref_image.copy()
			ref_image_copy[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[0])+5,:] = [255,0,0]
			ref_image_copy[int(bbox[1]):int(bbox[3]),int(bbox[2])-5:int(bbox[2]),:] = [255,0,0]
			ref_image_copy[int(bbox[1]):int(bbox[1])+5,int(bbox[0]):int(bbox[2]),:] = [255,0,0]
			ref_image_copy[int(bbox[3])-5:int(bbox[3]),int(bbox[0]):int(bbox[2]),:] = [255,0,0]
			print(ref_predictions['instances'].scores[i].item())
			visualize_img(ref_image_copy)



	ref_embeddings = np.array(ref_embeddings)
	query_embeddings = np.zeros((query_rois.shape[0],512))
	for i in range(query_rois.shape[0]):
		bbox = query_rois[i,:]
		cropped_image = query_image[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),:]
		# Convert image to PIL Image
		cropped_image = Image.fromarray(cropped_image)
		query_embeddings[i,:] = detic_interface.get_clip_embeddings(cropped_image).cpu().numpy()
	similarity_scores = np.matmul(ref_embeddings,query_embeddings.T)
	print(similarity_scores.shape)
	best_match = np.argmax(similarity_scores[4,:])
	print("Best match: ",best_match)
	# Display bbox on query_image and visualize
	bbox = query_rois[best_match,:]
	query_image_copy = query_image.copy()
	query_image_copy[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[0])+5,:] = [255,0,0]
	query_image_copy[int(bbox[1]):int(bbox[3]),int(bbox[2])-5:int(bbox[2]),:] = [255,0,0]
	query_image_copy[int(bbox[1]):int(bbox[1])+5,int(bbox[0]):int(bbox[2]),:] = [255,0,0]
	query_image_copy[int(bbox[3])-5:int(bbox[3]),int(bbox[0]):int(bbox[2]),:] = [255,0,0]
	print(query_predictions['instances'].scores[best_match].item())
	visualize_img(query_image_copy)



def visualize_img(img):
	# Visualize image
	# img: image
	# return: None
	plt.imshow(img)
	plt.show()


def main():
	detic_interface = DeticModule(["oven","sink","carrot","stove","handle"],output_score_threshold=0.0,load_clip=True)
	ref_image = read_image('PlayKitchen2_Oven.jpg')
	# visualize_img(ref_image)
	ref_point = np.array([26,39])
	query_image = read_image('PlayKitchen3_Oven.jpg')
	match(detic_interface,ref_image,ref_point,query_image)


if __name__ == '__main__':
	main()
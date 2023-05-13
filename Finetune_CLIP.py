import torch
import clip
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np


device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training

EPOCH = 10000
BATCH_SIZE = 30
DECAY_START = .996

class image_title_dataset(Dataset):
	def __init__(self, list_image_path,list_txt):

		self.image_path = list_image_path
		self.title  = clip.tokenize(list_txt) #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.

	def __len__(self):
		return len(self.title)

	def __getitem__(self, idx):
		im = Image.open(self.image_path[idx])
		# print(im)
		# print(im.size)
		# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		image = preprocess(im) # Image from PIL module
		title = self.title[idx]
		return image,title

# use your own data
 #Define your own dataloader

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
	for p in model.parameters(): 
		p.data = p.data.float()
		if p.grad is not None: 
			p.grad.data = p.grad.data.float() 

def convert_models_to_fp16(model): 
	for p in model.parameters(): 
		p.data = p.data.half()
		if p.grad is not None: 
			p.grad.data = p.grad.data.half()

class CLIPEmbed(nn.Module):
	def __init__(self, embed_dim):
		super().__init__()

		self.linear1 = nn.Linear(embed_dim, embed_dim)
		self.linear2 = nn.Linear(embed_dim, embed_dim)
		self.linear3 = nn.Linear(embed_dim, embed_dim)
		# self.linear4 = nn.Linear(embed_dim, embed_dim)

		self.relu1 = nn.ReLU()
		self.relu2 = nn.ReLU()
		# self.relu3 = nn.ReLU()

	def forward(self, x):
		# print("1",torch.mean(x))
		x = self.relu1(self.linear1(x))
		# print("2",torch.mean(x))
		x = self.relu2(self.linear2(x))
		# print("3",torch.mean(x))
		x = self.linear3(x)
		# print("4",torch.mean(x))
		# x = self.linear4(x)
		return x

def update_target_model(model,target_model,decay):
	for param, target_param in zip(model.parameters(), target_model.parameters()):
		target_param.data.copy_(decay * target_param.data + (1 - decay) * param.data)


		
def train():
	list_image_path = os.listdir('cropped_training_images/') 
	list_txt = [x[:-6].replace('_',' ') for x in list_image_path]
	print(list_txt)
	list_image_path = ['cropped_training_images/'+x for x in list_image_path]
	print(list_image_path)
	dataset = image_title_dataset(list_image_path,list_txt)
	train_dataloader = DataLoader(dataset,batch_size = BATCH_SIZE)
	if device == "cpu":
		model.float()
	else :
		clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

	loss_img = nn.CrossEntropyLoss()
	loss_txt = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-6,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

	# add your own code to track the training progress.

	for epoch in range(EPOCH):
		for batch in train_dataloader :
			optimizer.zero_grad()

			images,texts = batch 

			images= images.to(device)
			texts = texts.to(device)

			logits_per_image, logits_per_text = model(images, texts)

			ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

			total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
			print(total_loss.item())
			total_loss.backward()
			if device == "cpu":
				optimizer.step()
			else : 
				convert_models_to_fp32(model)
				optimizer.step()
				clip.model.convert_weights(model)

	torch.save({
	'epoch': epoch,
	'model_state_dict': model.state_dict(),
	'optimizer_state_dict': optimizer.state_dict(),
	'loss': total_loss,
	}, f"model_checkpoint/test_model_2.pt") #just change to your preferred folder/filename

def train_frozen_lang(byol_weight = 0.1):
	list_image_path = os.listdir('cropped_training_images/') 
	list_txt = [x[:-8].replace('_',' ') for x in list_image_path]
	# print(list_txt)
	list_image_path = ['cropped_training_images/'+x for x in list_image_path]
	# print(list_image_path)
	dataset = image_title_dataset(list_image_path,list_txt)
	train_dataloader = DataLoader(dataset,batch_size = BATCH_SIZE)


	val_list_image_path = os.listdir('cropped_val_images/')
	val_list_txt = [x[:-8].replace('_',' ') for x in val_list_image_path]
	# print(val_list_txt)
	val_list_image_path = ['cropped_val_images/'+x for x in val_list_image_path]
	# print(val_list_image_path)
	val_dataset = image_title_dataset(val_list_image_path,val_list_txt)
	val_dataloader = DataLoader(val_dataset,batch_size = BATCH_SIZE)


	# 
	# Image.open(list_image_path[0]).show()
	# 1/0
	target_model, _ = clip.load("ViT-B/32",device=device,jit=False)
	target_model = target_model.to(device)
	if device == "cpu":
		model.float()
		target_model.float()
	else :
		convert_models_to_fp32(model)
		convert_models_to_fp32(target_model)
		# pass
		# clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

	# im_embed_model = CLIPEmbed(512)
	# im_embed_model.to(device)

	# lang_embed_model = CLIPEmbed(512)
	# lang_embed_model.to(device)
	# checkpoint = torch.load("model_checkpoint/embed_model.pt")
	# embed_model.load_state_dict(checkpoint['model_state_dict'])
	# convert_models_to_fp16(embed_model)
	
	# print(combinatorial_map)
	# 1/0




	# loss_img = nn.CrossEntropyLoss()
	# loss_txt = nn.CrossEntropyLoss()
	loss = nn.CosineEmbeddingLoss()
	byolLoss = nn.MSELoss()
	# params = list(im_embed_model.parameters()) + list(lang_embed_model.parameters())
	optimizer = optim.Adam(model.parameters(), lr=1e-6,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

	# add your own code to track the training progress.
	all_loss = []
	all_val_loss = []

	fig = plt.figure(figsize=(12, 8))
	decay = DECAY_START

	for epoch in range(EPOCH):
		ep_loss = 0
		ep_counter = 0
		for batch in train_dataloader :
			optimizer.zero_grad()

			images,texts = batch 

			images= images.to(device)
			texts = texts.to(device)

			# with torch.no_grad():
			text_features = model.encode_text(texts)
			image_features = model.encode_image(images)

			xx,yy =np.meshgrid(np.arange(image_features.shape[0]),np.arange(text_features.shape[0]))
			combinatorial_map = np.stack((xx,yy),axis=2).reshape(-1,2)

			# new_image_features = im_embed_model(image_features)
			# new_text_features = lang_embed_model(text_features)

			expanded_text_features = text_features[combinatorial_map[:,0]]
			expanded_image_features = image_features[combinatorial_map[:,1]]
			#correlation is a torch tensor of shape (expanded_text_features.shape[0]) and is 1 if text_features[combinatorial_map[i,0]] == text_features[combinatorial_map[i,1]] else -1
			correlation = torch.ones(expanded_text_features.shape[0],device=device)
			for i in range(expanded_text_features.shape[0]):
				if torch.equal(text_features[combinatorial_map[i,0]],text_features[combinatorial_map[i,1]]) == False:
					correlation[i] = -1

			# print(expanded_image_features)
			# with torch.no_grad():
			# 	target_text_features = target_model.encode_text(texts)
			# 	target_image_features = target_model.encode_image(images)


			total_loss = loss(expanded_image_features,expanded_text_features,correlation)#+byol_weight*byolLoss(image_features,target_image_features)

			
			ep_loss += total_loss.item()
			ep_counter += 1
			# all_loss.append(total_loss.item())
			total_loss.backward()
			if device == "cpu":
				optimizer.step()
			else :
				# print(model)
				# convert_models_to_fp32(model)
				# convert_models_to_fp32(model)

				optimizer.step()
				# clip.model.convert_weights(model)
		
		all_loss.append(ep_loss/ep_counter)

		val_loss = 0
		counter = 0
		for batch in val_dataloader :
			images,texts = batch 

			images= images.to(device)
			texts = texts.to(device)

			with torch.no_grad():
				text_features = model.encode_text(texts)
				image_features = model.encode_image(images)

			xx,yy =np.meshgrid(np.arange(image_features.shape[0]),np.arange(text_features.shape[0]))
			combinatorial_map = np.stack((xx,yy),axis=2).reshape(-1,2)

			# new_image_features = im_embed_model(image_features)
			# new_text_features = lang_embed_model(text_features)

			expanded_text_features = text_features[combinatorial_map[:,0]]
			expanded_image_features = image_features[combinatorial_map[:,1]]
			#correlation is a torch tensor of shape (expanded_text_features.shape[0]) and is 1 if text_features[combinatorial_map[i,0]] == text_features[combinatorial_map[i,1]] else -1
			correlation = torch.ones(expanded_text_features.shape[0],device=device)
			for i in range(expanded_text_features.shape[0]):
				if torch.equal(text_features[combinatorial_map[i,0]],text_features[combinatorial_map[i,1]]) == False:
					correlation[i] = -1

			# print(expanded_image_features)


			total_loss = loss(expanded_image_features,expanded_text_features,correlation)
			val_loss += total_loss.item()
			counter += 1
		all_val_loss.append(val_loss/counter)

		decay = 1 - (1-DECAY_START) * (np.cos(np.pi*epoch/EPOCH) + 1) / 2
		update_target_model(model,target_model,decay)


		print("EPOCH",epoch,"EPOCH LOSS : ",ep_loss/ep_counter,"VAL LOSS : ",val_loss/counter)


		if epoch % 100 == 0 :
			torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': total_loss,
			}, f"model_checkpoint/cropped_clip_model%04d.pt"% (epoch,)) #just change to your preferred folder/filename
			# torch.save({
			# 'epoch': epoch,
			# 'model_state_dict': lang_embed_model.state_dict(),
			# 'optimizer_state_dict': optimizer.state_dict(),
			# 'loss': total_loss,
			# }, f"model_checkpoint/lang_embed_model%04d.pt"% (epoch,))


		#plot all_loss and all_val_loss in real time
		plt.clf()
		plt.plot(all_loss)
		plt.plot(all_val_loss)
		plt.pause(0.001)




	fig = plt.figure(figsize=(12,8))
	plt.plot(all_loss)
	plt.plot(all_val_loss)
	plt.savefig("byol_loss_%02f.png"%byol_weight)
	

if __name__ == '__main__':
	# for i in [0.001,0.01,0.1,1]:
	# 	print("BYOL WEIGHT : ",i)
	train_frozen_lang()

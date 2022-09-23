from tqdm import tqdm
from torch.utils.data import DataLoader
from DataLoader import *
import os
import torch.nn as nn
import argparse
import pandas as pd
from Utils import *
import Network as models
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Usage: python Evaluation.py --image_triplet_csv pregressor_train_data.csv --im_path ../../UTKFace --regression global --ckpt ../ckpt/global/utk/coral/fold0/utk_coral.pth --testing 1

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_triplet_csv', type=str)
	parser.add_argument('--im_path', type=str)
	parser.add_argument('--batchsize', type=str, default=11)
	parser.add_argument('--regression', type=str)
	parser.add_argument('--ckpt', type=str, default="./")
	parser.add_argument('--testing', type=bool, default=False)
	
	
	args = parser.parse_args()

	if args.regression == 'global':
		args.backbone = 'Global_Regressor'
	elif args.regression == 'local':
		args.backbone = 'Local_Regressor'

	torch.multiprocessing.freeze_support()

	img_df = pd.read_csv(args.image_triplet_csv)

	if (args.testing):
		img_df = img_df.head(10)

	train_df, val_df = train_test_split(img_df, test_size=0.2, stratify=img_df['p_rank'], random_state=10)
	train_df = train_df.reset_index(drop=True)
	val_df = val_df.reset_index(drop=True)

	Images_train = ImageLoaderForTrain(args, train_df)
	Images_val = ImageLoaderForTrain(args, val_df)
	dataloader_Images_train = DataLoader(Images_train, batch_size=args.batchsize, shuffle=True, num_workers=4)
	dataloader_Images_val = DataLoader(Images_val, batch_size=args.batchsize, shuffle=False, num_workers=4)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = models.create_model(args, args.backbone)
	criteria = nn.MSELoss()

	initial_model = os.path.join(args.ckpt)
	print(initial_model)

	### Load network parameters ###
	checkpoint = torch.load(initial_model, map_location=device)
	model_dict = model.state_dict()

	model_dict.update(checkpoint['model_state_dict'])
	model.load_state_dict(model_dict)
	print("=> loaded checkpoint '{}'".format(initial_model))

	global_step = 0
	best_epoch = 0
	best_val_loss = np.inf

	model.eval()

	with torch.no_grad():
		val_loss = 0
		for i, data in enumerate(tqdm(dataloader_Images_val, "Eval")):
			X, y = data
			
			# convert data to float
			# https://stackoverflow.com/questions/67456368/pytorch-getting-runtimeerror-found-dtype-double-but-expected-float
			X = X.float().to(device)
			y = y.float().to(device)
			
			output = torch.squeeze(model('train', x_1_1=X[:, 0, :, :, :], x_1_2=X[:, 1, :, :, :], x_2=X[:, 2, :, :, :]))
			loss = criteria(output, y)
			# output = output.cpu().detach().numpy()
			
			val_loss += loss.item()

	print(f"Validation loss: {val_loss/len(dataloader_Images_val)}")




		
		
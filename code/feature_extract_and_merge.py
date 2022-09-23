from tqdm import tqdm
from torch.utils.data import DataLoader
from DataLoader import *
import os
import torch.nn as nn
import argparse
import pandas as pd
from Utils import *
import Network as models
from DataSelector import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Usage: python feature_extract_and_merge.py --im_path ../../UTKFace --regression local --ckpt ../ckpt/global/utk/coral/fold0/utk_coral.pth --start 0 --data_path ../datalist

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=int, default=0)

	parser.add_argument('--dataset', type=str, default='utk')
	parser.add_argument('--data_path', type=str, default='datalist')
	parser.add_argument('--im_path', type=str)
	parser.add_argument('--reg_num', type=int, default=5)

	parser.add_argument('--experiment_setting', type=str)
	parser.add_argument('--experiment_title', type=str)

	parser.add_argument('--reference_list_path', type=str, default='../datalist/utk/utk_coral_sampled_5p.csv')
	parser.add_argument('--ckpt', type=str, default="./")
	parser.add_argument('--regression', type=str)
	parser.add_argument('--start', type=int)
	parser.add_argument('--number_of_sample_process', type=int, default=500)

	args = parser.parse_args()

	if args.regression == 'global':
		args.backbone = 'Global_Regressor'
	elif args.regression == 'local':
		args.backbone = 'Local_Regressor'

	assert args.regression == 'local'

	train_data, test_data, reg_bound, sampling, sample_rate = data_select(args)

	train_data_sampled = pd.read_csv(args.reference_list_path)[args.start:args.start + args.number_of_sample_process]
	print('Load reference datalist: ', args.reference_list_path)

	train_data = train_data_sampled
	train_data['age'] = train_data['age'].to_numpy().astype('int')

	# Load model
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

	# in local regression setting, features have shape: (n.o local regressor, n.o images, 512)
	features = feature_extraction_local_regression(args, train_data, test_data, model, device)
	np.savez_compressed(f"./train_feature_{args.start}_{args.start + args.number_of_sample_process}.npz", train=features['train'])
	np.savez_compressed(f"./test_feature_{args.start}_{args.start + args.number_of_sample_process}.npz", train=features['test'])

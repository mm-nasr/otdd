from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance
import numpy as np
import torch
import pandas as pd
import argparse

if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

parser = argparse.ArgumentParser(description='Process user specifications.')
parser.add_argument('--type', dest='dist_type', default='between',
                    help='between - distances between dataset \n inter - distances between classes')


args = parser.parse_args()

TO3CHANNELS = True

# Load datasets
DATASETS = ['MNIST', 'FashionMNIST', 'KMNIST', 'CIFAR10', 'SVHN']
# 'USPS', 'SVHN', 'EMNIST'
datasets = {}
n_datasets = len(DATASETS)

for ds_name in DATASETS:
	datasets[ds_name] = load_torchvision_data(ds_name, to3channels=TO3CHANNELS, resize=34, 
											valid_size=0, maxsize = 10000)[0]['train']

if args.dist_type == 'between':
	print('Calculating distance between datasets')
	distances = np.zeros((n_datasets,n_datasets))
	for i, set1 in enumerate(datasets):
		for j, set2 in enumerate(datasets):
			if  i >= j :
				continue
			dist = DatasetDistance(datasets[set1], datasets[set2],
	                       inner_ot_method = 'exact',
	                       debiased_loss = True,
	                       p = 2, entreg = 1e-2,
	                       device=device,
						   chosen_classes_1=None,
						   chosen_classes_2=None)

			d = dist.distance(maxsamples = 5000)
			if TO3CHANNELS:
				d /= 3
			print('OOTD({},{}) = {}'.format(set1,set2,d))
			distances[i,j] = d
			distances[j,i] = d
	dist_df = pd.DataFrame(distances, columns=DATASETS, index=DATASETS)

	#print(distances)
	print(dist_df)


elif args.dist_type == 'inter':
	print('Calculating distance between class groups of MNIST and CIFAR10')
	## Now do inter-dataset distance calculations
	classes_div = [[0,1], [2,3], [4,5], [6,7], [8,9]]

	for ds_name in ['MNIST', 'CIFAR10']:
		distances = np.zeros((len(classes_div),len(classes_div)))
		col_names = []
		for i in range(len(classes_div)):
			for j in range(i, len(classes_div)):
				classes1 = classes_div[i]
				classes2 = classes_div[j]
				dist = DatasetDistance(datasets[ds_name], datasets[ds_name],
		               inner_ot_method = 'exact',
		               debiased_loss = True,
		               p = 2, entreg = 1e-1,
		               device=device,
					   chosen_classes_1 = classes1,
					   chosen_classes_2 = classes2)

				d = dist.distance(maxsamples = 5000)
				if TO3CHANNELS:
					d /= 3
				print('OOTD({},{}) = {}'.format(ds_name,ds_name,d))
				distances[i,j] = d
				distances[j,i] = d

			col_names.append(ds_name + str(classes1))

		dist_df = pd.DataFrame(distances, columns=col_names, index=col_names)

		print(distances)
		print(dist_df)

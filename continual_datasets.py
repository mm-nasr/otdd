from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance
import numpy as np
import torch
import pandas as pd

if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')


# Load datasets
DATASETS = ['MNIST','KMNIST', 'FashionMNIST', 'SVHN', 'CIFAR10']
# 'USPS'
datasets = {}
n_datasets = len(DATASETS)

for ds_name in DATASETS:
	datasets[ds_name] = load_torchvision_data(ds_name, to3channels=True, resize=32, 
											valid_size=0, maxsize = 5000)[0]['train']

distances = np.zeros((n_datasets,n_datasets))
for i, set1 in enumerate(datasets):
	for j, set2 in enumerate(datasets):
		if  i >= j :
			continue
		dist = DatasetDistance(datasets[set1], datasets[set2],
                       inner_ot_method = 'exact',
                       debiased_loss = True,
                       p = 2, entreg = 1e-1,
                       device=device,
					   chosen_classes_1=None,
					   chosen_classes_2=None)

		d = dist.distance(maxsamples = 3000)
		print('OOTD({},{}) = {}'.format(set1,set2,d))
		distances[i,j] = d
		distances[j,i] = d
dist_df = pd.DataFrame(distances, columns=DATASETS, index=DATASETS)

#print(distances)
print(dist_df)

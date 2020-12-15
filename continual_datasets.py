from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance
import numpy as np

if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')


# Load datasets
DATASETS = ['MNIST', 'FashionMNIST', 'CIFAR10', 'KMNIST']
datasets_loaded = {}
n_datasets = len(DATASETS)

for ds_name in DATASETS:
	datasets[ds_name] = load_torchvision_data(ds_name, to3channels=True, resize=32, 
											valid_size=0, maxsize = 2000)[0]['train']

distances = np.zeros(n_datasets)
for i, set1 in enumerate(datasets):
	for j, set2 in enumerate(datasets):
		dist = DatasetDistance(datasets[set1], datasets[set2],
                       inner_ot_method = 'exact',
                       debiased_loss = True,
                       p = 2, entreg = 1e-1,
                       device=device)

		distances[i][j] = dist.distance(maxsamples = 1000)
		print('OOTD({},{}) = {}'.format(set1,set2,distances[i][j]))

print(distances)
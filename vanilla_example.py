from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance
import torch
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Load datasets
loaders_src = load_torchvision_data('MNIST', valid_size=0, resize = 32, to3channels=True, maxsize=2000)[0]
loaders_tgt = load_torchvision_data('FashionMNIST',  valid_size=0, resize = 32, to3channels=True, maxsize=2000)[0]
#print(np.shape(loaders_src['train']))

# Instantiate distance
dist = DatasetDistance(loaders_src['train'], loaders_tgt['train'],
                       inner_ot_method = 'exact',
                       debiased_loss = True,
                       p = 2, entreg = 1e-1,
                       device=device)

d = dist.distance(maxsamples = 1000)
print(f'OTDD(src,tgt)={d}')

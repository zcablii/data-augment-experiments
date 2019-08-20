import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import *
import math
import yaml
from easydict import EasyDict

def imshow(img):
    # img = img / 2 + 0.5    
    # npimg = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

with open('./config.yaml') as f:
    config = yaml.load(f)
    # convert to dict
    config = EasyDict(config)

transform= transforms.Compose(
           data_augmentation(config))

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

# get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

holes = get_holes(4)
count = 1

for batch_index, (inputs, targets) in enumerate(trainloader):
    inputs, labels, lams = randmix_data(
        holes, inputs, targets, 'cpu')
# show images
    if count == 1:
        imshow(torchvision.utils.make_grid(inputs))
        count =0
    else:
        break

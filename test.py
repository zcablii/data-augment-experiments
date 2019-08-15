import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import Cutout, rand_bbox, mixup_data
import math
           

def imshow(img):
    # img = img / 2 + 0.5    
    # npimg = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


def get_holes(points_num):
    center = (16,16)
    r = 12
    perimeter = 2*r*math.pi
    slot=perimeter/points_num
    x_vals=[]
    y_vals=[]
    for i in range(points_num):
        interval = slot*i+random.uniform(0, slot/2)
        angle = interval/r
        x=center[0]+r*math.cos(angle)
        y=center[1]+r*math.sin(angle)
        x_vals.append(round(x))
        y_vals.append(round(y))

    holes = []
    tot_area = 0
    for i in range(len(x_vals)):
        length = random.randint(14,20)
        x = x_vals[i]
        y = y_vals[i]
        y1 = np.clip(y - length // 2, 0, 32)
        y2 = np.clip(y + length // 2, 0, 32)
        x1 = np.clip(x - length // 2, 0, 32)
        x2 = np.clip(x + length // 2, 0, 32)
        area = (y2-y1)*(x2-x1)
        hole = [y1, y2, x1, x2, area]
        holes.append(hole)

    
    return holes

def randmix_data(holes, inputs, target, device):
    mask = np.zeros((inputs.size()[0], 3,32, 32), np.float32)
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(inputs)
    permu = []
    targets = []
    lams = []
    tot_area = 0
    for i in range(len(holes)):
        tot_area += holes[i][4]
    
    for i in range(len(holes)):
        rand_index = torch.randperm(inputs.size()[0])
        permu.append(rand_index)
        targets.append(target[rand_index])
        lams.append(holes[i][4]/tot_area)
        y21 = holes[i][0]
        y22 = holes[i][1]
        x21 = holes[i][2]
        x22 = holes[i][3]
        mask[:, :, y21: y22, x21: x22] = inputs[rand_index, :, y21: y22, x21: x22] 
        
        if i > 0:
            for k in range (i):
                y11 = holes[k][0]
                y12 = holes[k][1]
                x11 = holes[k][2]
                x12 = holes[k][3]
                if y12<y21 or y11 > y22 or x11>x22 or x12<x21:
                    continue
                elif x12>x22 and y11>=y21 and y12<=y22:
                    mask[:, :, y11:y12, x11:x22] = 0.7*inputs[rand_index, :, y11:y12, x11:x22] + 0.3*inputs[permu[k], :, y11:y12, x11:x22]
                elif x12>x22 and y11<=y21 and y12>=y22:
                    mask[:, :, y21:y22, x11:x22] = 0.7*inputs[rand_index, :, y21:y22, x11:x22] + 0.3*inputs[permu[k], :, y21:y22, x11:x22]
                elif x11<x21 and y11>=y21 and y12<=y22:
                    mask[:, :, y11:y12, x21:x12] = 0.7*inputs[rand_index, :, y11:y12, x21:x12] + 0.3*inputs[permu[k], :, y11:y12, x21:x12]
                elif x11<x21 and y11<=y21 and y12>=y22:
                    mask[:, :, y21:y22, x21:x12] = 0.7*inputs[rand_index, :, y21:y22, x21:x12] + 0.3*inputs[permu[k], :, y21:y22, x21:x12]

                elif y12>y22 and x11>=x21 and x12 <= x22:
                    mask[:, :, y11:y22, x11:x12] = 0.7*inputs[rand_index, :, y11:y22, x11:x12] + 0.3*inputs[permu[k], :, y11:y22, x11:x12]
                elif y12>y22 and x11<=x21 and x12 >= x22:
                    mask[:, :, y11:y22, x21:x22] = 0.7*inputs[rand_index, :, y11:y22, x21:x22] + 0.3*inputs[permu[k], :, y11:y22, x21:x22]
                elif y12>y22 and x11<x21 and x12 > x21:
                    mask[:, :, y11:y22, x21:x12] = 0.7*inputs[rand_index, :, y11:y22, x21:x12] + 0.3*inputs[permu[k], :, y11:y22, x21:x12]
                elif y12>y22 and x11>x21 and x12 > x22:
                    mask[:, :, y11:y22, x11:x22] = 0.7*inputs[rand_index, :, y11:y22, x11:x22] + 0.3*inputs[permu[k], :, y11:y22, x11:x22]
                elif y21>y11 and x11>=x21 and x12 <= x22:
                    mask[:, :, y21:y12, x11:x12] = 0.7*inputs[rand_index, :, y21:y12, x11:x12] + 0.3*inputs[permu[k], :, y21:y12, x11:x12]
                elif y21>y11 and x11<=x21 and x12 >= x22:
                    mask[:, :, y21:y12, x21:x22] = 0.7*inputs[rand_index, :,y21:y12, x21:x22] + 0.3*inputs[permu[k], :, y21:y12, x21:x22]
                elif y21>y11 and x11<x21 and x12 > x21 and x12 < x22:
                    mask[:, :, y21:y12, x21:x12] = 0.7*inputs[rand_index, :,y21:y12, x21:x12] + 0.3*inputs[permu[k], :, y21:y12, x21:x12]
                elif y21>y11 and x11>x21 and x12 > x22:
                    mask[:, :, y21:y12, x11:x22] = 0.7*inputs[rand_index, :,y21:y12, x11:x22] + 0.3*inputs[permu[k], :, y21:y12, x11:x22]
                else:
                    continue

    return mask,targets,lams
        

def randmix_criterion(criterion, pred, targets, lams):
    loss =0
    for i in range(len(targets)):
        loss+=lams[i]*criterion(pred, targets[i])

    return loss
    
'''
def cutmix_data(input, target, alpha, device):

    rand_index = torch.randperm(input.size()[0])

    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)

    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]

    # compute output
    input_var = torch.autograd.Variable(input, requires_grad=True)
    targets_a = torch.autograd.Variable(target_a)
    targets_b = torch.autograd.Variable(target_b)
    return input, targets_a, targets_b, lam


transform = transforms.Compose(
    [transforms.ToTensor()])

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
'''

holes = get_holes(5)
print(holes)
'''
count = 1

for batch_index, (inputs, targets) in enumerate(trainloader):
    inputs = randcut_mix(
        holes, inputs, targets, 'cpu')

# show images
    if count == 1:
        imshow(torchvision.utils.make_grid(inputs))
        count =0
    else:
        break
'''
 # -*-coding:utf-8-*-
import os
import math
import shutil
import logging
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class Logger(object):
    def __init__(self, log_file_name, log_level, logger_name):
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] - [%(filename)s line:%(lineno)d] : %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def data_augmentation(config, is_train=True):
    aug = []
    if is_train:
        # random crop
        if config.augmentation.random_crop:
            aug.append(transforms.RandomCrop(config.input_size, padding=4))
        # horizontal filp
        if config.augmentation.random_horizontal_filp:
            aug.append(transforms.RandomHorizontalFlip())

    aug.append(transforms.ToTensor())
    # normalize  [- mean / std]
    if config.augmentation.normalize:
        if config.dataset == 'cifar10':
            aug.append(transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        else:
            aug.append(transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))

    if is_train and config.augmentation.cutout:
        # cutout
        aug.append(Cutout(n_holes=config.augmentation.holes,
                          length=config.augmentation.length))
    return aug


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '.pth.tar')
    if is_best:
        shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')


def load_checkpoint(path, model, optimizer=None):
    if os.path.isfile(path):
        logging.info("=== loading checkpoint '{}' ===".format(path))

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        if optimizer is not None:
            best_prec = checkpoint['best_prec']
            last_epoch = checkpoint['last_epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info("=== done. also loaded optimizer from " +
                         "checkpoint '{}' (epoch {}) ===".format(
                             path, last_epoch + 1))
            return best_prec, last_epoch


def get_data_loader(transform_train, transform_test, config):
    assert config.dataset == 'cifar10' or config.dataset == 'cifar100'
    if config.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=config.data_path, train=True,
            download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR10(
            root=config.data_path, train=False,
            download=True, transform=transform_test)
    else:
        trainset = torchvision.datasets.CIFAR100(
            root=config.data_path, train=True,
            download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR100(
            root=config.data_path, train=False,
            download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.workers)

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=config.test_batch,
        shuffle=False, num_workers=config.workers)
    return train_loader, test_loader


def mixup_data(x, y, alpha, device):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(input, target, alpha, device):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(input.size()[0]).cuda()
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
    # compute output
    input_var = torch.autograd.Variable(input, requires_grad=True)
    targets_a = torch.autograd.Variable(target_a)
    targets_b = torch.autograd.Variable(target_b)
    return input_var, targets_a, targets_b, lam

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def imshow(img):
    # img = img / 2 + 0.5    
    # npimg = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

def get_area(y1, y2, x1, x2):
    return (y2-y1)*(x2-x1)

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
        hole = [y1, y2, x1, x2]
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
    r1 = 0.7
    r2 = 1-r1
    # for i in range(len(holes)):
    #     tot_area += holes[i][4]
    
    for i in range(len(holes)):
        rand_index = torch.randperm(inputs.size()[0])
        permu.append(rand_index)
        targets.append(target[rand_index])
        # lams.append(holes[i][4]/tot_area)
        y21 = holes[i][0]
        y22 = holes[i][1]
        x21 = holes[i][2]
        x22 = holes[i][3]
        mask[:, :, y21: y22, x21: x22] = inputs[rand_index, :, y21: y22, x21: x22] 
        lams[i] = get_area(y21, y22, x21, x22)
        tot_area+= lams[i]
        if i > 0:
            for k in range (i):
                y11 = holes[k][0]
                y12 = holes[k][1]
                x11 = holes[k][2]
                x12 = holes[k][3]
                if y12<y21 or y11 > y22 or x11>x22 or x12<x21:
                    continue
                elif x12>x22 and y11>=y21 and y12<=y22:
                    mask[:, :, y11:y12, x11:x22] = r1*inputs[rand_index, :, y11:y12, x11:x22] + r2*inputs[permu[k], :, y11:y12, x11:x22]
                    tot_area-=get_area(y11, y12, x11, x22)
                    lams[k]-=r1*get_area(y11, y12, x11, x22)
                    lams[i]-=r2*get_area(y11, y12, x11, x22)
                elif x12>x22 and y11<=y21 and y12>=y22:
                    mask[:, :, y21:y22, x11:x22] = r1*inputs[rand_index, :, y21:y22, x11:x22] + r2*inputs[permu[k], :, y21:y22, x11:x22]
                    tot_area-=get_area(y21, y22, x11, x22)
                    lams[k]-=r1*get_area(y21, y22, x11, x22)
                    lams[i]-=r2*get_area(y21, y22, x11, x22)
                elif x11<x21 and y11>=y21 and y12<=y22:
                    mask[:, :, y11:y12, x21:x12] = r1*inputs[rand_index, :, y11:y12, x21:x12] + r2*inputs[permu[k], :, y11:y12, x21:x12]
                    tot_area-=get_area(y11, y12, x21, x12)
                    lams[k]-=r1*get_area(y11, y12, x21, x12)
                    lams[i]-=r2*get_area(y11, y12, x21, x12)
                elif x11<x21 and y11<=y21 and y12>=y22:
                    mask[:, :, y21:y22, x21:x12] = r1*inputs[rand_index, :, y21:y22, x21:x12] + r2*inputs[permu[k], :, y21:y22, x21:x12]
                    tot_area-=get_area(y21, y22, x21, x12)
                    lams[k]-=r1*get_area(y21, y22, x21, x12)
                    lams[i]-=r2*get_area(y21, y22, x21, x12)
                elif y12>y22 and x11>=x21 and x12 <= x22:
                    mask[:, :, y11:y22, x11:x12] = r1*inputs[rand_index, :, y11:y22, x11:x12] + r2*inputs[permu[k], :, y11:y22, x11:x12]
                    tot_area-=get_area(y11, y22, x11, x12)
                    lams[k]-=r1*get_area(y11, y22, x11, x12)
                    lams[i]-=r2*get_area(y11, y22, x11, x12)
                elif y12>y22 and x11<=x21 and x12 >= x22:
                    mask[:, :, y11:y22, x21:x22] = r1*inputs[rand_index, :, y11:y22, x21:x22] + r2*inputs[permu[k], :, y11:y22, x21:x22]
                    tot_area-=get_area(y11, y22, x21, x22)
                    lams[k]-=r1*get_area(y11, y22, x21, x22)
                    lams[i]-=r2*get_area(y11, y22, x21, x22)
                elif y12>y22 and x11<x21 and x12 > x21:
                    mask[:, :, y11:y22, x21:x12] = r1*inputs[rand_index, :, y11:y22, x21:x12] + r2*inputs[permu[k], :, y11:y22, x21:x12]
                    tot_area-=get_area(y11, y22, x21, x12)
                    lams[k]-=r1*get_area(y11, y22, x21, x12)
                    lams[i]-=r2*get_area(y11, y22, x21, x12)
                elif y12>y22 and x11>x21 and x12 > x22:
                    mask[:, :, y11:y22, x11:x22] = r1*inputs[rand_index, :, y11:y22, x11:x22] + r2*inputs[permu[k], :, y11:y22, x11:x22]
                    tot_area-=get_area(y11, y22, x11, x22)
                    lams[k]-=r1*get_area(y11, y22, x11, x22)
                    lams[i]-=r2*get_area(y11, y22, x11, x22)
                elif y21>y11 and x11>=x21 and x12 <= x22:
                    mask[:, :, y21:y12, x11:x12] = r1*inputs[rand_index, :, y21:y12, x11:x12] + r2*inputs[permu[k], :, y21:y12, x11:x12]
                    tot_area-=get_area(y21, y12, x11, x12)
                    lams[k]-=r1*get_area(y21, y12, x11, x12)
                    lams[i]-=r2*get_area(y21, y12, x11, x12)
                elif y21>y11 and x11<=x21 and x12 >= x22:
                    mask[:, :, y21:y12, x21:x22] = r1*inputs[rand_index, :,y21:y12, x21:x22] + r2*inputs[permu[k], :, y21:y12, x21:x22]
                    tot_area-=get_area(y21, y12, x21, x22)
                    lams[k]-=r1*get_area(y21, y12, x21, x22)
                    lams[i]-=r2*get_area(y21, y12, x21, x22)
                elif y21>y11 and x11<x21 and x12 > x21 and x12 < x22:
                    mask[:, :, y21:y12, x21:x12] = r1*inputs[rand_index, :,y21:y12, x21:x12] + r2*inputs[permu[k], :, y21:y12, x21:x12]
                    tot_area-=get_area(y21, y12, x21, x12)
                    lams[k]-=r1*get_area(y21, y12, x21, x12)
                    lams[i]-=r2*get_area(y21, y12, x21, x12)
                elif y21>y11 and x11>x21 and x12 > x22:
                    mask[:, :, y21:y12, x11:x22] = r1*inputs[rand_index, :,y21:y12, x11:x22] + r2*inputs[permu[k], :, y21:y12, x11:x22]
                    tot_area-=get_area(y21, y12, x11, x22)
                    lams[k]-=r1*get_area(y21, y12, x11, x22)
                    lams[i]-=r2*get_area(y21, y12, x11, x22)
                else:
                    continue

    lams=lams/sum(lams)
    return mask,targets,lams
        

def randmix_criterion(criterion, pred, targets, lams):
    loss =0
    for i in range(len(targets)):
        loss+=lams[i]*criterion(pred, targets[i])

    return loss

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def adjust_learning_rate(optimizer, epoch, config):
    lr = get_current_lr(optimizer)
    if config.lr_scheduler.type == 'STEP':
        if epoch in config.lr_scheduler.lr_epochs:
            lr *= config.lr_scheduler.lr_mults
    elif config.lr_scheduler.type == 'COSINE':
        ratio = epoch / config.epochs
        lr = config.lr_scheduler.min_lr + \
            (config.lr_scheduler.base_lr - config.lr_scheduler.min_lr) * \
            (1.0 + math.cos(math.pi * ratio)) / 2.0
    elif config.lr_scheduler.type == 'HTD':
        ratio = epoch / config.epochs
        lr = config.lr_scheduler.min_lr + \
            (config.lr_scheduler.base_lr - config.lr_scheduler.min_lr) * \
            (1.0 - math.tanh(
                config.lr_scheduler.lower_bound
                + (config.lr_scheduler.upper_bound
                   - config.lr_scheduler.lower_bound)
                * ratio)
             ) / 2.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

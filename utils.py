#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import os.path
import copy
import PIL
import torch
import wget
from zipfile import ZipFile
from torchvision import datasets, transforms
from data_utils.sampling import imagenet_iid, imagenet_noniid


def parse_classes(file):
    classes = []
    filenames = []
    with open(file) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    for x in range(len(lines)):
        tokens = lines[x].split()
        classes.append(tokens[1])
        filenames.append(tokens[0])
    return filenames, classes

class TinyImageNetDataset(torch.utils.data.Dataset):
    """Dataset wrapping images and ground truths."""
    def __init__(self, img_path, gt_path, class_to_idx=None, transform=None):
        self.img_path = img_path
        self.transform = transform
        self.gt_path = gt_path
        self.class_to_idx = class_to_idx
        self.classidx = []
        self.imgs, self.classnames = parse_classes(gt_path)
        for classname in self.classnames:
            self.classidx.append(self.class_to_idx[classname])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, y) where y is the label of the image.
        """
        img = None
        with open(os.path.join(self.img_path, self.imgs[index]), 'rb') as f:
            img = PIL.Image.open(f).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        y = self.classidx[index]
        return img, y

    def __len__(self):
        return len(self.imgs)


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
                
    if args.dataset == 'imagenet':

        TINY_IMAGENET_ROOT = 'data/tiny-imagenet/'
        if os.path.exists('tiny-imagenet-200.zip') == False:
            url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
            tiny_imgdataset = wget.download(url, out = os.getcwd())
            with ZipFile('tiny-imagenet-200.zip', 'r') as zip_ref:
                zip_ref.extractall('data/')


        train_dataset = datasets.ImageFolder(
            os.path.join('data/', 'tiny-imagenet-200', 'train'),
            transform=transforms.Compose(
                [
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            )
        )
        test_dataset = TinyImageNetDataset(
            img_path=os.path.join('data/', 'tiny-imagenet-200', 'val', 'images'), 
            gt_path=os.path.join('data/', 'tiny-imagenet-200', 'val', 'val_annotations.txt'),
            class_to_idx=train_dataset.class_to_idx.copy(),
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            )
        )
        
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = imagenet_iid(train_dataset, args.num_users)
        else:
            # Chose euqal splits for every user
            user_groups = imagenet_noniid(train_dataset, args.num_users, args.alpha)


    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

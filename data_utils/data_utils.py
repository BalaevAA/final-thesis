from collections import OrderedDict
from typing import Tuple

import os
import sys
import glob
import wget
from zipfile import ZipFile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch import Tensor
from PIL import Image
import numpy as np


class TinyImageNet(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        NUM_IMAGES_PER_CLASS = 500
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split_dir = os.path.join(self.root, split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.JPEG'), recursive=True))
        
        self.labels = {}  # fname - label number mapping

        # build class label - number mapping
        with open(os.path.join(self.root, 'wnids.txt'), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    self.labels[f'{label_text}_{cnt}.JPEG'] = i
        elif split == 'val':
            with open(os.path.join(self.split_dir, 'val_annotations.txt'), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]
                    
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]
        label = self.labels[os.path.basename(file_path)]
        label = self.target_transform(label) if self.target_transform else label
        return self.read_image(file_path), label

    def read_image(self, path):
        img = Image.open(path)
        return self.transform(img) if self.transform else img


def one_hot(labels, classes):
    return np.eye(classes)[labels]


def load_data():

    # load dataset and split users
    TINY_IMAGENET_ROOT = 'data/tiny-imagenet-200/'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    tiny_imgdataset = wget.download(url, out = os.getcwd())
    with ZipFile('tiny-imagenet-200.zip', 'r') as zip_ref:
        zip_ref.extractall('data/')


    # Load ImageNet and normalize
    traindir = os.path.join(TINY_IMAGENET_ROOT, 'train')
    valdir = os.path.join(TINY_IMAGENET_ROOT, 'val')


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    dataset_train = TinyImageNet(
        TINY_IMAGENET_ROOT, 
        'train', 
        transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                    ]))
    dataset_test = TinyImageNet(
        TINY_IMAGENET_ROOT, 
        'val', transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        target_transform=lambda target: one_hot(target, 200))

    return dataset_train, dataset_test


def split_data(args, raw_train):

    if args.split_type == 'iid':
        shuffled_indices = np.random.permutation(len(raw_train))
        split_indices = np.array_split(shuffled_indices, args.K)
        split_map = {i: split_indices[i] for i in range(args.K)}
        return split_map
    elif args.split_type == 'dirichlet':        
        split_map = dict()
        client_indices_list = [[] for _ in range(args.K)]
        for c in range(args.num_classes):
            target_class_indices = np.where(np.array(raw_train.targets) == c)[0]
            np.random.shuffle(target_class_indices)
            proportions = np.random.dirichlet(np.repeat(args.alpha, args.K))
            proportions = np.array([p * (len(idx) < len(raw_train) / args.K) for p, idx in zip(proportions, client_indices_list)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(target_class_indices)).astype(int)[:-1]
            idx_split = np.array_split(target_class_indices, proportions)
            client_indices_list = [j + idx.tolist() for j, idx in zip(client_indices_list, idx_split)]
        for j in range(args.K):
            np.random.seed(args.global_seed); np.random.shuffle(client_indices_list[j])
            if len(client_indices_list[j]) > 10:
                split_map[j] = client_indices_list[j]
        return split_map
    
    


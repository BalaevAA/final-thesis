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
    """
    Contains 200 classes for training. Each class has 500 images. 
    Parameters
    ----------
    root: string
        Root directory including `train` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    """
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
    """Load ImageNet (training and val set).
    
    
      !wget --no-clobber http://cs231n.stanford.edu/tiny-imagenet-200.zip
      !unzip -n tiny-imagenet-200.zip
      TINY_IMAGENET_ROOT = './tiny-imagenet-200/'
    
    """

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
    """Split data indices using labels.
    
    Args:
        args (argparser): arguments
        raw_train (dataset): raw dataset object to parse 
        
    Returns:
        split_map (dict): dictionary with key is a client index (~args.K) and a corresponding value is a list of indice array
    """
    # IID split (i.e., statistical homogeneity)
    if args.split_type == 'iid':
        # randomly shuffle label indices
        shuffled_indices = np.random.permutation(len(raw_train))
        
        # split shuffled indices by the number of clients
        split_indices = np.array_split(shuffled_indices, args.K)
        
        # construct a hashmap
        split_map = {i: split_indices[i] for i in range(args.K)}
        return split_map
    
    # Non-IID split proposed in Hsu et al., 2019 (i.e., using Dirichlet distribution to simulate non-IID split)
    # https://github.com/QinbinLi/FedKT/blob/0bb9a89ea266c057990a4a326b586ed3d2fb2df8/experiments.py
    elif args.split_type == 'dirichlet':        
        split_map = dict()

        # container
        client_indices_list = [[] for _ in range(args.K)]

        # iterate through all classes
        for c in range(args.num_classes):
            # get corresponding class indices
            target_class_indices = np.where(np.array(raw_train.targets) == c)[0]

            # shuffle class indices
            np.random.shuffle(target_class_indices)

            # get label retrieval probability per each client based on a Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(args.alpha, args.K))
            proportions = np.array([p * (len(idx) < len(raw_train) / args.K) for p, idx in zip(proportions, client_indices_list)])

            # normalize
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(target_class_indices)).astype(int)[:-1]

            # split class indices by proportions
            idx_split = np.array_split(target_class_indices, proportions)
            client_indices_list = [j + idx.tolist() for j, idx in zip(client_indices_list, idx_split)]

        # shuffle finally and create a hashmap
        for j in range(args.K):
            np.random.seed(args.global_seed); np.random.shuffle(client_indices_list[j])
            if len(client_indices_list[j]) > 10:
                split_map[j] = client_indices_list[j]
        return split_map
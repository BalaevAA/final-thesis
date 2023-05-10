#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
from collections import defaultdict
import random


def imagenet_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def imagenet_noniid(dataset, no_participants, alpha=0.9):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    np.random.seed(666)
    random.seed(666)
    iamgenet_classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        if label in iamgenet_classes:
            iamgenet_classes[label].append(ind)
        else:
            iamgenet_classes[label] = [ind]

    per_participant_list = defaultdict(list)
    no_classes = len(iamgenet_classes.keys())
    class_size = len(iamgenet_classes[0])
    datasize = {}
    for n in range(no_classes):
        random.shuffle(iamgenet_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(no_participants * [alpha]))
        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            datasize[user, n] = no_imgs
            sampled_list = iamgenet_classes[n][:min(len(iamgenet_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            iamgenet_classes[n] = iamgenet_classes[n][min(len(iamgenet_classes[n]), no_imgs):]
    train_img_size = np.zeros(no_participants)
    for i in range(no_participants):
        train_img_size[i] = sum([datasize[i,j] for j in range(200)])
    class_weight = np.zeros((no_participants,200))
    for i in range(no_participants):
        for j in range(200):
            class_weight[i,j] = float(datasize[i,j])/float((train_img_size[i]))
    return per_participant_list, class_weight


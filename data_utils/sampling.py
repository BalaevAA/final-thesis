import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import random


##########################
##  Tiny-ImageNet-200   ##
##########################


def imagenet_iid(dataset, num_users):

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def imagenet_noniid(dataset, no_participants, alpha=0.9):
    np.random.seed(666)
    random.seed(666)
    for_graph = {}
    im_classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        if label in im_classes:
            im_classes[label].append(ind)
        else:
            im_classes[label] = [ind]

    per_participant_list = defaultdict(list)
    no_classes = len(im_classes.keys())
    class_size = len(im_classes[0])
    datasize = {}
    for n in range(no_classes):
        random.shuffle(im_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(np.array(no_participants * [alpha]))
        if n in for_graph.keys():
            for_graph[n].append(sampled_probabilities)

        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            datasize[user, n] = no_imgs
            sampled_list = im_classes[n][:min(len(im_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            im_classes[n] = im_classes[n][min(len(im_classes[n]), no_imgs):]
    train_img_size = np.zeros(no_participants)
    for i in range(no_participants):
        train_img_size[i] = sum([datasize[i,j] for j in range(200)])
    class_weight = np.zeros((no_participants,200))
    for i in range(no_participants):
        for j in range(200):
            class_weight[i,j] = float(datasize[i,j])/float((train_img_size[i]))
    return per_participant_list, class_weight, for_graph




#######################
##      CIFAR10      ## 
#######################


def cifar_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, no_participants, alpha=0.9):
    np.random.seed(666)
    random.seed(666)
    cifar_classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        if label in cifar_classes:
            cifar_classes[label].append(ind)
        else:
            cifar_classes[label] = [ind]

    per_participant_list = defaultdict(list)
    no_classes = len(cifar_classes.keys())
    class_size = len(cifar_classes[0])
    datasize = {}
    for n in range(no_classes):
        random.shuffle(cifar_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(no_participants * [alpha]))
        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            datasize[user, n] = no_imgs
            sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
    train_img_size = np.zeros(no_participants)
    for i in range(no_participants):
        train_img_size[i] = sum([datasize[i,j] for j in range(10)])
    class_weight = np.zeros((no_participants,10))
    for i in range(no_participants):
        for j in range(10):
            class_weight[i,j] = float(datasize[i,j])/float((train_img_size[i]))
    return per_participant_list, class_weight


##################################
##                              ##
##   cluster-based partitions   ##     
##                              ##
##################################


def iid_cluster_based(dataset, no_participants):
    np.random.seed(666)
    random.seed(666)

    cifar_classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        if label in cifar_classes:
            cifar_classes[label].append(ind)
        else:
            cifar_classes[label] = [ind]

    per_participant_list = {}
    class_counts_dict = {}
    no_classes = len(cifar_classes.keys())

    for n in range(no_classes):
        random.shuffle(cifar_classes[n])

    cluster_data = np.concatenate(list(cifar_classes.values())).reshape(-1, 1)
    kmeans = KMeans(n_clusters=no_participants, random_state=666)
    cluster_labels = kmeans.fit_predict(cluster_data)

    for user in range(no_participants):
        participant_indices = np.where(cluster_labels == user)[0]
        participant_samples = cluster_data[participant_indices].flatten()
        random.shuffle(participant_samples)

        per_participant_list[user] = []
        class_counts_dict[user] = {}

        for class_idx in range(no_classes):
            class_samples = cifar_classes[class_idx]
            no_imgs = min(len(class_samples), len(participant_samples) // no_classes)

            sampled_list = class_samples[:no_imgs]
            per_participant_list[user].extend(sampled_list)
            cifar_classes[class_idx] = class_samples[no_imgs:]

            class_counts_dict[user][class_idx] = no_imgs

    return per_participant_list, class_counts_dict


def noniid_cluster_based(dataset, no_participants):
    np.random.seed(666)
    random.seed(666)

    # Create a dictionary to store images of each class
    cifar_classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        if label in cifar_classes:
            cifar_classes[label].append(ind)
        else:
            cifar_classes[label] = [ind]

    per_participant_list = defaultdict(list)
    per_participant_dict = {}  # Dictionary to store class-wise indices for each participant
    class_counts_dict = {}  # Dictionary to store class-wise image counts for each participant
    no_classes = len(cifar_classes.keys())
    class_size = len(cifar_classes[0])
    datasize = defaultdict(lambda: defaultdict(int))

    # Shuffle the images within each class
    for n in range(no_classes):
        random.shuffle(cifar_classes[n])

    # Perform clustering on the class data to generate partitions
    cluster_data = np.concatenate(list(cifar_classes.values())).reshape(-1, 1)
    kmeans = KMeans(n_clusters=min(no_classes, no_participants), random_state=666)
    cluster_labels = kmeans.fit_predict(cluster_data)

    class_counts = np.array([len(cifar_classes[i]) for i in range(no_classes)])

    alpha = np.random.rand(no_participants, no_classes)

    for user in range(no_participants):
        participant_indices = np.where(cluster_labels == user)[0]
        participant_samples = cluster_data[participant_indices].flatten()
        random.shuffle(participant_samples)

        class_counts_per_user = class_counts // no_participants
        remainder = class_counts % no_participants

        for class_idx in range(no_classes):
            class_samples = cifar_classes[class_idx]

            if len(class_samples) > 0:
                no_imgs = int(round(alpha[user, class_idx] * class_counts_per_user[class_idx]))
                if user < remainder[class_idx]:
                    no_imgs += 1

                no_imgs = min(no_imgs, len(class_samples))

                sampled_list = class_samples[:no_imgs]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[class_idx] = class_samples[no_imgs:]

                if class_idx in per_participant_dict:
                    per_participant_dict[class_idx].extend(sampled_list)
                else:
                    per_participant_dict[class_idx] = sampled_list

                datasize[user][class_idx] = no_imgs

        class_counts_dict[user] = {class_idx: len(per_participant_dict[class_idx]) for class_idx in per_participant_dict}

    train_img_size = np.zeros(no_participants)
    for i in range(no_participants):
        train_img_size[i] = sum([datasize[i][j] for j in range(no_classes)])

    class_weight = np.zeros((no_participants, no_classes))
    for i in range(no_participants):
        for j in range(no_classes):
            class_weight[i, j] = float(datasize[i][j]) / float(train_img_size[i])

    return per_participant_list, class_weight, class_counts_dict

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from tqdm import tqdm

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.targets = dataset.targets
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdateWithLocalsData(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs),
                                batch_size=self.args.local_bs, shuffle=True)
        
    def train_val_split(self, dataset, idxs):
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.4*len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                batch_size=self.args.local_bs, shuffle=True)
        return trainloader

    def train(self, net):
        net.train()
        # train and update
        print('\nstart local train\n')
        
        if self.args.optimizer == 'sgd':
            # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=1e-4)
        elif self.args.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=6e-5)
        
        epoch_loss = []
        for iter in tqdm(range(self.args.local_ep)):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdateWithDataGen(object):
    def __init__(self, args, dataset=None, idx_cur_user=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idx_cur_user),
                                batch_size=self.args.local_bs, shuffle=True)
        
    def train_val_split(self, dataset, idxs):
        idxs_train = idxs[:int(0.4*len(idxs))]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                batch_size=self.args.local_bs, shuffle=True)

        return trainloader
    
    def make_iid(dataset, num_users):
        num_items = int(len(dataset)/num_users)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        return dict_users

    def train(self, net):
        net.train()
        # train and update
        print('\nstart local train\n')
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=1e-4)
        elif self.args.optimizer == 'rms':
            optimizer = torch.optim.RMSprop(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=6e-5)
        
        epoch_loss = []
        for iter in tqdm(range(self.args.local_ep)):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    


import os
import os.path
from tqdm import tqdm
import wget
from zipfile import ZipFile
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import pickle
from data_utils.sampling import imagenet_iid, imagenet_noniid, cifar_iid, cifar_noniid, noniid_cluster_based, iid_cluster_based
from data_utils.make_boxplot import create_boxplot
from data_utils.make_hist import create_hist_curves
from config.options import args_parser
from models.Local_train import LocalUpdateWithLocalsData, LocalUpdateWithDataGen
from models.Nets import MobileNetV2, vgg16, vgg19, MobileNetV3
from utils.Fed import FedAvg
from models.Test import test_img
from utils.util import setup_seed, exp_details
from data_utils.dataset_reader import TinyImageNetDataset
from datetime import datetime
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    setup_seed(args.seed)
    exp_details(args)

    data_dist = []
    x_client = []
    apa = {}
    if args.dataset == 'imagenet':
        TINY_IMAGENET_ROOT = 'data/tiny-imagenet-200/'
        if os.path.exists('tiny-imagenet-200.zip') == False:
            print('\nDownload dataset\n')
            url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
            tiny_imgdataset = wget.download(url, out = os.getcwd())
            with ZipFile('tiny-imagenet-200.zip', 'r') as zip_ref:
                zip_ref.extractall('data/')
        else:
            print('\nDataset is already downloaded\n')


        dataset_train = datasets.ImageFolder(
            os.path.join('data/', 'tiny-imagenet-200', 'train'),
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            )
        )
        dataset_test = TinyImageNetDataset(
            img_path=os.path.join('data/', 'tiny-imagenet-200', 'val', 'images'), 
            gt_path=os.path.join('data/', 'tiny-imagenet-200', 'val', 'val_annotations.txt'),
            class_to_idx=dataset_train.class_to_idx.copy(),
            transform=transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            )
        )
        if args.data_method:
            if args.data_dist == 'iid1':
                print('start separate dataset for iid')
                dict_users = imagenet_iid(dataset_train, args.num_users)
                x_client = [f'client{i}' for i in dict_users.keys()]
                data_dist = [len(dict_users[i]) for i in dict_users.keys()]
                print('end')
            elif args.data_dist == 'iid2':
                print('start separate dataset for iid')
                dict_users = iid_cluster_based(dataset_train, args.num_users)
                x_client = [f'client{i}' for i in dict_users.keys()]
                data_dist = [len(dict_users[i]) for i in dict_users.keys()]
                print('end')
            elif args.data_dist == 'noniid1':
                print('start separate dataset for non-iid using dirichlet distibution')
                dict_users, _ = imagenet_noniid(dataset_train, args.num_users, args.alpha)
                for k, v in dict_users.items():
                    data_dist.append(len(np.array(dataset_train.targets)[v]))
                    x_client.append(f'client{k}')
                print('end')
            elif args.data_dist =='noniid2':
                print('start separate dataset for non-iid using cluster-based partition')
                dict_users, _ = noniid_cluster_based(dataset_train, args.num_users, args.alpha)
                for k, v in dict_users.items():
                    data_dist.append(len(np.array(dataset_train.targets)[v]))
                    x_client.append(f'client{k}')
                print('end')
                 
    
    elif args.dataset == 'cifar':
        dataset_train = datasets.CIFAR10(
            'data/cifar', 
            train=True, 
            download=True, 
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(), 
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]
            )
        )
        dataset_test = datasets.CIFAR10(
            'data/cifar', 
            train=False, 
            download=True, 
            transform=transforms.Compose(
                [
                    transforms.ToTensor(), 
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]
            )
        )
        if args.data_method:
            if args.data_dist == 'iid1':
                print('start separate dataset for iid')
                dict_users = cifar_iid(dataset_train, args.num_users)
                x_client = [f'client{i}' for i in dict_users.keys()]
                data_dist = [len(dict_users[i]) for i in dict_users.keys()]
                print('end')
            elif args.data_dist == 'iid2':
                print('start separate dataset for iid')
                dict_users, apa = iid_cluster_based(dataset_train, args.num_users)
                x_client = [f'client{i}' for i in dict_users.keys()]
                data_dist = [len(dict_users[i]) for i in dict_users.keys()]
                print('end')
            elif args.data_dist == 'noniid1':
                print('start separate dataset for non-iid using dirichlet distibution')
                dict_users, _ = cifar_noniid(dataset_train, args.num_users, args.alpha)
                for k, v in dict_users.items():
                    data_dist.append(len(np.array(dataset_train.targets)[v]))
                    x_client.append(f'client{k}')
                print('end')
            elif args.data_dist =='noniid2':
                print('start separate dataset for non-iid using cluster-based partition')
                dict_users, _, apa = noniid_cluster_based(dataset_train, args.num_users, args.alpha)
                for k, v in dict_users.items():
                    data_dist.append(len(np.array(dataset_train.targets)[v]))
                    x_client.append(f'client{k}')
                print('end')
    

    # if args.iid and args.data_method:
    #     print('start separate dataset for iid')
    #     dict_users = imagenet_iid(dataset_train, args.num_users)
    #     x_client = [f'client{i}' for i in dict_users.keys()]
    #     data_dist = [len(dict_users[i]) for i in dict_users.keys()]
    #     print('end')
    # elif args.iid and args.data_method:
    #     print('start separate dataset for non-iid')
    #     dict_users, _ = imagenet_noniid(dataset_train, args.num_users, args.alpha)
    #     for k, v in dict_users.items():
    #         data_dist.append(len(np.array(dataset_train.targets)[v]))
    #         x_client.append(f'client{k}')
    #     print('end')
    
    
    plt.title("data distribution")
    plt.bar(x_client, data_dist, color ='maroon', width = 0.3)
    plt.savefig(f"imgs/data_dist_num_users{args.num_users}_iid_{args.data_dist}_epochs_{args.epochs}.png") 
    plt.close()  
        
        
    if args.dataset == 'imagenet':
        if args.model == 'mobilenetV3-small-pre':
            net_glob = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1).to(args.device)
        elif args.model == 'mobilenetV3-large-pre':
            net_glob = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2).to(args.device)
        elif args.model == 'mobilenetV3-small':
            net_glob = MobileNetV3(mode='small', classes_num=args.num_classes, input_size=56, 
                    width_multiplier=1, dropout=0.2, 
                    BN_momentum=0.1, zero_gamma=False)
    elif args.dataset == 'cifar':
        if args.model == 'vgg16':
            net_glob = vgg16().to(args.device)
        elif args.model == 'vgg19':
            net_glob = vgg19().to(args.device)


    # print(net_glob)
    net_glob.train()


    w_glob = net_glob.state_dict()


    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    test_best_acc = 0.0
    
    test_loss_ar = []
    test_loss_peer_batch = []
    tmp = 0
    test_acc_graph = []
    train_local_loss = {}
    
    num_items_l = int(len(dataset_train)/args.num_users)
    d_u, all_idxs_l = {}, [i for i in range(len(dataset_train))]

    for iter in tqdm(range(1, args.epochs + 1)):
        print(f'\nGlobal epoch {iter}\n')
        w_locals, loss_locals = [], []
        
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            print(f'\nclient {idx}\n')
            if args.data_method == True:
                local = LocalUpdateWithLocalsData(args=args, dataset=dataset_train, idxs=dict_users[idx])
            else:
                if idx not in d_u.keys():
                    print(f'Generated data for {idx} user')
                    d_u[idx] = set(np.random.choice(all_idxs_l, num_items_l, replace=False))
                    all_idxs_l = list(set(all_idxs_l) - d_u[idx])
                    local = LocalUpdateWithDataGen(args=args, dataset=dataset_train, idx_cur_user=d_u[idx])
                else:
                    print(f'load data for {idx} user')
                    local = LocalUpdateWithDataGen(args=args, dataset=dataset_train, idx_cur_user=d_u[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(w)
            loss_locals.append(loss)
            if idx in train_local_loss.keys():
                train_local_loss[idx].append(loss)
            else:
                train_local_loss[idx] = []
                train_local_loss[idx].append(loss)

        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        # with open(f'save/model_lust{iter}.pkl', 'wb') as fin:
        #     pickle.dump(net_glob, fin)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('\n==============================\n')
        print('Round {:3d}, Train loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        test_acc, test_loss, tmp = test_img(net_glob, dataset_test, args)
        test_loss_peer_batch.append(tmp)
        test_loss_ar.append(test_loss)
        test_acc_graph.append(test_acc)
        print('\n==============================\n')
        
        
    
    plt.title("global model")
    plt.plot(range(len(loss_train)), loss_train, label='train loss')
    plt.plot(range(len(loss_train)), test_loss_ar, label='test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(fontsize=12)
    plt.savefig("imgs/global model dataset_{}_model_{}_numUser_{}_epoch_{}_C_{}_iid_{}_optimizer_{}.png".format(args.dataset, args.model, args.num_users, args.epochs, args.frac, args.data_dist, args.optimizer))
    plt.close()
    
    
    k = 0
    # print(train_local_loss)
    for i in train_local_loss.keys():
        plt.plot(range(len(train_local_loss[i])), train_local_loss[i], label=f'client{i}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig("imgs/Train loss by each client. dataset_{}_model_{}_numUser_{}_epoch_{}_C_{}_iid_{}_optimizer_{}.png".format(args.dataset, args.model, args.num_users, args.epochs, args.frac, args.data_dist, args.optimizer))
    plt.close()
    
    
    create_boxplot(args, train_local_loss)
    create_hist_curves(args, train_local_loss)
    
    
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig("imgs/fed_dataset_{}_model_{}_numUser_{}_epoch_{}_C_{}_iid_{}_optimizer_{}.png".format(args.dataset, args.model, args.num_users, args.epochs, args.frac, args.data_dist, args.optimizer))
    plt.close()
    
    plt.title('Test accuracy')
    plt.plot(range(len(test_acc_graph)),test_acc_graph)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig("imgs/Test accuracy. dataset_{}_model_{}_numUser_{}_epoch_{}_C_{}_iid_{}_optimizer_{}.png".format(args.dataset, args.model, args.num_users, args.epochs, args.frac, args.data_dist, args.optimizer))
    plt.close()
    

    # testing
    a, b = 0, 0
    net_glob.eval()
    acc_train, loss_train, a = test_img(net_glob, dataset_train, args)
    acc_test, loss_test, b = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))


    with open('save/model_lust.pkl', 'wb') as fin:
        pickle.dump(net_glob, fin)
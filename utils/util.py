import copy
import torch
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed+1)
    torch.cuda.manual_seed_all(seed+123)
    np.random.seed(seed+1234)
    random.seed(seed+12345)
    torch.backends.cudnn.deterministic = True

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def exp_details(args):
    print('\n---------------Experimental details:-------------\n')
    print(f'\tModel           : {args.model}')
    print(f'\ttLearning       : {args.lr}')
    print(f'\tGlobal Rounds   : {args.epochs}\n')

    print('\n----------------Federated parameters:------------\n')
    if args.iid:
        print('\tIID')
    else:
        print('\tNon-IID')
    print(f'\tFraction of users  : {args.frac}')
    print(f'\tLocal Batch size   : {args.local_bs}')
    print(f'\tLocal Epochs       : {args.local_ep}\n')
    return

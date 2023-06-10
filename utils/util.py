import torch.backends.cudnn
import torch.cuda
import numpy as np
import random
import torch.nn as nn


def setup_seed(seed):
    torch.manual_seed(seed+1)
    torch.cuda.manual_seed_all(seed+123)
    np.random.seed(seed+1234)
    random.seed(seed+12345)
    torch.backends.cudnn.deterministic = True



def exp_details(args):
    print('***************************************************************************')
    print('***                   Experimental details:                             ***')
    print('***************************************************************************')
    print(f'***\t\tdataset\t\t\t:\t{args.dataset}\t\t\t***')
    print(f'***\t\tepochs\t\t\t:\t{args.epochs}\t\t\t***')
    print(f'***\t\tModel\t\t\t:\t{args.model}\t\t\t***')
    print(f'***\t\tLearning rate\t\t:\t{args.lr}\t\t\t***')
    print(f'***\t\tGlobal Rounds\t\t:\t{args.epochs}\t\t\t***')
    print(f'***\t\tOptimizer\t\t:\t{args.optimizer}\t\t\t***')
    print(f"***\t\tDevice\t\t\t:\t{'gpu' if torch.cuda.is_available() and args.gpu != -1 else 'cpu'}\t\t\t***")
    print(f'***\t\tData split\t\t:\t{args.data_dist}\t\t\t***')
    print(f'***\t\tFraction of users\t:\t{args.frac}\t\t\t***')
    print(f'***\t\tLocal Batch size\t:\t{args.local_bs}\t\t\t***')
    print(f'***\t\tLocal Epochs\t\t:\t{args.local_ep}\t\t\t***')
    print('***************************************************************************')
    return
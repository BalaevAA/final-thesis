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


def add_scalar(writer, user_num, test_result, epoch):
    test_loss, test_acc, user_loss, user_acc = test_result
    writer.add_scalar(f'user_{user_num}/global/test_loss', test_loss, epoch)
    writer.add_scalar(f'user_{user_num}/global/test_acc', test_acc, epoch)
    writer.add_scalar(f'user_{user_num}/local/test_loss', user_loss, epoch)
    writer.add_scalar(f'user_{user_num}/local/test_acc', user_acc, epoch)


def exp_details(args):
    print('*******************************************************************')
    print('***                   Experimental details:                     ***')
    print('*******************************************************************')
    print(f'***\t\tepochs\t\t\t:\t{args.epochs}\t\t***')
    print(f'***\t\tModel\t\t\t:\t{args.model}\t***')
    print(f'***\t\tLearning rate\t\t:\t{args.lr}\t\t***')
    print(f'***\t\tGlobal Rounds\t\t:\t{args.epochs}\t\t***')
    print(f"***\t\tDevice\t\t\t:\t{'gpu' if torch.cuda.is_available() and args.gpu != -1 else 'cpu'}\t\t***")
    if args.iid:
        print('***\t\tData split\t\t:\tIID\t\t***')
    else:
        print('***\t\tData split\t\t:\tNon-IID\t\t***')
    print(f'***\t\tFraction of users\t:\t{args.frac}\t\t***')
    print(f'***\t\tLocal Batch size\t:\t{args.local_bs}\t\t***')
    print(f'***\t\tLocal Epochs\t\t:\t{args.local_ep}\t\t***')
    print('*******************************************************************')
    return
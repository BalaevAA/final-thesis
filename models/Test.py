import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


def test(args, net_g, data_loader):
    # testing
    net_g.eval()
    test_loss = []
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            test_loss.append(nn.CrossEntropyLoss()(log_probs, target).item())
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum().item()

        loss_avg = sum(test_loss)/len(test_loss)
        test_acc = 100. * correct / len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        loss_avg, correct, len(data_loader.dataset), test_acc))

    return test_acc, loss_avg


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss_batch = []
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.test_bs)
    l = len(data_loader)
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        test_loss_batch.append(test_loss)
        accuracy = 100.00 * correct.item() / len(data_loader.dataset)
    # if args.verbose:
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss, test_loss_batch




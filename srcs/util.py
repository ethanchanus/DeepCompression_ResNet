import os
import torch
import math
import numpy as np
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torchvision import datasets, transforms
from datetime import date

log_filename = ""

def log(content, filename=""):
    global log_filename 
    if log_filename == "":
        log_filename = filename
    print(content)
    with open(log_filename, 'a') as f:
        content += "\n"
        f.write(content)


def print_model_parameters(model, with_values=False):
    log(f"{'Param name':20} {'Shape':30} {'Type':15}")
    log('-'*70)
    for name, param in model.named_parameters():
        log(f'{name:35} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            log(param)


def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        log(f'{name:35} | nonzeros = {nz_count:10} / {total_params:10} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    log( f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')

def loadMnistDataset(batch_size = 1000, test_batch_size = 1000, subdataset_size=-1, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.\data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.\data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def loadCifar10Dataset(batch_size = 1000, test_batch_size = 1000, subdataset_size=-1, **kwargs):
    
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    trainset = datasets.CIFAR10(
            root='.\data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(
            root='.\data', train=False, download=True, transform=transform_test)
    
    if subdataset_size != -1:
        trainset = torch.utils.data.Subset(trainset, range(0, subdataset_size))
        testset = torch.utils.data.Subset(testset, range(0, subdataset_size))

    train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    test_loader = torch.utils.data.DataLoader(
            testset, batch_size=test_batch_size, shuffle=False, num_workers=8)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader


def test(model, dataset="cifar-10", use_cuda=True):
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else 'cpu')

    test_loader = None
    if dataset == "mnist":
        _ , test_loader = loadMnistDataset(batch_size=1000)
    elif dataset == "cifar-10":
        _ , test_loader = loadCifar10Dataset(batch_size=1000)  


    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        log(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


# if __name__ == '__main__':
#     global log_filename = ""

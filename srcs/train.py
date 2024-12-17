import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from net.models import *

import util
from datetime import datetime


def train(epochs, log_interval = 500, model_checkpoint_name = "",  resume_train_batch=-1):
    start_epoch=0

    if resume_train_batch != -1:
        start_epoch=resume_train_batch 
    
    model.train()
    for epoch in range(start_epoch, epochs):
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        # print(len(train_loader))
        # exit()
        
        for (batch_idx, (data, target)) in pbar:
        # for data, target in train_loader:
            # print (batches.shape())
            # (data, target) = batches                
            # print (data.size(), target.size())
            
            data, target = data.to(device), target.to(device)
            # print(f'data.size = {data.size()}')
            optimizer.zero_grad()
            output = model(data)
            # loss = F.nll_loss(output, target)
            loss = criterion(output, target)
            loss.backward()
            
            # zero-out all the gradients corresponding to the pruned connections
            for name, p in model.named_parameters():
                # print (f'name {name}')
                # if 'label' in name:
                #     print (p.size())
                if 'mask' in name: # or 'label' in name:
                    continue
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor==0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
            
            optimizer.step()
            if batch_idx % log_interval == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')            
                # print(f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')            
            # exit()
        if model_checkpoint_name != "":
            torch.save(model, f"{model_checkpoint_name}.{epoch+1}")

        scheduler.step()


def test():
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
        util.log(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

def parsingArgs():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 50)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log', type=str, default='log.txt',
                        help='log file name')
    parser.add_argument('--sensitivity', type=float, default=2,
                        help="sensitivity value that is multiplied to layer's std in order to get threshold value")

    parser.add_argument('--dataset', type=str, default="cifar-10",
                        help="cifar-10 or mnist")
    parser.add_argument('--model', type=str, default="resnet-18",
                        help="lenet-300-100, lenet-5, resnet-18, resnet-34, resnet-50, resnet-101, resnet-152")
    parser.add_argument('--subdataset', type=int, default=-1,
                        help="Sub data set size - for quick run through the training process")

    parser.add_argument('--resumetrain', type=str, default="",
                        help="Resume training from")
    parser.add_argument('--resumebatch', type=int, default=-1,
                        help="Resume training from")

    parser.add_argument('--ignoretrain', type=bool, default=False,
                        help="Ignore initial train")
    parser.add_argument('--ignoreretrain', type=bool, default=False,
                        help="Ignore retrain")
    parser.add_argument('--ignoreprune', type=bool, default=False,
                        help="Ignore prune")                        

    parser.add_argument('--epochretrain', type=int, default=0,
                        help="Epoch for retraining")                        
    parser.add_argument('--resumeretrainopech', type=int, default=0,
                        help="start the retrainepoch")                        

    args = parser.parse_args()

    return args

if __name__ == '__main__':    

    args = parsingArgs()

    util.log(f'Starting {__file__} at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}', filename = args.log)
    util.log(f'Arguments: {args}')
    # exit()


    save_model_path = f'results/saves_{args.model}_{args.dataset}'
    os.makedirs(save_model_path, exist_ok=True)

    # Control Seed
    torch.manual_seed(args.seed)

    # Select Device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')
    if use_cuda:
        util.log("Using CUDA!")
        torch.cuda.manual_seed(args.seed)
    else:
        util.log('Not using CUDA!!!')

    # Loader
    util.log( "Loading dataset")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    if args.dataset == "mnist":
        train_loader , test_loader = util.loadMnistDataset(batch_size = args.batch_size, test_batch_size=args.test_batch_size, subdataset_size=args.subdataset )
    elif args.dataset == "cifar-10":
        train_loader , test_loader = util.loadCifar10Dataset(batch_size = args.batch_size, test_batch_size=args.test_batch_size, subdataset_size=args.subdataset )

    
    # Define which model to use    
    if args.resumetrain == "":
        model = newModel(args.model).to(device)
    else:
        model = torch.load(args.resumetrain)

    util.log("Model parameters:")
    # for module in model.children():
    #     print(module)
    #     print (module.weight.shape)

    util.print_model_parameters(model)

    # NOTE : `weight_decay` term denotes L2 regularization loss term

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    initial_optimizer_state_dict = optimizer.state_dict()

    if args.ignoretrain:
        util.log("Ignore intitial training")
    else:
        # Initial training
        util.log("\n\n\n--- Initial training ---")
        save_model_name = f"{save_model_path}/initial_model.ptmodel"

        train(args.epochs, log_interval=args.log_interval, model_checkpoint_name = save_model_name, resume_train_batch=args.resumebatch)
        util.log(f"Save the model {save_model_name}")
        torch.save(model, save_model_name)

    accuracy = test()
    util.log( f"Initial Training Accuracy {accuracy}")
    


    util.log("\n\n\n--------------------- Before pruning ---------------------")
    util.print_nonzeros(model)

    # model = torch.load( f"saves/{model.__class__.__name__}_initial_model.ptmodel")
    # accuracy = test()

    # Pruning

    if args.ignoreprune == False:
        model.prune_by_std(args.sensitivity)
    
    
    
        util.log("\n\n\n--------------------- After pruning ---------------------")
        util.print_nonzeros(model)
        save_model_name = f"{save_model_path}/model_after_pruning.ptmodel"
        util.log(f"Save the model {save_model_name}")
        torch.save(model, save_model_name)
        accuracy = test()
        util.log( f"Test accuracy AFTER pruning {accuracy}")
    


    save_model_name = f"{save_model_path}/model_after_retraining.ptmodel"
    if args.ignoreretrain:
        util.log("Ignore retraining")
    else:
        # Retrain
        util.log("\n\n\n--------------------- Retraining ---------------------")
        

        optimizer.load_state_dict(initial_optimizer_state_dict) # Reset the optimizer
        train(args.epochretrain, log_interval=args.log_interval, model_checkpoint_name = save_model_name, resume_train_batch=args.resumeretrainopech)
    
    util.log(f"Save the model {save_model_name}")
    torch.save(model, save_model_name)
        

    util.log("\n\n\n--------------------- After Retraining ---------------------")    
    util.print_nonzeros(model)
    accuracy = test()
    util.log( f"Test accuracy AFTER retraining {accuracy}")

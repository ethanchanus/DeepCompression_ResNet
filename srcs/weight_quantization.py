import argparse
import os

import torch
from torchsummary import summary

from net.models import LeNet
from deepcompress.quantization import apply_weight_sharing

import util
from datetime import datetime

def parseArgs():
    parser = argparse.ArgumentParser(description='This program quantizes weight by using weight sharing')
    parser.add_argument('model', type=str, help='path to saved pruned model')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # parser.add_argument('--output', default='saves/ResNet_model_after_weight_sharing.ptmodel', type=str,
    #                     help='path to model output')
    parser.add_argument('--dataset', type=str, default="cifar-10",
                        help="cifar-10 or mnist")
    parser.add_argument('--log', type=str, default='log.txt',
                        help='log file name')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parseArgs()
    util.log(f'Starting {__file__} at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}', filename = args.log)
    util.log(f'Arguments: {args}')

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    save_model_path = os.path.dirname(args.model)
    #f'saves_{os.path.basename(args.model)}_{args.dataset}'


    # Define the model
    model = torch.load(args.model)

    # summary(model, (1, 28, 28))
    # print(model)

    # for module in model.children():
    #     shape = module.weight.shape
    #     print(module, shape)


    print('Accuracy before weight sharing')
    util.test(model=model, dataset=args.dataset, use_cuda=use_cuda)

    # Weight sharing
    apply_weight_sharing(model)
    print('Accuacy after weight sharing')
    util.test(model=model, dataset=args.dataset, use_cuda=use_cuda)

    # Save the new model
    save_model_name = f"{save_model_path}/model_after_weight_sharing.ptmodel"
    # args.output = f'saves/{model.__class__.__name__}_model_after_weight_sharing.ptmodel'
    print (f'Save model to {save_model_name}')
    torch.save(model, save_model_name )

import argparse

import torch

from deepcompress.huffmancoding import huffman_encode_model,huffman_decode_model
import util
from net.models import *
import numpy as np
from datetime import datetime
import os


def model_check(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])#, key_item_1[1], key_item_2[0])
                # return False
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
        return True


def model_check1(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            print(p1, p2)
            return False
    return True


def parseArgs():
    parser = argparse.ArgumentParser(description='Huffman encode a quantized model')
    parser.add_argument('modelfile', type=str, help='path to saved pruned model')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # parser.add_argument('--output', default='saves/ResNet_model_after_weight_sharing.ptmodel', type=str,
    #                     help='path to model output')
    # parser.add_argument('--dataset', type=str, default="cifar-10",
    #                     help="cifar-10 or mnist")
    parser.add_argument('--log', type=str, default='log.txt',
                        help='log file name')

    parser.add_argument('--dataset', type=str, default="cifar-10",
                        help="cifar-10 or mnist")
    parser.add_argument('--model', type=str, default="resnet-18",
                        help="lenet-300-100, lenet-5, resnet-18, resnet-34, resnet-50, resnet-101, resnet-152")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = parseArgs()
    util.log(f'Starting {__file__} at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}', filename = args.log)
    util.log(f'Arguments: {args}')


    save_model_path = f'results/encoding_{os.path.basename(os.path.dirname(args.modelfile))}/'
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')

    model = torch.load(args.modelfile)

    util.log(f"Save the encoded model to {save_model_path}")
    huffman_encode_model(model, directory=save_model_path)
    
    util.test(model, use_cuda=use_cuda, dataset=args.dataset)
    # model_load= ResNet18().to(device)
    model_load = newModel(args.model).to(device)
    
    huffman_decode_model(model_load, directory=save_model_path)

    util.test(model_load, use_cuda=use_cuda, dataset=args.dataset)
    
    # model_check(model, model_load)
    # # print(f'model vs. model_load {model_check(model, model_load)}')

    # for name, param in model.named_parameters():
    #     if 'mask' in name:
    #         continue
    #     # print("model:", name)
    #     weight = param.data.cpu().numpy()

    #     weight_unload = None
    #     for name_, param_ in model_load.named_parameters():
    #         if name_ == name:
    #             weight_unload = param_
    #             break
    #     weight_unload = weight_unload.data.cpu().numpy()

    #     eqq = np.sum(weight_unload==weight)

    #     # print(weight.shape, weight_unload.shape, f'match: {eqq}/{weight_unload.size}')

    #     if eqq != weight.size:
    #         print("..........................NOT MATCH......")
        

    # util.test(model_load, use_cuda)
import argparse

import torch

from net.huffmancoding import huffman_encode_model, huffman_decode_model
import util
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
from net.quantization import apply_weight_sharing
import util

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Huffman encode a quantized model')
    # parser.add_argument('model', type=str, help='saved quantized model')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')

    # model = LeNet_5(mask=True).to(device)
    # model = LeNet_5(mask=True).to(device)
    model = ResNet18().to(device)

    # torch.load(args.model)
    huffman_decode_model(model)
    print(model)
    util.test(model, True)
'''
Author: fanlong
Date: 2022-04-20 14:43:30
LastEditors: fanlong
LastEditTime: 2022-06-27 09:20:49
FilePath: /workspace/code/mmGAN/test.py
Description: 

github: https://github.com/fanl0228
Email: fanl@smail.nju.edu.cn
Copyright (c) 2022 by fanlong/Nanjing University, All Rights Reserved. 
'''
#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import numpy as np

from models import Generator
from datasets import ComplexNumpyDataset

import pdb

def main(args):

    if torch.cuda.is_available() and args.gpu is not None:
        print("INFO CUDA is available")

    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator(args.input_nc, args.output_nc)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        netG_A2B.cuda(args.gpu)

    # Load state dicts
    netG_A2B.load_state_dict(torch.load(args.generator_A2B))

    # Set model's test mode
    netG_A2B.eval()

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if args.gpu is not None else torch.Tensor

    # Dataset loader
    transforms_ = []
    dataloader = DataLoader(ComplexNumpyDataset(args.dataroot, transforms_=transforms_, mode='test'), 
                            batch_size=args.batchSize, shuffle=False, num_workers=args.n_cpu)


    for i, batch in enumerate(dataloader):
        # print("----> number: {}".format(i))
        mmVocal = batch['mmVocal'].to(args.gpu)
        filename = batch['filename']

        # Generate output
        fake_B = 0.5*(netG_A2B(mmVocal).data + 1.0)
      
        # Save image files
        np.save(os.path.join(args.dataroot,'test/mmGAN_Out/{}'.format(filename[0])), fake_B.cpu())

        print('number: {} ---> Generated images {}\n'.format(i, filename[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='/root/workspace/dataset/mmGAN_Dataset_0623', help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=8, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=2, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
    parser.add_argument('--gpu', type=int, default=0, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str, default='weights_v2/netG_A2B_last.pth', help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='weights_v2/netG_B2A_last.pth', help='B2A generator checkpoint file')
    args = parser.parse_args()
    print(args)

    main(args)
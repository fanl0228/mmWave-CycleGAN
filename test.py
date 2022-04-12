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

from models import Generator
from datasets import ImageDataset

import pdb

def main(opt):

    if torch.cuda.is_available() and opt.gpu is not None:
        print("INFO CUDA is available")
    
    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    # netG_B2A = Generator(opt.output_nc, opt.input_nc)

    if opt.gpu is not None:
        torch.cuda.set_device(opt.gpu)
        netG_A2B.cuda(opt.gpu)
        # netG_B2A.cuda(opt.gpu)

    # Load state dicts
    netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
    # netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

    # Set model's test mode
    netG_A2B.eval()
    # netG_B2A.eval()

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.gpu is not None else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    # input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

    # Dataset loader
    transforms_ = [ transforms.Resize([opt.size, opt.size], interpolation=InterpolationMode.BICUBIC), # Image.BICUBIC
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)) ]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='train'), 
                            batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
    ###################################

    ###### Testing######

    # Create output dirs if they don't exist
    # if not os.path.exists('output/A'):
    #     os.makedirs('output/A')
    # if not os.path.exists('output/B'):
    #     os.makedirs('output/B')

    for i, batch in enumerate(dataloader):
        # import pdb
        # pdb.set_trace()
        
        # Set model input
        real_A = Variable(input_A.copy_(batch['mmwave']))
        filename = batch['filename']
        # real_B = Variable(input_B.copy_(batch['B']))

        # Generate output
        fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
        # fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

        
        # Save image files
        # save_image(fake_A, 'output/A/%04d.png' % (i+1))
        save_image(fake_B, 'output/{}'.format(filename[0]))

        sys.stdout.write('\rGenerated images {}\n'.format(filename[0]))

    sys.stdout.write('\n')
    ###################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='/root/workspace/dataset/mmGAN_Dataset_Digital/', help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=128, help='size of the data (squared assumed)')
    parser.add_argument('--gpu', type=int, default=0, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str, default='weights_cycle/netG_A2B_last.pth', help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='weights/netG_B2A_last.pth', help='B2A generator checkpoint file')
    opt = parser.parse_args()
    print(opt)

    main(opt)
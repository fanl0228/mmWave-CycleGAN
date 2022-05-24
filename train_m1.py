#!/usr/bin/python3

import argparse
from asyncio import FastChildWatcher
import itertools

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset

import pdb

def main(args):

    if torch.cuda.is_available() :
        print("INFO: CUDA is available")

    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator(args.input_nc, args.output_nc)
    netG_B2A = Generator(args.output_nc, args.input_nc)
    netD_A = Discriminator(args.input_nc)
    netD_B = Discriminator(args.output_nc)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        netG_A2B.cuda(args.gpu)
        netG_B2A.cuda(args.gpu)
        netD_A.cuda(args.gpu)
        netD_B.cuda(args.gpu)

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # Load pre training model.
    if args.netD_A != "":
        netD_A.load_state_dict(torch.load(args.netD_A))
    if args.netD_B != "":
        netD_B.load_state_dict(torch.load(args.netD_B))
    if args.netG_A2B != "":
        netG_A2B.load_state_dict(torch.load(args.netG_A2B))
    if args.netG_B2A != "":
        netG_B2A.load_state_dict(torch.load(args.netG_B2A))

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion_similiraty = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                    lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if args.gpu is not None else torch.Tensor
    input_A = Tensor(args.batchSize, args.input_nc, args.size, args.size)
    input_B = Tensor(args.batchSize, args.output_nc, args.size, args.size)
    target_real = Variable(Tensor(args.batchSize, 1).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(args.batchSize, 1).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()
    
    # Dataset loader
    transforms_ = [ transforms.Resize([args.size, args.size],interpolation=InterpolationMode.BICUBIC), # Image.BICUBIC
                    # transforms.RandomCrop(args.size), 
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)) ]
    dataloader = DataLoader(ImageDataset(args.dataroot, transforms_=transforms_, mode='train'), 
                            batch_size=args.batchSize, shuffle=False, num_workers=args.n_cpu, drop_last=True)

    # Loss plot
    logger = Logger(args.n_epochs, len(dataloader))
    ###################################

    ###### Training ######
    for epoch in range(args.epoch, args.n_epochs):
        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(input_A.copy_(batch['mmwave']))
            real_B = Variable(input_B.copy_(batch['audio']))

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)*5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)*5.0

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
            loss_A2B_similarity = criterion_similiraty(fake_B, real_B)*10.0

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
            loss_B2A_similarity = criterion_similiraty(fake_A, real_A)*10.0 
            
            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_A2B_similarity + loss_B2A_similarity
            loss_G.backward()
            
            optimizer_G.step()
            ###################################

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            optimizer_D_B.step()
            ###################################

            # Progress report (http://localhost:8097)
            logger.log({'loss_G_M1': loss_G, 
                        'loss_G_identity_M1': (loss_identity_A + loss_identity_B), 
                        'loss_G_GAN_M1': (loss_GAN_A2B + loss_GAN_B2A),
                        'loss_G_cycle_M1': (loss_cycle_ABA + loss_cycle_BAB), 
                        'loss_G_similarity_M1': (loss_A2B_similarity + loss_B2A_similarity),
                        'loss_D_M1': (loss_D_A + loss_D_B)}, 
                        images={'real_A_M1': real_A, 'real_B_M1': real_B, 'fake_A_M1': fake_A, 'fake_B_M1': fake_B})
            
        # if epoch % 100 == 0:
        #     # Save models checkpoints
        #     torch.save(netG_A2B.state_dict(), 'weights/netG_A2B_epoch{}.pth'.format(epoch))
        #     torch.save(netG_B2A.state_dict(), 'weights/epoch/netG_B2A_epoch{}.pth'.format(epoch))
        #     torch.save(netD_A.state_dict(), 'weights/epoch/netD_A_epoch{}.pth'.format(epoch))
        #     torch.save(netD_B.state_dict(), 'weights/netD_B_epoch{}.pth'.format(epoch))

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), 'weights/netG_A2B_M1_last.pth')
        torch.save(netG_B2A.state_dict(), 'weights/netG_B2A_M1_last.pth')
        torch.save(netD_A.state_dict(), 'weights/netD_A_M1_last.pth')
        torch.save(netD_B.state_dict(), 'weights/netD_B_M1_last.pth')
    ###################################

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',       type=int,  default=0,      help='starting epoch')
    parser.add_argument('--n_epochs',    type=int,  default=1000,    help='number of epochs of training')
    parser.add_argument('--batchSize',   type=int,  default=16,     help='size of the batches')
    parser.add_argument('--dataroot',    type=str,  default='/root/workspace/dataset/mmGAN_Dataset_Digital/', help='root directory of the dataset')
    parser.add_argument("--netD_A",      type=str,  default="",     help="Path to Discriminator checkpoint.")
    parser.add_argument("--netD_B",      type=str,  default="",     help="Path to Discriminator checkpoint.")
    parser.add_argument("--netG_A2B",    type=str,  default="",     help="Path to Generator checkpoint.")
    parser.add_argument("--netG_B2A",    type=str,  default="",     help="Path to Generator checkpoint.")
    parser.add_argument('--lr',          type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int,  default=500,    help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size',        type=int,  default=128,    help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc',    type=int,  default=1,      help='number of channels of input data')
    parser.add_argument('--output_nc',   type=int,  default=1,      help='number of channels of output data')
    parser.add_argument("--gpu",         type=int,  default=0,      help="GPU id to use.")
    parser.add_argument('--n_cpu',       type=int,  default=16,      help='number of cpu threads to use during batch generation')
    args = parser.parse_args()
    print(args)

    main(args)

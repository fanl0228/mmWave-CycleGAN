#!/usr/bin/python3

import argparse
import logging
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

from models import Generator
from datasets import get_batch_spectrum

import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def main(args):
    if torch.cuda.is_available() and args.gpu is not None:
        logger.info(f"cuda is available and gpu: `{args.gpu}`")

    # model
    netG_A2B = Generator(args.input_nc, args.output_nc)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        netG_A2B.cuda(args.gpu)
    
    # load state dicts
    netG_A2B.load_state_dict(torch.load(args.generator_A2B, map_location=torch.device('cpu')))

    # eval
    netG_A2B.eval()

    if not os.path.exists("predict_out"):
        os.makedirs("predict_out")
    
    transforms_ = [ transforms.Resize(int(args.size), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)) ]
    imageA = get_batch_spectrum(image_path=args.image_path, transforms_=transforms_)
    imageA = torch.unsqueeze(imageA, dim=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imageA = imageA.to(device)

    # Generate output
    imageB = 0.5*(netG_A2B(imageA).data + 1.0)

    outpath = args.image_path[:-4] + "_out.png"
    save_image(imageB, outpath)
    logger.info("save path: " + outpath)

    if 1:
        plt.figure(1)
        plt.subplot(121)
        plt.imshow(imageA[0][0], cmap='jet', aspect='auto')
        plt.title("imageA")

        plt.subplot(122)
        plt.imshow(imageB[0][0], cmap='jet', aspect='auto')
        plt.title("imageB")

        plt.pause(0.05)

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='./images/spectrum_23_s2_p7_up.png', help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=128, help='size of the data (squared assumed)')
    parser.add_argument('--gpu', type=int, default=None, help='use GPU computation')
    parser.add_argument('--generator_A2B', type=str, default='weights/netG_A2B_last.pth', help='A2B generator checkpoint file')
    args = parser.parse_args()
    print(args)

    main(args)
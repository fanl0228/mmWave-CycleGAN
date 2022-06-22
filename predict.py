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


def main(image_file,
         gt_image_file,
         out_path,
         generator_A2B,
         input_nc=1,
         output_nc=1,
         size=128,
         gpu=None):

    if torch.cuda.is_available() and gpu is not None:
        logger.info(f"cuda is available and gpu: `{gpu}`")

    # model
    netG_A2B = Generator(input_nc, output_nc)
    # netG_B2A = Generator(args.input_nc, args.output_nc)

    if gpu is not None:
        torch.cuda.set_device(gpu)
        netG_A2B.cuda(gpu)
        # netG_B2A.cuda(args.gpu)
    
    # load state dicts
    netG_A2B.load_state_dict(torch.load(generator_A2B, map_location=torch.device('cpu')))
    # netG_B2A.load_state_dict(torch.load(args.generator_B2A, map_location=torch.device('cpu')))

    # eval
    netG_A2B.eval()
    # netG_B2A.eval()

    if not os.path.exists("predict_out"):
        os.makedirs("predict_out")
    
    transforms_ = [ transforms.Resize(int(size), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)) ]
    imageA = get_batch_spectrum(image_path=image_file, transforms_=transforms_)
    imageA = torch.unsqueeze(imageA, dim=0)

    imageA_GT = get_batch_spectrum(image_path=gt_image_file, transforms_=transforms_)
    imageA_GT = torch.unsqueeze(imageA_GT, dim=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imageA = imageA.to(device)

    # Generate output
    imageB = 0.5 * (netG_A2B(imageA).data + 1.0)
    # imageA_Re = 0.5 * (netG_B2A(imageB).data + 1.0)

    img_name = os.path.basename(image_file)
    savepath = os.path.join(out_path, img_name[:-4] + "_out.png")
    # save_image(imageB, outpath)
    # logger.info("save path: " + outpath)

    if 1:
        plt.figure(1)
        plt.subplot(131)
        plt.imshow(imageA[0][0], cmap='jet', aspect='auto')
        plt.title("imageA")

        plt.subplot(132)
        plt.imshow(imageB[0][0], cmap='jet', aspect='auto')
        plt.title("imageB")

        plt.subplot(133)
        plt.imshow(imageA_GT[0][0], cmap='jet', aspect='auto')
        plt.title("imageA_GT")

        plt.savefig(savepath)
        plt.pause(0.05)

    print("done")


if __name__ == "__main__":
    image_dir = "testData/images/"
    gt_image_dir = "testData/gt_images"
    for i, file in enumerate(os.listdir(image_dir)):
        main(image_file=os.path.join(image_dir, file),
             gt_image_file=os.path.join(gt_image_dir, file),
             out_path="./output/",
             generator_A2B='weights/netG_A2B_last.pth',
             input_nc=1,
             output_nc=1,
             size=128,
             gpu=None)
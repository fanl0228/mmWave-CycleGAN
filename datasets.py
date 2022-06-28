import glob
import random
import os

import numpy
import torch
from regex import F

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import re
import numpy as np
import pdb


def get_batch_spectrum(image_path, transforms_=None):
    transform = transforms.Compose(transforms_)
    img = Image.open(image_path)
    img = transform(img)
    return img

class ImageDatasetBase(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)

        self.files_mmwave = sorted(glob.glob(os.path.join(root, '%s/mmwave' % mode) + '/*.*'))
        self.files_audio = sorted(glob.glob(os.path.join(root, '%s/audio' % mode) + '/*.*'))
        

    def __getitem__(self, index):
        file_name = self.files_mmwave[index % len(self.files_mmwave)].split('/')[-1]
        # print("file_name: {}\n".format(file_name))
        
        path_mmwave = '/'.join(self.files_mmwave[index % len(self.files_mmwave)].split('/')[:-1])
        path_audio = '/'.join(self.files_audio[index % len(self.files_audio)].split('/')[:-1])

        file_mmwave_name = os.path.join(path_mmwave, file_name)
        file_audio_name = os.path.join(path_audio, file_name)
        # print("mmwave: {}, audio:  {}\n".format(file_mmwave_name, file_audio_name))

        item_mmwave = self.transform(Image.open(file_mmwave_name))
        item_audio = self.transform(Image.open(file_audio_name))

        return {'mmwave': item_mmwave, 'audio': item_audio, 'filename': file_name}

    def __len__(self):
        return max(len(self.files_mmwave), len(self.files_audio))


class ImageDatasetV2(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)

        self.files_audio = sorted(glob.glob(os.path.join(root, '%s/Audio' % mode) + '/*.png'))
        self.files_mmVocal_Rx0 = sorted(glob.glob(os.path.join(root, '%s/mmVocal_RX0' % mode) + '/*.png'))
        self.files_mmVocal_Rx1 = sorted(glob.glob(os.path.join(root, '%s/mmVocal_RX1' % mode) + '/*.png'))
        self.files_mmVocal_Rx2 = sorted(glob.glob(os.path.join(root, '%s/mmVocal_RX2' % mode) + '/*.png'))
        self.files_mmVocal_Rx3 = sorted(glob.glob(os.path.join(root, '%s/mmVocal_RX3' % mode) + '/*.png'))
        self.files_mmlip = sorted(glob.glob(os.path.join(root, '%s/mmLip' % mode) + '/*.npy'))
        
    def __getitem__(self, index):
        file_name = os.path.basename(self.files_audio[index % len(self.files_audio)])
        # print("file_name: {}\n".format(file_name))

        path_audio = '/'.join(self.files_audio[index % len(self.files_audio)].split('/')[:-1])
        file_audio_name = os.path.join(path_audio, file_name)

        path_mmVocal_Rx0 = '/'.join(self.files_mmVocal_Rx0[index % len(self.files_mmVocal_Rx0)].split('/')[:-1])
        file_mmVocal_Rx0 = os.path.join(path_mmVocal_Rx0, file_name)
        path_mmVocal_Rx1 = '/'.join(self.files_mmVocal_Rx1[index % len(self.files_mmVocal_Rx1)].split('/')[:-1])
        file_mmVocal_Rx1 = os.path.join(path_mmVocal_Rx1, file_name)
        path_mmVocal_Rx2 = '/'.join(self.files_mmVocal_Rx2[index % len(self.files_mmVocal_Rx2)].split('/')[:-1])
        file_mmVocal_Rx2 = os.path.join(path_mmVocal_Rx2, file_name)
        path_mmVocal_Rx3 = '/'.join(self.files_mmVocal_Rx3[index % len(self.files_mmVocal_Rx3)].split('/')[:-1])
        file_mmVocal_Rx3 = os.path.join(path_mmVocal_Rx3, file_name)
        

        path_mmlip = '/'.join(self.files_mmlip[index % len(self.files_mmlip)].split('/')[:-1])
        file_mmLip = os.path.join(path_mmlip, file_name)

        # print("file_mmVocal_Rx0:{}\nfile_mmVocal_Rx0:{}\nfile_mmVocal_Rx0:{}\nfile_mmVocal_Rx0:{}\naudio:{}\n"
        #       .format(file_mmVocal_Rx0, file_mmVocal_Rx1, file_mmVocal_Rx2, file_mmVocal_Rx3, file_audio_name))

        item_audio = self.transform(Image.open(file_audio_name))
        item_Rx0 = self.transform(Image.open(file_mmVocal_Rx0))
        item_Rx1 = self.transform(Image.open(file_mmVocal_Rx1))
        item_Rx2 = self.transform(Image.open(file_mmVocal_Rx2))
        item_Rx3 = self.transform(Image.open(file_mmVocal_Rx3))
        
        # _mmLip = np.load(file_mmLip)
        # item_mmLip = np.array([_mmLip.real, _mmLip.imag])
        item_mmLip = [] #torch.Tensor(item_mmLip)

        labelStr = file_name.split('_')[1]  # 类别位于文件名下划线第一个位置
        classLabel = torch.Tensor([int(re.sub('\D', '', labelStr))])
        return {'mmVocal': item_Rx0, 'audio': item_audio,
                'mmlip': item_mmLip, 'classLabel': classLabel}

    def __len__(self):
        return max(len(self.files_mmVocal_Rx0), len(self.files_audio))


class Rx0_ComplexNumpyDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        
        self.files_audio = sorted(glob.glob(os.path.join(root, '%s/Audio' % mode) + '/*.npy'))  # reference filename
        self.files_mmVocal_Rx0 = sorted(glob.glob(os.path.join(root, '%s/mmVocal_RX0' % mode) + '/*.npy'))

        self.files_mmlip = sorted(glob.glob(os.path.join(root, '%s/mmLip' % mode) + '/*.npy'))

    def __getitem__(self, index):
        file_name = os.path.basename(self.files_audio[index % len(self.files_audio)])

        path_audio = '/'.join(self.files_audio[index % len(self.files_audio)].split('/')[:-1])
        file_audio_name = os.path.join(path_audio, file_name)

        path_mmVocal_Rx0 = '/'.join(self.files_mmVocal_Rx0[index % len(self.files_mmVocal_Rx0)].split('/')[:-1])
        file_mmVocal_Rx0 = os.path.join(path_mmVocal_Rx0, file_name)

        _audio = np.load(file_audio_name)
        item_audio = np.array([_audio.real, _audio.imag])

        _Rx0 = np.load(file_mmVocal_Rx0)

        # item_mmVocal = np.array([_Rx0.real, _Rx0.imag])
        item_mmVocal = np.array([_Rx0.real, _Rx0.imag])

        item_audio_resize = numpy.append(item_audio, np.zeros((2, 256, 256-item_audio.shape[-1])), axis=-1)
        item_mmVocal_resize = numpy.append(item_mmVocal, np.zeros((2, 256, 256 - item_mmVocal.shape[-1])), axis=-1)

        # 标准化到[-1, 1]
        item_audio_resize = item_audio_resize + abs(item_audio_resize.min())
        item_audio_resize = item_audio_resize / item_audio_resize.max()
        item_audio_resize = (item_audio_resize - 0.5) / 0.5
        item_mmVocal_resize = item_mmVocal_resize + abs(item_mmVocal_resize.min())
        item_mmVocal_resize = item_mmVocal_resize / item_mmVocal_resize.max()
        item_mmVocal_resize = (item_mmVocal_resize - 0.5) / 0.5

        item_audio_torch = torch.Tensor(item_audio_resize)
        item_mmVocal_torch = torch.Tensor(item_mmVocal_resize)

        labelStr = file_name.split('_')[1]  # 类别位于文件名下划线第一个位置
        label_index = int(re.sub('\D', '', labelStr))
        class_label = label_index

        return {'mmVocal': item_mmVocal_torch, 'audio': item_audio_torch,
                'classLabel': class_label, 
                'filename': file_name}

    def __len__(self):
        return max(len(self.files_mmVocal_Rx0), len(self.files_audio))


class ComplexNumpyDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        
        self.files_audio = sorted(glob.glob(os.path.join(root, '%s/Audio' % mode) + '/*.npy'))  # reference filename
        self.files_mmVocal_Rx0 = sorted(glob.glob(os.path.join(root, '%s/mmVocal_RX0' % mode) + '/*.npy'))
        self.files_mmVocal_Rx1 = sorted(glob.glob(os.path.join(root, '%s/mmVocal_RX1' % mode) + '/*.npy'))
        self.files_mmVocal_Rx2 = sorted(glob.glob(os.path.join(root, '%s/mmVocal_RX2' % mode) + '/*.npy'))
        self.files_mmVocal_Rx3 = sorted(glob.glob(os.path.join(root, '%s/mmVocal_RX3' % mode) + '/*.npy'))

    def __getitem__(self, index):
        file_name = os.path.basename(self.files_audio[index % len(self.files_audio)])
        # print("file_name: {}\n".format(file_name))

        path_audio = '/'.join(self.files_audio[index % len(self.files_audio)].split('/')[:-1])
        file_audio_name = os.path.join(path_audio, file_name)

        path_mmVocal_Rx0 = '/'.join(self.files_mmVocal_Rx0[index % len(self.files_mmVocal_Rx0)].split('/')[:-1])
        file_mmVocal_Rx0 = os.path.join(path_mmVocal_Rx0, file_name)
        path_mmVocal_Rx1 = '/'.join(self.files_mmVocal_Rx1[index % len(self.files_mmVocal_Rx1)].split('/')[:-1])
        file_mmVocal_Rx1 = os.path.join(path_mmVocal_Rx1, file_name)
        path_mmVocal_Rx2 = '/'.join(self.files_mmVocal_Rx2[index % len(self.files_mmVocal_Rx2)].split('/')[:-1])
        file_mmVocal_Rx2 = os.path.join(path_mmVocal_Rx2, file_name)
        path_mmVocal_Rx3 = '/'.join(self.files_mmVocal_Rx3[index % len(self.files_mmVocal_Rx3)].split('/')[:-1])
        file_mmVocal_Rx3 = os.path.join(path_mmVocal_Rx3, file_name)

        _audio = np.load(file_audio_name)
        item_audio = np.array([_audio.real, _audio.imag])

        _Rx0 = np.load(file_mmVocal_Rx0)
        _Rx1 = np.load(file_mmVocal_Rx1)
        _Rx2 = np.load(file_mmVocal_Rx2)
        _Rx3 = np.load(file_mmVocal_Rx3)
        # item_mmVocal = np.array([_Rx0.real, _Rx0.imag])
        item_mmVocal = np.array([_Rx0.real, _Rx0.imag, _Rx1.real, _Rx1.imag,
                                 _Rx2.real, _Rx2.imag, _Rx3.real, _Rx3.imag])

        item_audio_resize = numpy.append(item_audio, np.zeros((2, 256, 256-item_audio.shape[-1])), axis=-1)
        item_mmVocal_resize = numpy.append(item_mmVocal, np.zeros((8, 256, 256 - item_mmVocal.shape[-1])), axis=-1)
        
        item_audio_resize = item_audio_resize + abs(item_audio_resize.min())
        item_audio_resize = item_audio_resize / item_audio_resize.max()
        item_audio_resize = (item_audio_resize - 0.5) / 0.5

        item_mmVocal_resize = item_mmVocal_resize + abs(item_mmVocal_resize.min())
        item_mmVocal_resize = item_mmVocal_resize / item_mmVocal_resize.max()
        item_mmVocal_resize = (item_mmVocal_resize - 0.5) / 0.5

        item_audio_torch = torch.Tensor(item_audio_resize)
        item_mmVocal_torch = torch.Tensor(item_mmVocal_resize)

        labelStr = file_name.split('_')[1]  # 类别位于文件名下划线第一个位置
        label_index = int(re.sub('\D', '', labelStr))
        class_label = label_index

        return {'mmVocal': item_mmVocal_torch, 'audio': item_audio_torch,
                'classLabel': class_label, 
                'filename': file_name}

    def __len__(self):
        return max(len(self.files_mmVocal_Rx0), len(self.files_audio))


if __name__ == "__main__":
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image

    transforms_ = []
    dataloader = DataLoader(ComplexNumpyDataset(r"/root/workspace/dataset/mmGAN_Dataset_0623",
                            transforms_=transforms_, mode='test'), 
                            batch_size=1, shuffle=False, num_workers=1, drop_last=True)

    # transforms_ = [
    #                 transforms.Resize([256, 64]),
    #                 transforms.ToTensor(),
    #                 # transforms.Normalize((0.5,), (0.5,))
    #                 ]
    # dataloader = DataLoader(ImageDatasetV2(r"/root/workspace/dataset/mmGAN_Dataset_0623",
    #                         transforms_=transforms_, mode='train'), 
    #                         batch_size=1, shuffle=False, num_workers=1, drop_last=True)

    for i, batch in enumerate(dataloader):
        print("batch number: {}".format(i))
        print("mmvoal shape: {}".format(batch['mmVocal'].shape))
        print("audio shape:{}".format(batch["audio"].shape))
        # print("mmlip shape:{}".format(batch["mmlip"].shape))
        # print(batch["mmlip"])
        print("class label: {}".format(batch["classLabel"]))
        print(batch["audio"])


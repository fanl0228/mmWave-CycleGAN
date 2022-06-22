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

class ImageDataset(Dataset):
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


class ComplexNumpyDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)

        self.files_audio = sorted(glob.glob(os.path.join(root, r'%s/Audio' % mode) + r'/*.npy'))  # reference filename
        self.files_mmVocal_Rx0 = sorted(glob.glob(os.path.join(root, r'%s/mmVocal_Rx0' % mode) + r'/*.npy'))
        self.files_mmVocal_Rx1 = sorted(glob.glob(os.path.join(root, r'%s/mmVocal_Rx1' % mode) + r'/*.npy'))
        self.files_mmVocal_Rx2 = sorted(glob.glob(os.path.join(root, r'%s/mmVocal_Rx2' % mode) + r'/*.npy'))
        self.files_mmVocal_Rx3 = sorted(glob.glob(os.path.join(root, r'%s/mmVocal_Rx3' % mode) + r'/*.npy'))

        self.files_mmlip = sorted(glob.glob(os.path.join(root, r'%s/mmLip' % mode) + r'/*.npy'))

    def __getitem__(self, index):
        file_name = os.path.basename(self.files_audio[index % len(self.files_audio)])
        # print("file_name: {}\n".format(file_name))

        path_audio = r'/'.join(self.files_audio[index % len(self.files_audio)].split('\\')[:-1])
        file_audio_name = os.path.join(path_audio, file_name)

        path_mmVocal_Rx0 = r'/'.join(self.files_mmVocal_Rx0[index % len(self.files_mmVocal_Rx0)].split('\\')[:-1])
        file_mmVocal_Rx0 = os.path.join(path_mmVocal_Rx0, file_name)
        path_mmVocal_Rx1 = r'/'.join(self.files_mmVocal_Rx1[index % len(self.files_mmVocal_Rx1)].split('\\')[:-1])
        file_mmVocal_Rx1 = os.path.join(path_mmVocal_Rx1, file_name)
        path_mmVocal_Rx2 = r'/'.join(self.files_mmVocal_Rx2[index % len(self.files_mmVocal_Rx2)].split('\\')[:-1])
        file_mmVocal_Rx2 = os.path.join(path_mmVocal_Rx2, file_name)
        path_mmVocal_Rx3 = r'/'.join(self.files_mmVocal_Rx3[index % len(self.files_mmVocal_Rx3)].split('\\')[:-1])
        file_mmVocal_Rx3 = os.path.join(path_mmVocal_Rx3, file_name)

        path_mmlip = r'/'.join(self.files_mmlip[index % len(self.files_mmlip)].split('\\')[:-1])
        file_mmLip = os.path.join(path_mmlip, file_name)

        # print("file_mmVocal_Rx0:{}\nfile_mmVocal_Rx0:{}\nfile_mmVocal_Rx0:{}\nfile_mmVocal_Rx0:{}\naudio:{}\n"
        #       .format(file_mmVocal_Rx0, file_mmVocal_Rx1, file_mmVocal_Rx2, file_mmVocal_Rx3, file_audio_name))

        _audio = np.load(file_audio_name)
        item_audio = np.array([_audio.real, _audio.imag])

        _Rx0 = np.load(file_mmVocal_Rx0)
        _Rx1 = np.load(file_mmVocal_Rx1)
        _Rx2 = np.load(file_mmVocal_Rx2)
        _Rx3 = np.load(file_mmVocal_Rx3)
        # item_mmVocal = np.array([_Rx0.real, _Rx0.imag])
        item_mmVocal = np.array([_Rx0.real, _Rx0.imag, _Rx1.real, _Rx1.imag,
                                 _Rx2.real, _Rx2.imag, _Rx3.real, _Rx3.imag])

        _mmLip = np.load(file_mmLip)
        item_mmLip = np.array([_mmLip.real, _mmLip.imag])

        item_audio_resize = numpy.append(item_audio, np.zeros((2, 256, 256-item_audio.shape[-1])), axis=-1)
        item_mmVocal_resize = numpy.append(item_mmVocal, np.zeros((8, 256, 256 - item_mmVocal.shape[-1])), axis=-1)
        item_mmLip_resize = numpy.append(item_mmLip, np.zeros((2, 16, 64 - item_mmLip.shape[-1])), axis=-1)

        item_audio_torch = torch.Tensor(item_audio_resize)
        item_mmVocal_torch = torch.Tensor(item_mmVocal_resize)
        item_mmLip_torch = torch.Tensor(item_mmLip_resize)

        labelStr = file_name.split('_')[1]  # 类别位于文件名下划线第一个位置
        classLabel = torch.Tensor([int(re.sub('\D', '', labelStr))])
        return {'mmVocal': item_mmVocal_torch, 'audio': item_audio_torch,
                'mmlip': item_mmLip_torch, 'classLabel': classLabel}

    def __len__(self):
        return max(len(self.files_mmVocal_Rx0), len(self.files_audio))


if __name__ == "__main__":
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image

    transforms_ = [transforms.Resize([256, 40]),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.5,), (0.5,))
                    ]
    dataloader = DataLoader(ComplexNumpyDataset(r"D:\03_Dataset\PhoneticSymbol_Dataset_0612\data_Segmentation_Output",
                            transforms_=transforms_, mode='train'), 
                            batch_size=16, shuffle=True, num_workers=1, drop_last=True)

    for i, batch in enumerate(dataloader):
        print("batch number: {}".format(i))
        print("mmvoal shape: {}".format(batch['mmVocal'].shape))
        print("audio shape:{}".format(batch["audio"].shape))
        print("mmlip shape:{}".format(batch["mmlip"].shape))
        # print(batch["mmlip"])
        print("class label: {}".format(batch["classLabel"]))
        # print(batch["audio"])



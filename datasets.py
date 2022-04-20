import glob
import random
import os
from regex import F

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

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



if __name__ == "__main__":
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image

    transforms_ = [ transforms.Resize([128,128], interpolation=InterpolationMode.BICUBIC), 
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)) ]
    dataloader = DataLoader(ImageDataset("/root/workspace/dataset/mmGAN_Dataset_Digital/", 
                            transforms_=transforms_, mode='train'), 
                            batch_size=16, shuffle=False, num_workers=1, drop_last=True)

    for i, batch in enumerate(dataloader):
        print(i)
        save_image(batch['mmwave'], "./batchA.png")
        save_image(batch['audio'], "./batchB.png")



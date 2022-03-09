# Pytorch-CycleGAN
A clean and readable Pytorch implementation of CycleGAN (https://arxiv.org/abs/1703.10593)

## Prerequisites
Code is intended to work with ```Python 3.7.x```, it hasn't been tested with previous versions

### [PyTorch & torchvision](http://pytorch.org/)
Follow the instructions in [pytorch.org](http://pytorch.org) for your current setup

### [Visdom](https://github.com/facebookresearch/visdom)
To plot loss graphs and draw images in a nice web browser view
```
pip3 install visdom
```

## Training
### 1. Setup the dataset
First, you will need to download and setup a dataset. The easiest way is to use one of the already existing datasets on UC Berkeley's repository:
```
./download_dataset <dataset_name>
```
Alternatively you can build your own dataset by setting up the following directory structure:

    .
    ├── datasets                   
    |   ├── <dataset_name>
    |   |   ├── train              # Training
    |   |   |   ├── A              # Contains domain A images (i.e. mmSpectrum)
    |   |   |   └── B              # Contains domain B images (i.e. audio Spectrum)
    |   |   └── test               # Testing
    |   |   |   ├── A              # Contains domain A images
    |   |   |   └── B              # Contains domain B images
    
### 2. Train!
```
python train.py --dataroot datasets/<dataset_name>/ --gpu 0
```
This command will start a training session using the images under the *dataroot/train* directory with the hyperparameters that showed best results according to CycleGAN authors. You are free to change those hyperparameters, see ```./train --help``` for a description of those.

Both generators and discriminators weights will be saved under the output directory.

If you don't own a GPU remove the --gpu option, although I advise you to get one!

You can also view the training progress as well as live output images by running ```python3 -m visdom``` in another terminal and opening [http://localhost:8097/](http://localhost:8097/) in your favourite web browser. This should generate training loss progress as shown below (default params, horse2zebra dataset):

### 3. Train with Pretrain

```
python train.py --netD_A "weights/netD_A_last.pth" --netD_B "weights/netD_B_last.pth" --netG_A2B "weights/netG_A2B_last.pth" --netG_B2A "weights/netG_B2A_last.pth"
```

## Testing
```
python test.py --dataroot datasets/<dataset_name>/
```
This command will take the images under the *dataroot/test* directory, run them through the generators and save the output under the *output/A* and *output/B* directories. As with train, some parameters like the weights to load, can be tweaked, see ```./test --help``` for more information.


## License
This project is licensed under the GPL v3 License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
### refcode: 
1. [PyTorch-CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN)
2. [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

### paper:
1. [CycleGAN](https://arxiv.org/abs/1703.10593)

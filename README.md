<!--
 * @Author: fanlong
 * @Date: 2022-03-08 14:11:52
 * @LastEditors: fanlong
 * @LastEditTime: 2022-06-23 10:53:43
 * @FilePath: /workspace/code/mmGAN/README.md
 * @Description: 
 * 
 * @github: https://github.com/fanl0228
 * @Email: fanl@smail.nju.edu.cn
 * Copyright (c) 2022 by fanlong/Nanjing University, All Rights Reserved. 
-->
# mmCycle-CGAN


## Prerequisites
Code is intended to work with ```Python 3.7.x```, it hasn't been tested with previous versions


### Visualize the training process
[Visdom](https://github.com/facebookresearch/visdom)
To plot loss graphs and draw images in a nice web browser view

```python
pip3 install visdom
```

```python
python -m visdom.server -p 2022
```

You can also view the training progress as well as live output images by running ```python3 -m visdom``` in another terminal and opening [http://localhost:8097/](http://localhost:8097/) in your favourite web browser.


If it is a remote server, a local ssh login is required to establish a connection
```
ssh -L 18097:127.0.0.1:8097 user@ip

ssh -L 2022:127.0.0.1:2022 root@8.136.3.19

```
> Open the link [http://localhost:18097/](http://localhost:18097/) in your local browser



## Training
### 1. Prepare the dataset

- directory structure:
>
    .  

    ├── <dataset_name>
    |   ├── train              # Training
    |   |   ├── mmwave              # Contains domain mmSpectrum images
    |   |   └── audio              # Contains domain audio Spectrum images
    |   └── test               # Testing
    |   |   ├── mmwave              # Contains domain mmSpectrum images
    |   |   └── audio              # Contains domain audio Spectrum images
    
### 2. Train
```
python train.py --dataroot ${datasets_path} 
```
We use gpu0 for training by default


### 3. Model save path

Both generators and discriminators weights will be saved under the ```weights``` directory.


### 4. Train with Pretrain

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

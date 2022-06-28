'''
Author: fanlong
Date: 2022-06-26 11:25:47
LastEditors: fanlong
LastEditTime: 2022-06-28 17:00:26
FilePath: /workspace/code/mmGAN/Npy2Wav.py
Description: 

github: https://github.com/fanl0228
Email: fanl@smail.nju.edu.cn
Copyright (c) 2022 by fanlong/Nanjing University, All Rights Reserved. 
'''

import os
import numpy as np
from scipy import signal
import soundfile as sf
import glob

import pdb

if __name__=="__main__":
    data_root = "/root/workspace/dataset/mmGAN_Dataset_0623"
    
    filenames = sorted(glob.glob(os.path.join(data_root, '%s/mmGAN_Out' % "test") + '/*.npy'))

    for i in range(len(filenames)):
        # pdb.set_trace()
        npy_data = np.load(filenames[i], allow_pickle=True)
        npy_complex_data = npy_data[0,0,:,:] + 1j*npy_data[0,1,:,:]
        wav_time , wav_data = signal.istft(npy_complex_data, fs=19385, window='hann', 
                                nperseg=1024, noverlap=512, nfft=1024)
        
        sf.write(filenames[i][:-4]+".wav", data=wav_data, samplerate=19385)
        print("processing...{}".format(i))
        


        



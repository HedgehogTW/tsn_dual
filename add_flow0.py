import pathlib
import shutil
# from multiprocessing import Pool, current_process
import argparse
import random
import math 
from tqdm.autonotebook import tqdm
import cv2 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

from time import time
import platform

from collections import defaultdict
import re
from re import sub             


np.set_printoptions(precision=4, suppress=True)
pd.set_option('display.float_format', '{:,.5f}'.format)
pd.set_option('display.max_colwidth', 80)

_platform = platform.platform()
print('platform:', _platform, platform.node())
if 'Linux' in _platform: # linux
    rat_path = '/home/ece/rat_data/'
    tsn_path = '/home/ece/tsn_data/'
    
elif 'macOS' in _platform: # MAC OS X
    rat_path = '/Users/cclee/rat_data/'
    tsn_path = '/Users/cclee/tsn_data/'     
elif 'Windows' in _platform: # Windows
    if platform.node()=='Mozart':
        rat_path = 'e:/rat_data/'
        tsn_path = 'e:/tsn_data/' 
    else:
        rat_path = 'd:/rat_data/'   
        tsn_path = 'd:/tsn_data/' 
  
path_rat = pathlib.Path(rat_path)
path_tsn = pathlib.Path(tsn_path)
path_new_grooming = path_rat.joinpath('new_grooming')

print(cv2.__version__)

rat_lst = [ 921111, 921216, 930203, 930217, 930302, 930309, 930316, 930323, 930330] #get_rat_lst()  921111,
for rat in rat_lst:
    print(rat)
    outpath = path_tsn.joinpath('{}'.format(rat))
    path_frames = outpath.joinpath('frames')
    clip_lst = [x for x in path_frames.iterdir() if x.is_dir() ]
    
    pbar = tqdm(total=len(clip_lst), ascii=True)
    for clip in clip_lst:
        flowx = clip.joinpath('flow_x_00002.jpg')
        shutil.copyfile(flowx, clip.joinpath('flow_x_00001.jpg'))

        flowy = clip.joinpath('flow_y_00002.jpg')
        shutil.copyfile(flowy, clip.joinpath('flow_y_00001.jpg'))

        pbar.update(1)

    pbar.close() 

    fname = outpath.joinpath('grooming_lst.txt')
    with open(fname) as f:
        clip_gm_lst = f.readlines()
    print('clip_gm_lst',len(clip_gm_lst))

#     clip_nongm_lst = sorted(path_rat_nongrooming.glob('9*.mp4'))
    fname = outpath.joinpath('nongrooming_lst.txt')
    with open(fname) as f:
        clip_nongm_lst = f.readlines()
    print('clip_nongm_lst',len(clip_nongm_lst))

    pbar = tqdm(total=len(clip_gm_lst), ascii=True)
    for clip in clip_gm_lst:
        cpath = pathlib.Path(clip)
        vid_name = cpath.stem.split('.')[0]
        clip_dir = path_frames.joinpath('Grooming_'+vid_name)
        
        # calc_optical_flow(clip, clip_dir )  
        cap = cv2.VideoCapture(str(clip))

        bOpenVideo = cap.isOpened()
        if bOpenVideo == False:
            print('Open Video failed')
        else:
            ret, frame1 = cap.read()
            save_img=clip_dir.joinpath('img_{:05d}.jpg'.format(1))
            cv2.imwrite(str(save_img), frame1)
        cap.release()  
        pbar.update(1)

    pbar.close() 

    pbar = tqdm(total=len(clip_nongm_lst), ascii=True)
    for clip in clip_nongm_lst:
        cpath = pathlib.Path(clip)
        vid_name = cpath.stem.split('.')[0]
        clip_dir = path_frames.joinpath('Nongrooming_'+vid_name)
        
        # calc_optical_flow(clip, clip_dir )  
        cap = cv2.VideoCapture(str(clip))

        bOpenVideo = cap.isOpened()
        if bOpenVideo == False:
            print('Open Video failed')
        else:
            ret, frame1 = cap.read()
            save_img=clip_dir.joinpath('img_{:05d}.jpg'.format(1))
            cv2.imwrite(str(save_img), frame1)
        cap.release()  
        pbar.update(1)

    pbar.close()     
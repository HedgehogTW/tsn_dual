import pathlib
import shutil
# from multiprocessing import Pool, current_process
import argparse
import random
import math 
from tqdm.autonotebook import tqdm

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
if platform.node() =='v100': # linux
    rat_path = '/home/ece/rat_data/'
    tsn_path = '/home/ece/tsn_data/'
elif platform.node() =='lab70808': # linux
    rat_path = '/home/lab70808/rat_data/'
    tsn_path = '/home/lab70808/tsn_data/'    
    
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

# print(cv2.__version__)


SPLIT = 1
SEED = 43

rat_lst = [921111, 930217, 930302, 930316, 921216, 930203]

df_test_lst = []


for rat in rat_lst:
    path_frames = path_tsn.joinpath(str(rat), 'frames')
    print(path_frames)
    if not path_frames.exists():
        print('cannot find path ', path_frames)
        break   
  

    grooming_train_lst = [str(x) for x in path_frames.iterdir() if x.is_dir() and x.name[0]=='G' ]
    non_grooming_train_lst = [str(x) for x in path_frames.iterdir() if x.is_dir() and x.name[0]=='N' ]
    
    print('grooming_train_lst', len(grooming_train_lst))
    print('non_grooming_train_lst', len(non_grooming_train_lst))
    x = grooming_train_lst.copy()
    
    x.extend(non_grooming_train_lst)
    print('total x', len(x))
    
    y = list(np.ones(len(grooming_train_lst), np.int8))
    y.extend(list(np.zeros(len(non_grooming_train_lst), np.int8)))
    
    print('total y',len(y), 'sum y',sum(y), sum(y)/len(y))
    
 
    df = pd.DataFrame({'x':x, 'y':y}) 
    # print(df)
 

    # df_test_lst.append(df)

    # end of for loop

    # df_test = pd.concat(df_test_lst)

    ###########################################
    #### output test file list
    df_test = df.reset_index(drop=True)
    frame_count_lst = []
    for row in df_test.itertuples():
        path_folder = pathlib.Path(row.x)
        img_lst = list(path_folder.glob('img*.jpg'))
        frame_count_lst.append(len(img_lst))

    ss = pd.Series(frame_count_lst)    
    df_test.insert(1, 'count', ss)
    print('df_test')
    print(df_test)

    path_tsn_data = path_tsn.joinpath('rat')
    if not path_tsn_data.exists():
        path_tsn_data.mkdir() 

    fname = path_tsn_data.joinpath('test_flow_{}.txt'.format(rat))
    df_test.to_csv(fname, header=False, index = False, sep = ' ')
    fname = path_tsn_data.joinpath('test_rgb_{}.txt'.format(rat))
    df_test.to_csv(fname, header=False, index = False, sep = ' ')


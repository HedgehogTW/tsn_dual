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


def get_rat_lst(allrat=True):
    rat_lst = []
    if allrat:
        all_rat_set = set()
        csv_lst = sorted(path_new_grooming.glob('9*.csv'))
        for csvf in csv_lst:
            tok = csvf.stem.split('_')
            rat_date = int(tok[0])
            all_rat_set.add(rat_date)
        
        rat_lst = list(all_rat_set)
        rat_lst.sort()
        
        print('all ', rat_lst)
        remove_lst = []
        for i in range(len(rat_lst)-1):
            if rat_lst[i]+1==rat_lst[i+1]:
                remove_lst.append(rat_lst[i+1])
          
        print('remove_lst ', remove_lst)
        for r in remove_lst:
            rat_lst.remove(r)    
                
#         for r in finished_lst:
#             rat_lst.remove(r)                        
        
    print('rat_lst ', rat_lst)
    return rat_lst


SPLIT = 1
SEED = 43
non_grm_factor = 2  # 1.2

rat_lst = [921111, 921216, 930203, 930217, 930302, 930309, 930316, 930323, 930330] #get_rat_lst() 
# rat_lst = [930217]
df_train_lst = []
df_test_lst = []


for rat in rat_lst:
    path_frames = path_tsn.joinpath(str(rat), 'frames')
    print(path_frames)
    if not path_frames.exists():
        print('cannot find path ', path_frames)
        break    

    grooming_train_lst = ['/'.join(x.parts[2:]) for x in path_frames.iterdir() if x.is_dir() and x.name[0]=='G' ]
    non_grooming_train_lst = ['/'.join(x.parts[2:]) for x in path_frames.iterdir() if x.is_dir() and x.name[0]=='N' ]
    
    print('grooming_train_lst', len(grooming_train_lst))
    print('non_grooming_train_lst', len(non_grooming_train_lst))
    x = grooming_train_lst.copy()
    
    x.extend(non_grooming_train_lst)
    print('total x', len(x))
    
    y = list(np.ones(len(grooming_train_lst), np.int8))
    y.extend(list(np.zeros(len(non_grooming_train_lst), np.int8)))
    
    print('total y',len(y), 'sum y',sum(y), sum(y)/len(y))
    
    skf = StratifiedShuffleSplit(n_splits=SPLIT, random_state=SEED, test_size=0.2)
    for train_index, test_index in skf.split(x, y):
        print("TRAIN:", len(train_index), "TEST:", len(test_index), 'SUM:', len(train_index)+len(test_index))

    df = pd.DataFrame({'x':x, 'y':y}) 
    print(df)
    df_train = df.iloc[train_index]
    print('df_train', df_train.shape)
    
    # nongroom 取1.2倍的grooming 數量
    num_train_groom = sum(df_train['y'])
    num_train_nongroom = int(num_train_groom * non_grm_factor)
    
    df_train_groom = df_train[df_train['y']==1]
    df_train_nongroom1 = df_train[df_train['y']==0]
    df_train_nongroom2 = df_train_nongroom1.sample(n=num_train_nongroom, random_state=SEED)
    df_train_nongroom3 = df_train_nongroom1.drop(df_train_nongroom2.index)
    
    df_train = df_train_groom.append(df_train_nongroom2)
    df_train = df_train.sample(frac=1, random_state=SEED)
    
    print('df_train: sum of 1', sum(df_train['y']), sum(df_train['y'])/len(df_train))
    print('df_train_nongroom1 (origin)', len(df_train_nongroom1))
    print('df_train_nongroom2', len(df_train_nongroom2))
    print('df_train_nongroom3 (rest)', len(df_train_nongroom3))
    print('df_train', len(df_train))
    
    df_test = df.iloc[test_index]
    print('df_test', df_test.shape)
    
    # nongroom 取1.2倍的grooming 數量
    num_test_groom = sum(df_test['y'])
    num_test_nongroom = int(num_test_groom * non_grm_factor)
    
    df_test_groom = df_test[df_test['y']==1]
    df_test_nongroom1 = df_test[df_test['y']==0]
    df_test_nongroom2 = df_test_nongroom1.sample(n=num_test_nongroom, random_state=SEED)
    df_test_nongroom3 = df_test_nongroom1.drop(df_test_nongroom2.index)
    
    df_test = df_test_groom.append(df_test_nongroom2)
    df_test = df_test.sample(frac=1, random_state=SEED)
    
    
    print('df_test: sum of 1', sum(df_test['y']), sum(df_test['y'])/len(df_test))
    print('df_test (after append)', len(df_test))
    print('df_test: sum of 1', sum(df_test['y']), sum(df_test['y'])/len(df_test))
 
    df_train_lst.append(df_train)
    df_test_lst.append(df_test)
    
df_train = pd.concat(df_train_lst) 
df_test = pd.concat(df_test_lst)

#### output train file list
df_train = df_train.reset_index(drop=True)
frame_count_lst = []
for row in df_train.itertuples():
    count = row.x.rsplit('_', 1)[-1]
    # path_folder = pathlib.Path(row.x)
    # img_lst = list(path_folder.glob('img*.jpg'))
    frame_count_lst.append(count)

ss = pd.Series(frame_count_lst)    
df_train.insert(1, 'count', ss)
print('df_train')
print(df_train)

path_tsn_data = path_tsn.joinpath('rat_two')
if not path_tsn_data.exists():
    path_tsn_data.mkdir() 
fname = path_tsn_data.joinpath('train_split1.txt')
df_train.to_csv(fname, header=False, index = False, sep = ' ')
# fname = path_tsn_data.joinpath('train_rgb_split1.txt')
# df_train.to_csv(fname, header=False, index = False, sep = ' ')

# shutil.copyfile(fname, pathlib.Path('.').joinpath('datasets', 'settings', 'rat', 'train_flow_split1.txt'))
# shutil.copyfile(fname, pathlib.Path('.').joinpath('datasets', 'settings', 'rat', 'train_rgb_split1.txt'))

###########################################
#### output test file list
df_test = df_test.reset_index(drop=True)
frame_count_lst = []
for row in df_test.itertuples():
    count = row.x.rsplit('_', 1)[-1]
    # path_folder = pathlib.Path(row.x)
    # img_lst = list(path_folder.glob('img*.jpg'))
    frame_count_lst.append(count)

ss = pd.Series(frame_count_lst)    
df_test.insert(1, 'count', ss)
print('df_test')
print(df_test)

fname = path_tsn_data.joinpath('val_split1.txt')
df_test.to_csv(fname, header=False, index = False, sep = ' ')
# fname = path_tsn_data.joinpath('val_rgb_split1.txt')
# df_test.to_csv(fname, header=False, index = False, sep = ' ')

# shutil.copyfile(fname, pathlib.Path('.').joinpath('datasets', 'settings', 'rat', 'val_flow_split1.txt'))
# shutil.copyfile(fname, pathlib.Path('.').joinpath('datasets', 'settings', 'rat', 'val_rgb_split1.txt'))

# print('copy tran and val list files to "datasets/settings/rat"')
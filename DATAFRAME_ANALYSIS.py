#!/usr/bin/env python
# coding: utf-8

# In[5]:


import getopt
import sys
import csv
import os
import analysis
import dill as pickle # see https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
import torch
import os
import simulation as sim
import math

from analysis import gData, MODE_MAP
from tqdm import tqdm
from numpy.random import choice
#from training_layer2 import Net
from torch.autograd import Variable
import pandas as pd
import analysis

import numpy as np
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt
from analysis import save_plt_figure


# In[6]:


analysis_object="C://Users/User/Desktop/개별연구 권하람/tst_1(02-24 기준)/history_results/2020-03-25/2020-03-25-23-21-36/Analysis-Object 25-23-29-38.pkl"
PARAMETER_FILE="C://Users/User/Desktop/Jaehoon_Shin/연구/Samsung/code_20190611/regdata.csv"
MODE_LIST = ['min-spe', 'max-spe', 'min-rpe', 'max-rpe', 'min-rpe-min-spe', 'max-rpe-max-spe', 'max-rpe-min-spe', 'min-rpe-max-spe', 'min-rpe-PMB', 'max-rpe-PMB']
MODE_MAP = {
    'min-spe' : ['spe', None, 'red'],
    'max-spe' : ['spe', None, 'mediumseagreen'],
    'min-rpe' : ['rpe', None, 'royalblue'],
    'max-rpe' : ['rpe', None, 'plum'],
    'min-rpe-min-spe' : ['spe', 'rpe', 'tomato'],
    'max-rpe-max-spe' : ['spe', 'rpe', 'dodgerblue'],
    'max-rpe-min-spe' : ['spe', 'rpe', 'y'],
    'min-rpe-max-spe' : ['spe', 'rpe', 'mediumvioletred'],
    'min-rpe-PMB' : ['rpe', None, 'royalblue'],
    'max-rpe-PMB' : ['rpe', None, 'plum'],
}


# In[7]:


with open(analysis_object, 'rb') as pkl_file:
        gData = pickle.load(pkl_file)

control = gData.detail
human =gData.data



'''
Below code dumps DataFrame data to excel
'''
#print(type(control['min-spe'][0]))
#print(control['min-spe'][0])
i = 0
for val in control.values():
    df = val[0]
    mode = MODE_LIST[i]
    filepath = 'C://Users/User/Desktop/개별연구 권하람/tst_1(02-24 기준)/detail_{}.xlsx'.format(mode)
    df.to_excel(filepath, index=False)
    i += 1
    #val[0].values --> in numpy_array type.
i = 0
for val in human.values():
    df = val[0]
    mode = MODE_LIST[i]
    filepath = 'C://Users/User/Desktop/개별연구 권하람/tst_1(02-24 기준)/data_{}.xlsx'.format(mode)
    df.to_excel(filepath, index=False)
    i += 1




# In[7]: pkl load 후 plot하기.


MODE_IDENTIFIER = 'NN Sequence Control'
stacked_ori_mean=[]
stacked_ori_sem=[]
stacked_new_mean=[]
stacked_new_sem=[]
stacked_ori_mean_2nd=[]
stacked_ori_sem_2nd=[]
stacked_new_mean_2nd=[]
stacked_new_sem_2nd=[]

for i in range():
    stacked_original_series=[]
    stacked_new_series=[]
    for j in range(82):
        data_df = gData.data[MODE_LIST[i]][j]
        detail_df = gData.detail[MODE_LIST[i]][j]
        target_val = MODE_MAP[MODE_LIST[i]][0]
        episode_index = data_df['ctrl_reward'].loc[0.2 * len(data_df):].idxmax() #.rolling(EPISODE_SMOOTH_WINDOW).mean().idxmax()
        #episode_index2 = gData.data[MODE_IDENTIFIER+'-'+MODE_LIST[i]][j]['ctrl_reward'].idxmax() #.rolling(EPISODE_SMOOTH_WINDOW).mean().idxmax()
        episode_index2 = 99
        original_series = gData.detail[MODE_LIST[i]][j][target_val].loc[episode_index * gData.trial_separation :
                                       (episode_index + 1) * gData.trial_separation - 1].copy().tolist()
        new_series = gData.detail[MODE_IDENTIFIER+'-'+MODE_LIST[i]][j][target_val].loc[episode_index2 * gData.trial_separation : (episode_index2 + 1) * gData.trial_separation - 1].copy().tolist()

        stacked_original_series=stacked_original_series+[original_series];
        stacked_new_series=stacked_new_series+[new_series];
    mean_ori=[]
    mean_new=[]
    sem_ori=[];
    sem_new=[];
    for k in range(len(original_series)):
        tmp_ori=[]
        tmp_new=[]
        for l in range(len(stacked_original_series)):
            tmp_ori=tmp_ori+[stacked_original_series[l][k]]
            tmp_new=tmp_new+[stacked_new_series[l][k]]
        mean_ori=mean_ori+[sum(tmp_ori)/len(tmp_ori)]
        mean_new=mean_new+[sum(tmp_new)/len(tmp_new)]
        tmp_ori=[]
        tmp_new=[]
        for l in range(len(stacked_original_series)):
            tmp_ori=tmp_ori+[(stacked_original_series[l][k]-mean_ori[k])*(stacked_original_series[l][k]-mean_ori[k])]
            tmp_new=tmp_new+[(stacked_new_series[l][k]-mean_new[k])*(stacked_new_series[l][k]-mean_new[k])]
        sem_ori=sem_ori+[math.sqrt(sum(tmp_ori)/len(tmp_ori)/(len(tmp_ori)-1))]
        sem_new=sem_new+[math.sqrt(sum(tmp_new)/len(tmp_new)/(len(tmp_new)-1))]     
    stacked_ori_mean=stacked_ori_mean+[mean_ori]
    stacked_new_mean=stacked_new_mean+[mean_new]
    stacked_ori_sem=stacked_ori_sem+[sem_ori]
    stacked_new_sem=stacked_new_sem+[sem_new]




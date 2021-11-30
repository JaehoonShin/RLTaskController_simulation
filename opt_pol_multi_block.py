import analysis
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
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
from statistics import stdev
from analysis import gData, MODE_MAP
from tqdm import tqdm
from numpy.random import choice
#from training_layer2 import Net
from torch.autograd import Variable
import analysis
from scipy import stats
from scipy.interpolate import pchip_interpolate
from analysis import save_plt_figure
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from itertools import permutations
import scipy.io

action_list = ['Nill', 'R-recover-visited', 'R-recover-unvisited']#['Nil','R-swap', 'R-decay','R-recover']#, 'Uncertainty']
tsne_go = False
NUM_ACT =len(action_list)
pol_list = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe','random']
NUM_POL = len(pol_list)
file_suffix = '_2020_2_TR1_no_PMB'
#COLUMNS = ['ctrl_reward']
#full_data = np.zeros((5,82,240,1000)) # Feature parameters x Policy block x Subject x Session x epi
opt_index= np.zeros(82)
opt_pol = np.zeros((120,82,80))
ctrl_rwd = np.zeros((120,82))
sess_type = list(permutations(['1','2','3','4','5'],4))
#for pol in range(len(pol_list)):
sc_cnt = np.zeros(NUM_POL)

sess_cnt = 0
for sess in sess_type:
    print(sess)
    for sbj in range(82):
        file_name = 'history_results/Analysis-Object-'+''.join(list(sess))+'-{0:02d}'.format(sbj)+file_suffix+'.pkl'
        with open(file_name,'rb') as f:
            data = pickle.load(f)
            #data.current_data: eps, data.current_detial : eps*trials
            blck_scs = list(map(int,list(sess)))
            opt_index[sbj] = data.data[''.join(list(sess))][0]['ctrl_reward'].loc[
                                  0.2 * len(data.data[''.join(list(sess))][0]):].idxmax()
            ctrl_rwd[sess_cnt][sbj] = data.data[''.join(list(sess))][0]['ctrl_reward'][opt_index[sbj]]
            opt_pol[sess_cnt][sbj] = data.detail[''.join(list(sess))][0]['action'].loc[
                                opt_index[sbj] * 80 - 80:opt_index[sbj] * 80 - 1]
    sess_cnt += 1
print(np.mean(ctrl_rwd, axis = 1))
print(np.mean(ctrl_rwd, axis = 1).argmax())
max_sbj = np.mean(ctrl_rwd, axis = 1).argmax()
np.save('history_results/multi_optimal_policy'+file_suffix+'.npy',opt_pol)
np.save('history_results/multi_optimal_score'+file_suffix+'.npy',np.mean(ctrl_rwd, axis = 1))

sess_cnt = 0
save_pol = np.zeros((120,80))
for sess in sess_type:
    save_pol[sess_cnt] = opt_pol[sess_cnt][max_sbj]
    sess_cnt += 1
scipy.io.savemat("opt_pol.mat",{'pol':save_pol})




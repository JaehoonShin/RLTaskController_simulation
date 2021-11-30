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
import scipy.io
from math import ceil

action_list = ['Nill', 'R-swa[', 'R-recover-visited', 'R-recover-unvisited']#['Nil','R-swap', 'R-decay','R-recover']#, 'Uncertainty']
tsne_go = False
NUM_ACT =len(action_list)
# pol_list = ['min-rpe','max-rpe','min-spe','max-spe','min-rpe-min-spe','max-rpe-max-spe','min-rpe-max-spe','max-rpe-min-spe']
pol_list = ['max-rpe']
NUM_POL = len(pol_list)
file_suffix =  '_20210329_Q_fix'#'_20210325_RPE' #'_20210317_natural2' #'_20210304' #_repro_mode202010'
#COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'd_p_mb', 'ctrl_reward', 'score']
COLUMNS = ['rpe','rpe1','rpe2','ctrl_reward','score','p_mb','0','10','20','40','visit','applied_reward']
max_indx = 82
TRIALS_PER_EPISODE = 40
NUM_EPISODES = 10000
full_data = np.zeros((len(COLUMNS),NUM_POL,max_indx,NUM_EPISODES))
full_detail = np.zeros((len(COLUMNS),NUM_POL,max_indx,NUM_EPISODES*TRIALS_PER_EPISODE))
full_opt = np.zeros((len(COLUMNS),NUM_POL,max_indx))
full_plot = np.zeros((len(COLUMNS),NUM_POL,2))
full_SBJ_plot = np.zeros((len(COLUMNS),max_indx,2))
opt_index= np.zeros((NUM_POL,max_indx))
opt_pol = np.zeros((NUM_POL,max_indx,TRIALS_PER_EPISODE,2))
RPE_plot_detail = np.zeros((NUM_POL,2))
SPE_plot_detail = np.zeros((NUM_POL,2))
PMB_plot_detail = np.zeros((NUM_POL,2))
RWD_plot_detail = np.zeros((NUM_POL,2))
save_pol = np.zeros((NUM_POL,TRIALS_PER_EPISODE,2))
time_delay = [1.7, 4.06, 6.99]
single_game_time = 8.4
TR = 2.8
full_detail_expanded = np.zeros((ceil(max_indx*NUM_POL*0.8*NUM_EPISODES),ceil(TRIALS_PER_EPISODE*single_game_time/TR)))
for pol in range(len(pol_list)):
    print(pol_list[pol])
    for sbj in range(max_indx):
        with open('history_results/Analysis-Object-'+pol_list[pol]+'-{0:02d}'.format(sbj)+file_suffix+'.pkl','rb') as f:
            data = pickle.load(f)
            #data.current_data: eps, data.current_detial : eps*trials
        for feat_indx in range(len(COLUMNS)):
            if feat_indx == 0: full_data[feat_indx][pol][sbj] = data.data[pol_list[pol]][0][COLUMNS[feat_indx]]
            full_detail[feat_indx][pol][sbj] = data.detail[pol_list[pol]][0][COLUMNS[feat_indx]]
            opt_index[pol][sbj]=data.data[pol_list[pol]][0]['ctrl_reward'].loc[ceil(0.2 * len(data.data[pol_list[pol]][0])):].idxmax()
            for t_indx in range(TRIALS_PER_EPISODE):
                opt_pol[pol][sbj][t_indx] = data.detail[pol_list[pol]][0]['action'].loc[
                                      opt_index[pol][sbj]*TRIALS_PER_EPISODE-TRIALS_PER_EPISODE+t_indx]
                # opt_pol[pol][sbj][t_indx][1] = data.detail[pol_list[pol]][0]['action'].loc[opt_index[pol][sbj] * 20 - 20 + t_indx][1]
            full_opt[feat_indx][pol][sbj] = sum(full_detail[feat_indx][pol][sbj]
                                                [int(TRIALS_PER_EPISODE*opt_index[pol][sbj]-TRIALS_PER_EPISODE):
                                                 int(TRIALS_PER_EPISODE*opt_index[pol][sbj]-1)]) / TRIALS_PER_EPISODE
            full_plot[feat_indx][pol][0] += np.mean(full_detail[feat_indx][pol][sbj][ceil(0.2*NUM_EPISODES*TRIALS_PER_EPISODE):])/max_indx
            full_SBJ_plot[feat_indx][sbj][0] += np.mean(full_detail[feat_indx][pol][sbj][ceil(0.2*NUM_EPISODES*TRIALS_PER_EPISODE):])/4
        for ep_indx in range(ceil(0.2*NUM_EPISODES+1),NUM_EPISODES):
            for trial_indx in range(TRIALS_PER_EPISODE):
                rpe1_tindx = ceil((trial_indx * single_game_time + time_delay[1]) / TR)
                rpe2_tindx = ceil((trial_indx * single_game_time + time_delay[2]) / TR)
                full_detail_expanded[ceil(pol * max_indx * 0.8*NUM_EPISODES + sbj * 0.8*NUM_EPISODES + ep_indx -
                                     0.2*NUM_EPISODES - 1)][rpe1_tindx] = \
                                    full_detail[1][pol][sbj][ep_indx * TRIALS_PER_EPISODE + trial_indx]
                full_detail_expanded[ceil(pol * max_indx * 0.8*NUM_EPISODES + sbj * 0.8*NUM_EPISODES + ep_indx -
                                     0.2*NUM_EPISODES - 1)][rpe2_tindx] = \
                                    full_detail[2][pol][sbj][ep_indx * TRIALS_PER_EPISODE + trial_indx]
#    RPE_plot[pol][0] = sum(opt_RPE[pol])/len(opt_RPE[pol])
 #   SPE_plot[pol][0] = sum(opt_SPE[pol]) / len(opt_SPE[pol])
    for feat_indx in range(len(COLUMNS)):
        full_plot[feat_indx][pol][1] = stdev(full_opt[feat_indx][pol])/np.sqrt(len(full_opt[feat_indx][pol]))
    save_pol[pol] = opt_pol[pol][max_indx]
'''
for sbj in range(max_indx):
    tmp = np.transpose(full_opt[feat_indx])
    full_SBJ_plot[feat_indx][sbj][1] = stdev(tmp[sbj])/np.sqrt(len(pol_list))
'''
np.save('history_results/optimal_policy'+file_suffix+'.npy',save_pol)
np.save('history_results/feat'+file_suffix+'.npy',full_detail)
#scipy.io.savemat('history_results/RPE_regressor'+file_suffix+'.mat',{'RPE': full_detail_expanded})
scipy.io.savemat('history_results/full_detail'+file_suffix+'.mat',{'detail': full_detail})
scipy.io.savemat('history_results/full_data'+file_suffix+'.mat',{'data': full_data})
'''
for feat_indx in range(len(COLUMNS)):
    temp_mean = np.zeros(max_indx)
    temp_std = np.zeros(max_indx)
    for ii in range(max_indx):
        temp_mean[ii]=full_SBJ_plot[feat_indx][ii][0]
        temp_std[ii] = full_SBJ_plot[feat_indx][ii][1]
    plt.bar(range(max_indx), temp_mean)
    plt.errorbar(range(max_indx), temp_mean, yerr=temp_std)
    plt.xticks(range(max_indx), range(82))
    plt.title(COLUMNS[feat_indx])
    plt.savefig('history_results/'+COLUMNS[feat_indx]+'_SBJ_plot' + file_suffix + '.png')
    plt.clf()
'''
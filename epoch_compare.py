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

#action_list = ['Nill','R-recover-visited', 'R-recover-unvisited']#['Nil','R-swap', 'R-decay','R-recover']#, 'Uncertainty']
action_list = ['Nil','0.5<->0.9','S<->F','R-recover-visited','R-recover-unvisit']#, 'Uncertainty']
tsne_go = False
NUM_ACT =len(action_list)
pol_list = ['min-rpe','max-rpe','min-spe','max-spe','min-rpe-min-spe','max-rpe-max-spe','min-rpe-max-spe','max-rpe-min-spe']
pol_list_tick = ['minR(LR)','maxR(HR)','minS(LS)','maxS(HS)','LRLS','HRHS','LRHS','HRLS']
NUM_POL = len(pol_list)
COLUMNS = ['rpe','spe','ctrl_reward','score','0','10','20','40','visit', 'applied_reward']
file_suffix = '_20210616_2019_delta_control'#'_20210331_Q_fix_delta_control'#'_20210329_Q_fix'#'_20210325_RPE' #'_20210304' #_repro_mode202010'
file_suffix = '_20210601_2021_ep1000_delta_control' #2021 task + 1000 eps
#COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'd_p_mb', 'ctrl_reward', 'score']
data_1000 = np.load('history_results/feat_full_opt' + file_suffix + '.npy')
file_suffix = '_20210601_2021_delta_control' #2021 task + 10000 eps_
data_10000 = np.load('history_results/feat_full_opt' + file_suffix + '.npy')
max_indx = 82
full_opt_plot_1000 = np.zeros((len(COLUMNS),NUM_POL,2))
full_opt_plot_10000 = np.zeros((len(COLUMNS),NUM_POL,2))
TRIALS_PER_EPISODE = 40

for pol in range(len(pol_list)):
    for sbj in range(max_indx):
        for feat_indx in range(len(COLUMNS)):
            full_opt_plot_1000[feat_indx][pol][0] += np.mean(data_1000[feat_indx][pol][sbj])/max_indx
            full_opt_plot_10000[feat_indx][pol][0] += np.mean(data_10000[feat_indx][pol][sbj]) / max_indx
    for feat_indx in range(len(COLUMNS)):
        full_opt_plot_1000[feat_indx][pol][1] = stdev(data_1000[feat_indx][pol]) / np.sqrt(len(data_1000[feat_indx][pol]))
        full_opt_plot_10000[feat_indx][pol][1] = stdev(data_10000[feat_indx][pol]) / np.sqrt(
            len(data_10000[feat_indx][pol]))

for feat_indx in range(len(COLUMNS)):
    temp1_mean = np.zeros(NUM_POL)
    temp1_std = np.zeros(NUM_POL)
    temp2_mean = np.zeros(NUM_POL)
    temp2_std = np.zeros(NUM_POL)
    for ii in range(NUM_POL):
        temp1_mean[ii]=full_opt_plot_1000[feat_indx][ii][0]
        temp1_std[ii] = full_opt_plot_1000[feat_indx][ii][1]
        temp2_mean[ii] = full_opt_plot_10000[feat_indx][ii][0]
        temp2_std[ii] = full_opt_plot_10000[feat_indx][ii][1]
    plt.bar(range(NUM_POL), temp1_mean)
    plt.errorbar(range(NUM_POL), temp1_mean, yerr=temp1_std)
    plt.bar(range(NUM_POL), temp2_mean)
    plt.errorbar(range(NUM_POL), temp2_mean, yerr=temp2_std)
    plt.xticks(range(NUM_POL), pol_list_tick)
    plt.title(COLUMNS[feat_indx])
    plt.savefig('history_results/'+COLUMNS[feat_indx]+'_plot_epoch_compare.png')
    plt.clf()


file_suffix = '_20210601_2021_ep1000_delta_control' #2021 task + 1000 eps
pol_1000 = np.load('history_results/optimal_policy'+file_suffix+'.npy')
file_suffix = '_20210601_2021_delta_control' #2021 task + 10000 eps_
pol_10000 = np.load('history_results/optimal_policy'+file_suffix+'.npy')

for pol in range(NUM_POL):
    print(pol_list[pol])
    pol_acts_1000 = np.zeros((NUM_ACT, TRIALS_PER_EPISODE))
    pol_acts_10000 = np.zeros((NUM_ACT, TRIALS_PER_EPISODE))
    for ii in range(82):
        for t_indx in range(TRIALS_PER_EPISODE ):
            pol_acts_1000[int(pol_1000[pol][ii][t_indx][0])][t_indx] += 1
            pol_acts_10000[int(pol_10000[pol][ii][t_indx][0])][t_indx] += 1

    pol_acts_1000 /= 82
    pol_acts_10000 /= 82

    for ii in range(len(action_list)):
        plt.plot(pol_acts_1000[ii], label=ii)
        plt.plot(pol_acts_10000[ii], label=ii)
        plt.legend(['1000 eps','10000 eps'], loc=5)
        plt.ylabel('Action Frequency')
        plt.xlabel('Episode')
        plt.title('Action frequency in the '+pol_list[pol]+' optimal sequences')
        plt.ylim((0, 1))
        plt.savefig('history_results/Action_frequency_ep_compare_'+pol_list[pol] +'_'+ action_list[ii] + file_suffix +'_opt.png')
        plt.clf()
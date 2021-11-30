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
from sklearn.metrics import r2_score
import matplotlib.cm as cm
import scipy.io
from math import ceil
from sklearn.linear_model import LinearRegression

#action_list = ['Nill','R-recover-visited', 'R-recover-unvisited']#['Nil','R-swap', 'R-decay','R-recover']#, 'Uncertainty']
action_list = ['Nil','0.5<->0.9','S<->F','R-recover-visited','R-recover-unvisit']#, 'Uncertainty']
tsne_go = False
NUM_ACT =len(action_list)
pol_list = ['min-rpe','max-rpe','min-spe','max-spe','min-rpe-min-spe','max-rpe-max-spe','min-rpe-max-spe','max-rpe-min-spe']
pol_list_tick = ['minR(LR)','maxR(HR)','minS(LS)','maxS(HS)','LRLS','HRHS','LRHS','HRLS']
file_suffix = '_2021_20_trials_delta_control_highest' #2021 task + 10000 eps_
#file_suffix = '_20210616_2019_delta_control'#'_20210331_Q_fix_delta_control'#'_20210329_Q_fix'#'_20210325_RPE' #'_20210304' #_repro_mode202010'
#file_suffix = '_20210601_2021_ep1000_delta_control' #2021 task + 1000 eps
#file_suffix = '_20210601_2021_delta_control' #2021 task + 10000 eps_
#file_suffix = '_20210520_2020_delta_control'
#file_suffix = '20210616_2019_delta_control'
if file_suffix == '_20210520_2020_delta_control':
    pol_list = ['min-rpe', 'max-rpe']
if file_suffix == '_2020_20_trials_delta_control_highest':
    pol_list = ['min-rpe', 'max-rpe']
NUM_POL = len(pol_list)
#COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'd_p_mb', 'ctrl_reward', 'score']
COLUMNS = ['rpe','spe','ctrl_reward','score','0','10','20','40','visit', 'applied_reward']
DETAIL_COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'd_p_mb', 'ctrl_reward', 'score', 'action', 'rpe1', 'rpe2','spe1','spe2']
DETAIL_COLUMNS = DETAIL_COLUMNS + ['0', '10', '20', '40', 'visit', 'applied_reward']
folderpath = '20210721'
with open('history_results/' + folderpath + '/Analysis-Object-' + pol_list[0] + '-00' + file_suffix + '.pkl','rb') as f:
    data = pickle.load(f)
    NUM_EPISODES, NUM_FULL_FEATS_DATA = data.data[pol_list[0]][0].shape
    NUM_FULL_TRIALS, NUM_FULL_FEATS_DETAIL = data.detail[pol_list[0]][0].shape
    TRIALS_PER_EPISODE = ceil(NUM_FULL_TRIALS / NUM_EPISODES)

opt_data = np.load('history_results/' + folderpath + '/feat_full_opt_full_trials'+file_suffix+'.npy')
max_indx = 82
R2 = np.zeros((len(pol_list),max_indx))
R2_1 = np.zeros((len(pol_list),max_indx))
R2_2 = np.zeros((len(pol_list),max_indx))

#RPE1 = opt_data[DETAIL_COLUMNS.index('rpe1')]
#RPE2 = opt_data[DETAIL_COLUMNS.index('rpe2')]
#SPE1 = opt_data[DETAIL_COLUMNS.index('spe1')]
#SPE2 = opt_data[DETAIL_COLUMNS.index('spe2')]
scores = np.zeros((len(pol_list),max_indx))
scores_1 = np.zeros((len(pol_list),max_indx))
scores_2 = np.zeros((len(pol_list),max_indx))
for pol in range(len(pol_list)):
    for sbj in range(max_indx):
        with open('history_results/' + folderpath + '/Analysis-Object-'+pol_list[pol]+'-{0:02d}'.format(sbj)+file_suffix+'.pkl','rb') as f:
            data = pickle.load(f)
        RPE1 = data.detail[pol_list[pol]][0]['rpe1'][2000:-1]
        RPE2 = data.detail[pol_list[pol]][0]['rpe2'][2000:-1]
        SPE1 = data.detail[pol_list[pol]][0]['spe1'][2000:-1]
        SPE2 = data.detail[pol_list[pol]][0]['spe2'][2000:-1]
        reg_ols = LinearRegression().fit(SPE1.values.reshape(-1,1), RPE1.values.reshape(-1,1))
        tmp = reg_ols.predict(SPE1.values.reshape(-1,1))
        R2_1[pol][sbj] = r2_score(RPE1.values.reshape(-1, 1),tmp)
        #print(str(reg_ols.coef_))
        scores_1[pol][sbj] = reg_ols.score(SPE1.values.reshape(-1,1), RPE1.values.reshape(-1,1))
        reg_ols = LinearRegression().fit(SPE2.values.reshape(-1, 1), RPE2.values.reshape(-1, 1))
        tmp = reg_ols.predict(SPE2.values.reshape(-1, 1))
        R2_2[pol][sbj] = r2_score(RPE2.values.reshape(-1, 1), tmp)
        # print(str(reg_ols.coef_))
        scores_2[pol][sbj] = reg_ols.score(SPE2.values.reshape(-1, 1), RPE2.values.reshape(-1, 1))
        reg_ols = LinearRegression().fit(np.concatenate((SPE1.values.reshape(-1, 1), SPE2.values.reshape(-1, 1))),
                          np.concatenate((RPE1.values.reshape(-1, 1), RPE2.values.reshape(-1, 1))))
        # print(np.concatenate((SPE1[pol][sbj].reshape(-1,1),SPE2[pol][sbj].reshape(-1,1))))
        tmp = reg_ols.predict(
            np.concatenate((SPE1.values.reshape(-1, 1), SPE2.values.reshape(-1, 1))))
        R2[pol][sbj] = r2_score(np.concatenate((RPE1.values.reshape(-1, 1), RPE2.values.reshape(-1, 1))), tmp)
        #print(str(reg_ols.coef_))
        scores[pol][sbj] = reg_ols.score(np.concatenate((SPE1.values.reshape(-1, 1), SPE2.values.reshape(-1, 1))),
                                         np.concatenate((RPE1.values.reshape(-1, 1), RPE2.values.reshape(-1, 1))))
        plt.scatter(np.concatenate((SPE1.values.reshape(-1, 1), SPE2.values.reshape(-1, 1))),
                    np.concatenate((RPE1.values.reshape(-1, 1), RPE2.values.reshape(-1, 1))))
    plt.savefig(
        'history_results/' + folderpath + '/PE_corrs_plot_whole_action_' + pol_list[pol] + file_suffix + '.png')
    plt.clf()
    results1 = stats.ttest_1samp(R2_1[pol],0)
    results1_2 = stats.ttest_1samp(scores_1[pol], 0)
    #print(pol_list[pol])
    #print(scores[pol])
    #print(str(results2.pvalue))
    results2 = stats.ttest_1samp(R2_2[pol],0)
    results2_2 = stats.ttest_1samp(scores_2[pol], 0)
    #print(pol_list[pol])
    #print(scores[pol])
    #print(str(results2.pvalue))
    results = stats.ttest_1samp(R2[pol],0)
    results_2 = stats.ttest_1samp(scores[pol], 0)
    print(pol_list[pol])
    #print(scores)
    #print(R2[pol])
    print(np.mean(R2[pol]))
    print(str(results.pvalue))

#print(R2)

temp_mean = np.mean(scores_1, axis = 1)
temp_std = np.std(scores_1, axis = 1)
plt.bar(range(NUM_POL), temp_mean)
plt.errorbar(range(NUM_POL), temp_mean, yerr=temp_std)
plt.xticks(range(NUM_POL), pol_list_tick)
plt.title('PE correlation')
plt.savefig('history_results/' + folderpath + '/PE_corrs_plot_1st_action' + file_suffix + '.png')
plt.clf()

temp_mean = np.mean(scores_2, axis = 1)
temp_std = np.std(scores_2, axis = 1)
plt.bar(range(NUM_POL), temp_mean)
plt.errorbar(range(NUM_POL), temp_mean, yerr=temp_std)
plt.xticks(range(NUM_POL), pol_list_tick)
plt.title('PE correlation')
plt.savefig('history_results/' + folderpath + '/PE_corrs_plot_2nd_action' + file_suffix + '.png')
plt.clf()

temp_mean = np.mean(scores, axis = 1)
temp_std = np.std(scores, axis = 1)
plt.bar(range(NUM_POL), temp_mean)
plt.errorbar(range(NUM_POL), temp_mean, yerr=temp_std)
plt.xticks(range(NUM_POL), pol_list_tick)
plt.title('PE correlation')
plt.savefig('history_results/' + folderpath + '/PE_corrs_plot_whole_action' + file_suffix + '.png')
plt.clf()



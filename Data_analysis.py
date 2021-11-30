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
file_suffix = '_20210616_2019_delta_control'#'_20210331_Q_fix_delta_control'#'_20210329_Q_fix'#'_20210325_RPE' #'_20210304' #_repro_mode202010'
#file_suffix = '_20210601_2021_ep1000_delta_control' #2021 task + 1000 eps
file_suffix = '_20210601_2021_delta_control' #2021 task + 10000 eps_
#file_suffix = '_20210520_2020_delta_control'
#file_suffix = '20210616_2019_delta_control'
file_suffix = '_2021_20_trials_delta_control' # _{2019,2020,2021}_20_trials_delta_control
if file_suffix == '_20210520_2020_delta_control':
    pol_list = ['min-rpe', 'max-rpe']
NUM_POL = len(pol_list)
#COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'd_p_mb', 'ctrl_reward', 'score']
#COLUMNS = ['rpe','spe','ctrl_reward','score','0','10','20','40','visit', 'applied_reward','rpe1','rpe2','spe1','spe2']
COLUMNS = ['ctrl_reward']
with open('history_results/Analysis-Object-' + pol_list[0] + '-00' + file_suffix + '.pkl','rb') as f:
    data = pickle.load(f)
    NUM_EPISODES, NUM_FULL_FEATS_DATA = data.data[pol_list[0]][0].shape
    NUM_FULL_TRIALS, NUM_FULL_FEATS_DETAIL = data.detail[pol_list[0]][0].shape
    TRIALS_PER_EPISODE = ceil(NUM_FULL_TRIALS / NUM_EPISODES)
max_indx = 82
#TRIALS_PER_EPISODE = 20
#NUM_EPISODES = 1000
full_data = np.zeros((len(COLUMNS),NUM_POL,max_indx,NUM_EPISODES))
full_detail = np.zeros((len(COLUMNS),NUM_POL,max_indx,TRIALS_PER_EPISODE*NUM_EPISODES))
full_opt = np.zeros((len(COLUMNS),NUM_POL,max_indx))
full_trials_opt = np.zeros((len(COLUMNS),NUM_POL,max_indx,TRIALS_PER_EPISODE))
real_opt = np.zeros((len(COLUMNS),NUM_POL,max_indx))
full_plot = np.zeros((len(COLUMNS),NUM_POL,2))
full_opt_plot = np.zeros((len(COLUMNS),NUM_POL,2))
real_opt_plot = np.zeros((len(COLUMNS),NUM_POL,2))
full_SBJ_plot = np.zeros((len(COLUMNS),max_indx,2))
opt_index= np.zeros((NUM_POL,max_indx))
opt_pol = np.zeros((NUM_POL,max_indx,TRIALS_PER_EPISODE ,2))
RPE_plot_detail = np.zeros((NUM_POL,2))
SPE_plot_detail = np.zeros((NUM_POL,2))
PMB_plot_detail = np.zeros((NUM_POL,2))
RWD_plot_detail = np.zeros((NUM_POL,2))
save_pol = np.zeros((NUM_POL,TRIALS_PER_EPISODE ,2))
time_delay = [1.7, 4.06, 6.99]
single_game_time = 8.4
gen_regressor = False
TR = 2.8
full_detail_expanded = np.zeros((ceil(max_indx*NUM_POL*0.8*NUM_EPISODES),ceil(TRIALS_PER_EPISODE *single_game_time/TR)))
for pol in range(len(pol_list)):
    print(pol_list[pol])
    for sbj in range(max_indx):
        with open('history_results/Analysis-Object-'+pol_list[pol]+'-{0:02d}'.format(sbj)+file_suffix+'.pkl','rb') as f:
            data = pickle.load(f)
            #data.current_data: eps, data.current_detial : eps*trials
        for feat_indx in range(len(COLUMNS)):
            # if feat_indx == 0: full_data[feat_indx][pol][sbj] = data.data[pol_list[pol]][0][COLUMNS[feat_indx]]
            full_data[feat_indx][pol][sbj] = data.data[pol_list[pol]][0][COLUMNS[feat_indx]]
            full_detail[feat_indx][pol][sbj] = data.detail[pol_list[pol]][0][COLUMNS[feat_indx]]
            opt_index[pol][sbj]=data.data[pol_list[pol]][0]['ctrl_reward'].loc[ceil(0.2 * len(data.data[pol_list[pol]][0])):].idxmax()
            #print(data.data[pol_list[pol]][0]['ctrl_reward'][opt_index[pol][sbj]])
            for t_indx in range(TRIALS_PER_EPISODE ):
                opt_pol[pol][sbj][t_indx] = data.detail[pol_list[pol]][0]['action'].loc[
                                      opt_index[pol][sbj]*TRIALS_PER_EPISODE -TRIALS_PER_EPISODE +t_indx]
                # opt_pol[pol][sbj][t_indx][1] = data.detail[pol_list[pol]][0]['action'].loc[opt_index[pol][sbj] * TRIALS_PER_EPISODE  - TRIALS_PER_EPISODE  + t_indx][1]
            full_opt[feat_indx][pol][sbj] = sum(full_detail[feat_indx][pol][sbj]
                                                [int(TRIALS_PER_EPISODE *opt_index[pol][sbj]-TRIALS_PER_EPISODE ):
                                                 int(TRIALS_PER_EPISODE *opt_index[pol][sbj])]) / TRIALS_PER_EPISODE
            full_trials_opt[feat_indx][pol][sbj] = full_detail[feat_indx][pol][sbj][int(TRIALS_PER_EPISODE *opt_index[pol][sbj]-TRIALS_PER_EPISODE ):
                                                 int(TRIALS_PER_EPISODE *opt_index[pol][sbj])]
            full_opt_plot[feat_indx][pol][0] += np.mean(full_opt[feat_indx][pol][sbj])/max_indx
            real_opt[feat_indx][pol][sbj] = sum(full_detail[feat_indx][pol][sbj]
                                                [int(TRIALS_PER_EPISODE * opt_index[pol][sbj] - ceil(TRIALS_PER_EPISODE /2)):
                                                 int(TRIALS_PER_EPISODE * opt_index[pol][sbj])]) / TRIALS_PER_EPISODE
            real_opt_plot[feat_indx][pol][0] += np.mean(real_opt[feat_indx][pol][sbj])/max_indx
            full_plot[feat_indx][pol][0] += np.mean(full_detail[feat_indx][pol][sbj][ceil(0.2*TRIALS_PER_EPISODE*NUM_EPISODES):])/max_indx
            full_SBJ_plot[feat_indx][sbj][0] += np.mean(full_detail[feat_indx][pol][sbj][ceil(0.2*TRIALS_PER_EPISODE*NUM_EPISODES):])/4
        for ep_indx in range(ceil(0.2*NUM_EPISODES+1),NUM_EPISODES):
            for trial_indx in range(TRIALS_PER_EPISODE ):
                rpe1_tindx = ceil((trial_indx * single_game_time + time_delay[1]) / TR)
                rpe2_tindx = ceil((trial_indx * single_game_time + time_delay[2]) / TR)
                if gen_regressor == True:
                    full_detail_expanded[ceil(pol * max_indx * 0.8 *TRIALS_PER_EPISODE + sbj * 0.8 *TRIALS_PER_EPISODE + ep_indx
                                         - 0.2 *TRIALS_PER_EPISODE)][rpe1_tindx] = \
                                        full_detail[1][pol][sbj][ep_indx * TRIALS_PER_EPISODE  + trial_indx]
                    full_detail_expanded[ceil(pol * max_indx * 0.8 *TRIALS_PER_EPISODE + sbj * 0.8 *TRIALS_PER_EPISODE + ep_indx
                                         - 0.8 *TRIALS_PER_EPISODE)][rpe2_tindx] = \
                                        full_detail[2][pol][sbj][ep_indx * TRIALS_PER_EPISODE  + trial_indx]
#    RPE_plot[pol][0] = sum(opt_RPE[pol])/len(opt_RPE[pol])
 #   SPE_plot[pol][0] = sum(opt_SPE[pol]) / len(opt_SPE[pol])
    for feat_indx in range(len(COLUMNS)):
        full_plot[feat_indx][pol][1] = stdev(full_opt[feat_indx][pol])/np.sqrt(len(full_opt[feat_indx][pol]))
        full_opt_plot[feat_indx][pol][1] = stdev(full_opt[feat_indx][pol]) / np.sqrt(len(full_opt[feat_indx][pol]))
        real_opt_plot[feat_indx][pol][1] = stdev(real_opt[feat_indx][pol]) / np.sqrt(len(real_opt[feat_indx][pol]))
    save_pol[pol] = opt_pol[pol][81]

for sbj in range(max_indx):
    tmp = np.transpose(full_opt[feat_indx])
    full_SBJ_plot[feat_indx][sbj][1] = stdev(tmp[sbj])/np.sqrt(len(pol_list))

np.save('history_results/optimal_policy'+file_suffix+'.npy',opt_pol)
np.save('history_results/feat'+file_suffix+'.npy',full_detail)
print(full_opt.shape)
np.save('history_results/feat_full_opt'+file_suffix+'.npy',full_opt)
np.save('history_results/feat_full_opt_full_trials'+file_suffix+'.npy',full_trials_opt)
np.save('history_results/optimal_policy_index'+file_suffix+'.npy',opt_index)
#if gen_regressor == True:
#   scipy.io.savemat('history_results/RPE_regressor'+file_suffix+'.mat',{'RPE': full_detail_expanded})

for feat_indx in range(len(COLUMNS)):
    temp_mean = np.zeros(NUM_POL)
    temp_std = np.zeros(NUM_POL)
    for ii in range(NUM_POL):
        temp_mean[ii]=full_plot[feat_indx][ii][0]
        temp_std[ii] = full_plot[feat_indx][ii][1]
    plt.bar(range(NUM_POL), temp_mean)
    plt.errorbar(range(NUM_POL), temp_mean, yerr=temp_std)
    plt.xticks(range(NUM_POL), pol_list_tick)
    plt.title(COLUMNS[feat_indx])
    plt.savefig('history_results/'+COLUMNS[feat_indx]+'_plot' + file_suffix + '.png')
    plt.clf()


'''
for feat_indx in range(len(COLUMNS)):
    temp_mean = np.zeros(NUM_POL)
    temp_std = np.zeros(NUM_POL)
    for ii in range(NUM_POL):
        temp_mean[ii]=full_opt_plot[feat_indx][ii][0]
        temp_std[ii] = full_opt_plot[feat_indx][ii][1]
    plt.bar(range(NUM_POL), temp_mean)
    plt.errorbar(range(NUM_POL), temp_mean, yerr=temp_std)
    plt.xticks(range(NUM_POL), pol_list)
    plt.title(COLUMNS[feat_indx])
    plt.savefig('history_results/'+COLUMNS[feat_indx]+'_full_opt_plot' + file_suffix + '.png')
    plt.clf()
'''
'''
for feat_indx in range(len(COLUMNS)):
    temp_mean = np.zeros(NUM_POL)
    temp_std = np.zeros(NUM_POL)
    for ii in range(NUM_POL):
        temp_mean[ii]= real_opt_plot[feat_indx][ii][0]
        temp_std[ii] = real_opt_plot[feat_indx][ii][1]
    plt.bar(range(NUM_POL), temp_mean)
    plt.errorbar(range(NUM_POL), temp_mean, yerr=temp_std)
    plt.xticks(range(NUM_POL), pol_list)
    plt.title(COLUMNS[feat_indx])
    plt.savefig('history_results/'+COLUMNS[feat_indx]+'_real_opt_plot' + file_suffix + '.png')
    plt.clf()
'''
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


'''
#Scatter plot
for mode_idf in range(len(pol_list)):
    plt.scatter(full_opt[4][mode_idf],full_opt[0][mode_idf])
    plt.title('PMB-RPE '+pol_list[mode_idf])
    plt.savefig('history_results/PMB_'+pol_list[mode_idf]+'_RPE_plot'+file_suffix+'.png')
    plt.clf()

    plt.scatter(full_opt[4][mode_idf],full_opt[6][mode_idf])
    plt.title('PMB-CTR '+pol_list[mode_idf])
    plt.savefig('history_results/PMB_'+pol_list[mode_idf]+'_CTR_plot'+file_suffix+'.png')
    plt.clf()

    plt.scatter(full_opt[4][mode_idf],full_opt[7][mode_idf])
    plt.title('PMB-RWD '+pol_list[mode_idf])
    plt.savefig('history_results/PMB_'+pol_list[mode_idf]+'_plot'+file_suffix+'.png')
    plt.clf()

'''
for feat_indx in range(len(COLUMNS)):
    for mode_idf in range(len(pol_list)):
        #plt.plot(np.transpose(full_detail[feat_indx][mode_idf]))
        #plt.savefig('history_results/'+COLUMNS[feat_indx]+'_'+pol_list[mode_idf]+'_full_plot' + file_suffix + '.png')
        #plt.clf()
        plt.plot(np.transpose(full_data[feat_indx][mode_idf]))
        plt.savefig(
            'history_results/' + COLUMNS[feat_indx] + '_' + pol_list[mode_idf] + '_abstract_plot' + file_suffix + '.png')
        plt.clf()

for pol in range(NUM_POL):
    print(pol_list[pol])
    pol_acts = np.zeros((NUM_ACT, TRIALS_PER_EPISODE ))
    for ii in range(82):
        for t_indx in range(TRIALS_PER_EPISODE ):
            pol_acts[int(opt_pol[pol][ii][t_indx][0])][t_indx] += 1

    pol_acts /= 82

    for ii in range(len(pol_acts)):
        plt.plot(pol_acts[ii], label=ii)
    plt.legend(action_list, loc=5)
    plt.ylabel('Action Frequency')
    plt.xlabel('Episode')
    plt.title('Action frequency in the '+pol_list[pol]+' optimal sequences')
    plt.ylim((0, 1))
    plt.savefig('history_results/Action_frequency_'+pol_list[pol] + file_suffix +'_opt.png')
    plt.clf()

'''
    pol_acts_decay = np.zeros((max_indx,20))
    for ii in range(82):
        for t_indx in range(20):
            pol_acts_decay[ii][t_indx] = opt_pol[pol][ii][t_indx][1]
        plt.plot(pol_acts_decay[ii], label=ii)
        print(pol_acts_decay[ii])
    plt.ylabel('decay_values')
    plt.xlabel('Episode')
    plt.title('Decay rate in the '+pol_list[pol]+' optimal sequences')
    plt.ylim((0, 1))
    plt.savefig('history_results/Decay_rate_'+pol_list[pol] + file_suffix +'_opt.png')
    plt.clf()
'''

'''
for pol_indx,pol_name in enumerate(pol_list):
    plt.imshow(opt_pol[pol_indx], interpolation='bilinear', cmap=cm.Greys_r)
    plt.title(pol_name)
    plt.savefig(pol_name+'_policy_map.png')
    plt.clf()
    '''
'''
pre_RPE_df=pd.DataFrame(data=np.transpose(opt_RPE),index=range(82),columns=pol_list)
print(stats.ttest_rel(pre_RPE_df['min-rpe'],pre_RPE_df['max-rpe']))
pre_SPE_df=pd.DataFrame(data=np.transpose(opt_SPE),index=range(82),columns=pol_list)
print(stats.ttest_rel(pre_SPE_df['min-spe'],pre_SPE_df['max-spe']))
pre_PMB_df=pd.DataFrame(data=np.transpose(opt_PMB),index=range(82),columns=pol_list)
print(stats.ttest_rel(pre_PMB_df['min-spe'],pre_PMB_df['max-spe']))
print(stats.ttest_rel(pre_PMB_df['min-rpe'],pre_PMB_df['max-rpe']))


#RPE_df=pd.melt(pre_RPE_df.reset_index(),id_vars=['index'],value_vars=range(82))
'''

'''
#
if tsne_go == True:
    for pol in range(NUM_POL):
        print(pol)
        sample_tsne = TSNE(n_components=2, learning_rate=200)
        tsne_results = sample_tsne.fit_transform(opt_pol[pol])
        tsne_x = []
        tsne_y = []
        tsne_c = []
        for ii in range(82):
            tsne_x.append(tsne_results[ii][0])
            tsne_y.append(tsne_results[ii][1])
        scatter = plt.scatter(tsne_x, tsne_y)
        print(len(tsne_x))
        plt.title(pol_list[pol])
        plt.savefig('policy tSNE result in the '+ pol_list[pol] + file_suffix +'.png')
        plt.clf()

    print('tots')
    opt_pol_tot = np.concatenate([opt_pol[0], opt_pol[1], opt_pol[2], opt_pol[3], opt_pol[4], opt_pol[5], opt_pol[6], opt_pol[7]])
    tsne_results = sample_tsne.fit_transform(opt_pol_tot)
    tsne_x = []
    tsne_y = []
    fig, ax = plt.subplots()
    if len(opt_pol) > 4:
        for ii in range(82 * 8):
            tsne_x.append(tsne_results[ii][0])
            tsne_y.append(tsne_results[ii][1])

        ax.scatter(tsne_x[0:82], tsne_y[0:82], color='blue', label='SPE max')
        ax.scatter(tsne_x[82:164], tsne_y[82:164], color='red', label='SPE min')
        ax.scatter(tsne_x[164:246], tsne_y[164:246], color='green', label='RPE max')
        ax.scatter(tsne_x[246:328], tsne_y[246:328], color='orange', label='RPE min')
        ax.scatter(tsne_x[328:410], tsne_y[328:410], color='purple', label='COR max')
        ax.scatter(tsne_x[410:492], tsne_y[410:492], color='yellow', label='COR min')
        ax.scatter(tsne_x[492:574], tsne_y[492:574], color='black', label='SEP SPE')
        ax.scatter(tsne_x[574:], tsne_y[574:], color='cyan', label='SEP RPE')
        plt.legend(('SPE max', 'SPE min', 'RPE max', 'RPE min', 'COR max', 'COR min', 'SEP SPE', 'SEP RPE'))
    else:
        for ii in range(82 * 4):
            tsne_x.append(tsne_results[ii][0])
            tsne_y.append(tsne_results[ii][1])

        ax.scatter(tsne_x[0:82], tsne_y[0:82], color='blue', label='SPE max')
        ax.scatter(tsne_x[82:164], tsne_y[82:164], color='red', label='SPE min')
        ax.scatter(tsne_x[164:246], tsne_y[164:246], color='green', label='RPE max')
        ax.scatter(tsne_x[246:], tsne_y[246:], color='orange', label='RPE min')
        plt.legend(('SPE max', 'SPE min', 'RPE max', 'RPE min'))
    print(len(tsne_x))
    plt.title('Total polices')
    plt.savefig('policy tSNE result in the total scenarios' + file_suffix + '.png')
    plt.clf()


if tsne_go == True:
    sample_tsne = TSNE(n_components=2, learning_rate=200)
    SPEs=np.concatenate([opt_pol[0], opt_pol[1]])
    RPEs=np.concatenate([opt_pol[2], opt_pol[3]])
    TOTs = np.concatenate([RPEs, SPEs])
    if len(opt_pol) > 4:
        CORs = np.concatenate([opt_pol[4], opt_pol[5]])
        SEPs = np.concatenate([opt_pol[6], opt_pol[7]])
        TOTs = np.concatenate([RPEs, SPEs, CORs, SEPs])
    SPE_feat = np.zeros((164,NUM_ACT))
    RPE_feat = np.zeros((164,NUM_ACT))
    TOT_feat = np.zeros((328,NUM_ACT))
    if len(opt_pol) > 4:
        COR_feat = np.zeros((164, NUM_ACT))
        SEP_feat = np.zeros((164, NUM_ACT))
        TOT_feat = np.zeros((656, NUM_ACT))
    for sbj in range(len(SPEs)):
        for index in range(len(SPEs[sbj])):
            SPE_feat[sbj][int(SPEs[sbj][index])] += 1

    for sbj in range(len(RPEs)):
        for index in range(len(RPEs[sbj])):
            RPE_feat[sbj][int(RPEs[sbj][index])] += 1

    for sbj in range(len(TOTs)):
        for index in range(len(TOTs[sbj])):
            TOT_feat[sbj][int(TOTs[sbj][index])] += 1
    if len(opt_pol) > 4:
        for sbj in range(len(CORs)):
            for index in range(len(CORs[sbj])):
                COR_feat[sbj][int(CORs[sbj][index])] += 1
        for sbj in range(len(SEPs)):
            for index in range(len(SEPs[sbj])):
                SEP_feat[sbj][int(SEPs[sbj][index])] += 1

    print('SPEs')
    tsne_results = sample_tsne.fit_transform(SPE_feat)
    tsne_x = []
    tsne_y = []
    fig, ax = plt.subplots()
    for ii in range(82*2):
        tsne_x.append(tsne_results[ii][0])
        tsne_y.append(tsne_results[ii][1])
    ax.scatter(tsne_x[0:82], tsne_y[0:82], color = 'blue', label='SPE max')
    ax.scatter(tsne_x[82:], tsne_y[82:], color = 'red', label='SPE min')
    print(len(tsne_x))
    plt.title('SPE min-max polices')
    plt.legend(('SPE max','SPE min'))
    plt.savefig('policy feature tSNE result in the SPE scenarios'+ file_suffix +'.png')
    plt.clf()

    print('RPEs')
    tsne_results = sample_tsne.fit_transform(RPE_feat)
    tsne_x = []
    tsne_y = []
    fig, ax = plt.subplots()
    for ii in range(82*2):
        tsne_x.append(tsne_results[ii][0])
        tsne_y.append(tsne_results[ii][1])
    ax.scatter(tsne_x[0:82], tsne_y[0:82], color = 'green', label='RPE max')
    ax.scatter(tsne_x[82:], tsne_y[82:], color = 'orange', label='RPE min')
    print(len(tsne_x))
    plt.title('RPE min-max polices')
    plt.legend(('RPE max' ,'RPE min'))
    plt.savefig('policy feature tSNE result in the RPE scenarios'+ file_suffix +'.png')
    plt.clf()

    if len(opt_pol) > 4:
        print('CORs')
        tsne_results = sample_tsne.fit_transform(COR_feat)
        tsne_x = []
        tsne_y = []
        fig, ax = plt.subplots()
        for ii in range(82 * 2):
            tsne_x.append(tsne_results[ii][0])
            tsne_y.append(tsne_results[ii][1])
        ax.scatter(tsne_x[0:82], tsne_y[0:82], color='purple', label='COR max')
        ax.scatter(tsne_x[82:], tsne_y[82:], color='yellow', label='COR min')
        print(len(tsne_x))
        plt.title('COR min-max polices')
        plt.legend(('COR max', 'COR min'))
        plt.savefig('policy feature tSNE result in the COR scenarios' + file_suffix + '.png')
        plt.clf()

        print('SEPs')
        tsne_results = sample_tsne.fit_transform(SEP_feat)
        tsne_x = []
        tsne_y = []
        fig, ax = plt.subplots()
        for ii in range(82 * 2):
            tsne_x.append(tsne_results[ii][0])
            tsne_y.append(tsne_results[ii][1])
        ax.scatter(tsne_x[0:82], tsne_y[0:82], color='black', label='SEP SPE')
        ax.scatter(tsne_x[82:], tsne_y[82:], color='cyan', label='SEP RPE')
        print(len(tsne_x))
        plt.title('SEP SPE-RPE polices')
        plt.legend(('SEP SPE', 'SEP RPE'))
        plt.savefig('policy feature tSNE result in the SEP scenarios' + file_suffix + '.png')
        plt.clf()

    print('tots')
    tsne_results = sample_tsne.fit_transform(TOT_feat)
    tsne_x = []
    tsne_y = []
    fig, ax = plt.subplots()
    if len(opt_pol) > 4:
        for ii in range(82 * 8):
            tsne_x.append(tsne_results[ii][0])
            tsne_y.append(tsne_results[ii][1])

        ax.scatter(tsne_x[0:82], tsne_y[0:82], color='blue', label='SPE max')
        ax.scatter(tsne_x[82:164], tsne_y[82:164], color='red', label='SPE min')
        ax.scatter(tsne_x[164:246], tsne_y[164:246], color='green', label='RPE max')
        ax.scatter(tsne_x[246:328], tsne_y[246:328], color='orange', label='RPE min')
        ax.scatter(tsne_x[328:410], tsne_y[328:410], color='purple', label='COR max')
        ax.scatter(tsne_x[410:492], tsne_y[410:492], color='yellow', label='COR min')
        ax.scatter(tsne_x[492:574], tsne_y[492:574], color='black', label='SEP SPE')
        ax.scatter(tsne_x[574:], tsne_y[574:], color='cyan', label='SEP RPE')
        plt.legend(('SPE max', 'SPE min', 'RPE max', 'RPE min', 'COR max', 'COR min', 'SEP SPE', 'SEP RPE'))
    else:
        for ii in range(82*4):
            tsne_x.append(tsne_results[ii][0])
            tsne_y.append(tsne_results[ii][1])

        ax.scatter(tsne_x[0:82], tsne_y[0:82], color = 'blue', label='SPE max')
        ax.scatter(tsne_x[82:164], tsne_y[82:164], color = 'red', label='SPE min')
        ax.scatter(tsne_x[164:246], tsne_y[164:246], color = 'green', label='RPE max')
        ax.scatter(tsne_x[246:], tsne_y[246:], color = 'orange', label='RPE min')
        plt.legend(('SPE max', 'SPE min', 'RPE max', 'RPE min'))
    print(len(tsne_x))
    plt.title('Total polices')
    plt.savefig('policy feature tSNE result in the total scenarios'+ file_suffix +'.png')
    plt.clf()
'''
"""
for sbj in range(len(SPEs)):
    is_stoc = -1
    for index in range(len(SPEs[sbj])):

        SPE_feat[sbj][int(SPEs[sbj][index])] += 1
        if int(SPEs[sbj][index])==1:
            if is_stoc == 1:
                SPEs[sbj][index] = 0
            else:
                is_stoc = 1
        elif int(SPEs[sbj][index])==0:
            if is_stoc == 0:
                SPEs[sbj][index]==1
            else:
                is_stoc = 0

for sbj in range(len(RPEs)):
    is_stoc = -1
    for index in range(len(RPEs[sbj])):
        RPE_feat[sbj][int(RPEs[sbj][index])] += 1
        if int(RPEs[sbj][index]) == 1:
            if is_stoc == 1:
                RPEs[sbj][index] = 0
            else:
                is_stoc = 1
        elif int(RPEs[sbj][index]) == 0:
            if is_stoc == 0:
                RPEs[sbj][index] == 1
            else:
                is_stoc = 0

for sbj in range(len(TOTs)):
    is_stoc = -1
    for index in range(len(TOTs[sbj])):
        TOT_feat[sbj][int(TOTs[sbj][index])] += 1
        if int(TOTs[sbj][index]) == 1:
            if is_stoc == 1:
                TOTs[sbj][index] = 0
            else:
                is_stoc = 1
        elif int(TOTs[sbj][index]) == 0:
            if is_stoc == 0:
                TOTs[sbj][index] == 1
            else:
                is_stoc = 0
print('SPEs')
tsne_results = sample_tsne.fit_transform(SPEs)
tsne_x = []
tsne_y = []
fig, ax = plt.subplots()
for ii in range(82*2):
    tsne_x.append(tsne_results[ii][0])
    tsne_y.append(tsne_results[ii][1])
ax.scatter(tsne_x[0:82], tsne_y[0:82], color = 'blue', label='SPE max')
ax.scatter(tsne_x[82:], tsne_y[82:], color = 'red', label='SPE min')
print(len(tsne_x))
plt.title('SPE min-max polices')
plt.legend(('SPE max','SPE min'))
plt.savefig('redundancy decreased policy tSNE result in the SPE scenarios.png')
plt.clf()

print('RPEs')
tsne_results = sample_tsne.fit_transform(RPEs)
tsne_x = []
tsne_y = []
fig, ax = plt.subplots()
for ii in range(82*2):
    tsne_x.append(tsne_results[ii][0])
    tsne_y.append(tsne_results[ii][1])
ax.scatter(tsne_x[0:82], tsne_y[0:82], color = 'green', label='RPE max')
ax.scatter(tsne_x[82:], tsne_y[82:], color = 'orange', label='RPE min')
print(len(tsne_x))
plt.title('RPE min-max polices')
plt.legend(('RPE max' ,'RPE min'))
plt.savefig('redundancy decreased policy tSNE result in the RPE scenarios.png')
plt.clf()

print('tots')
tsne_results = sample_tsne.fit_transform(TOTs)
tsne_x = []
tsne_y = []
fig, ax = plt.subplots()
for ii in range(82*4):
    tsne_x.append(tsne_results[ii][0])
    tsne_y.append(tsne_results[ii][1])

ax.scatter(tsne_x[0:82], tsne_y[0:82], color = 'blue', label='SPE max')
ax.scatter(tsne_x[82:164], tsne_y[82:164], color = 'red', label='SPE min')
ax.scatter(tsne_x[164:246], tsne_y[164:246], color = 'green', label='RPE max')
ax.scatter(tsne_x[246:], tsne_y[246:], color = 'orange', label='RPE min')
print(len(tsne_x))
plt.title('Total polices')
plt.legend(('SPE max','SPE min', 'RPE max', 'RPE min'))
plt.savefig('redundancy decreased policy tSNE result in the total scenarios.png')
plt.clf()
"""
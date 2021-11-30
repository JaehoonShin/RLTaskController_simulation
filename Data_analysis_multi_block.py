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

action_list = ['Nill', 'R-recover-visited', 'R-recover-unvisited']#['Nil','R-swap', 'R-decay','R-recover']#, 'Uncertainty']
tsne_go = False
NUM_ACT =len(action_list)
pol_list = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe','random']
NUM_POL = len(pol_list)
file_suffix = '_2020_2_TR1_no_PMB'
COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'd_p_mb', 'ctrl_reward', 'score']
#COLUMNS = ['ctrl_reward']
#full_data = np.zeros((5,82,240,1000)) # Feature parameters x Policy block x Subject x Session x epi
full_detail = np.zeros((NUM_POL,82,240,20000))
full_opt = np.zeros((NUM_POL,82,240))
full_plot = np.zeros((NUM_POL,2))
sc_cnt = np.zeros(NUM_POL)
opt_index= np.zeros(82)
opt_pol = np.zeros((NUM_POL,82,20))
RPE_plot_detail = np.zeros((NUM_POL,2))
SPE_plot_detail = np.zeros((NUM_POL,2))
PMB_plot_detail = np.zeros((NUM_POL,2))
RWD_plot_detail = np.zeros((NUM_POL,2))
save_pol = np.zeros((NUM_POL,20))
sess_type = list(permutations(['1','2','3','4','5'],4))
#for pol in range(len(pol_list)):
for feat_indx in range(len(COLUMNS)):
    sc_cnt = np.zeros(NUM_POL)
    full_plot = np.zeros((NUM_POL, 2))
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
                #full_data[pol][sbj] = data.data[''.join(list(sess))][0][COLUMNS[feat_indx]]
                for blck in range(4):
                    pol = int(blck_scs[blck] - 1)
                    blck_indx = list(map(lambda x, y: x + y + 20 * (blck), list(range(20)) * 1000,
                             list(map(lambda z: (z // 20) * 80, list(range(20 * 1000))))))
                    full_detail[pol][sbj][int(sc_cnt[pol])] = data.detail[''.join(list(sess))][0][COLUMNS[feat_indx]][blck_indx]
                    opt_pol[pol][sbj] = data.detail[''.join(list(sess))][0]['action'].loc[
                                          opt_index[sbj]*80-80+blck*20:opt_index[sbj]*80-61+blck*20]
                    full_opt[pol][sbj][int(sc_cnt[pol])] = sum(full_detail[pol][sbj][int(sc_cnt[pol])][int(80*opt_index[sbj]-80+20*blck):int(80*opt_index[sbj]-61+20*blck)]) / 20
                    full_plot[pol][0] += np.mean(full_detail[pol][sbj][int(sc_cnt[pol])][4000:])/(82*240)
    #    RPE_plot[pol][0] = sum(opt_RPE[pol])/len(opt_RPE[pol])
     #   SPE_plot[pol][0] = sum(opt_SPE[pol]) / len(opt_SPE[pol])
        save_pol[pol] = opt_pol[pol][81]
        for indx in list(map(int,list(sess))):
            sc_cnt[indx-1] += 1


    for pol in range(5):
        tmp = np.zeros(240*82)
        for sbj in range(82):
            tmp[240*sbj:240*sbj+240] = np.mean(full_detail[pol][sbj][int(sc_cnt[pol])][4000:])
        full_plot[pol][1] = stdev(tmp)/np.sqrt(len(tmp))

    np.save('history_results/multi_optimal_policy'+file_suffix+COLUMNS[feat_indx]+'.npy',save_pol)


    plt.bar(range(NUM_POL), np.transpose(full_plot)[0])
    plt.errorbar(range(NUM_POL), np.transpose(full_plot)[0], yerr=np.transpose(full_plot)[1])
    plt.xticks(range(NUM_POL), pol_list)
    plt.title(COLUMNS[feat_indx])
    plt.savefig('history_results/multi_'+COLUMNS[feat_indx]+'_new_plot' + file_suffix + '.png')
    plt.clf()
'''
#Scatter plot
for mode_idf in range(len(pol_list)):
    plt.scatter(full_opt[4][mode_idf],full_opt[0][mode_idf])
    plt.title('PMB-RPE '+pol_list[mode_idf])
    plt.savefig('history_results/multi_PMB_'+pol_list[mode_idf]+'_RPE_plot'+file_suffix+'.png')
    plt.clf()

    plt.scatter(full_opt[4][mode_idf],full_opt[6][mode_idf])
    plt.title('PMB-CTR '+pol_list[mode_idf])
    plt.savefig('history_results/multi_PMB_'+pol_list[mode_idf]+'_CTR_plot'+file_suffix+'.png')
    plt.clf()

    plt.scatter(full_opt[4][mode_idf],full_opt[7][mode_idf])
    plt.title('PMB-RWD '+pol_list[mode_idf])
    plt.savefig('history_results/multi_PMB_'+pol_list[mode_idf]+'_plot'+file_suffix+'.png')
    plt.clf()


for feat_indx in range(len(COLUMNS)):
    for mode_idf in range(len(pol_list)):
        plt.plot(np.transpose(full_detail[feat_indx][mode_idf]))
        plt.savefig('history_results/multi_'+COLUMNS[feat_indx]+'_'+pol_list[mode_idf]+'_full_plot' + file_suffix + '.png')
        plt.clf()
        plt.plot(np.transpose(full_data[feat_indx][mode_idf]))
        plt.savefig(
            'history_results/multi_' + COLUMNS[feat_indx] + '_' + pol_list[mode_idf] + '_abstract_plot' + file_suffix + '.png')
        plt.clf()
'''
for pol in range(NUM_POL):
    pol_acts = np.zeros((NUM_ACT, 20))
    for ii in range(82):
        for t_indx in range(20):
            pol_acts[int(opt_pol[pol][ii][t_indx])][t_indx] += 1

    pol_acts /= 82

    for ii in range(len(pol_acts)):
        plt.plot(pol_acts[ii], label=ii)
    plt.legend(action_list, loc=5)
    plt.ylabel('Action Frequency')
    plt.xlabel('Episode')
    plt.title('Action frequency in the '+pol_list[pol]+' optimal sequences')
    plt.ylim((0, 1))
    plt.savefig('history_results/multi_Action_frequency_'+pol_list[pol] + file_suffix +'_opt.png')
    plt.clf()

for pol_indx,pol_name in enumerate(pol_list):
    plt.imshow(opt_pol[pol_indx], interpolation='bilinear', cmap=cm.Greys_r)
    plt.title(pol_name)
    plt.savefig('multi_'+pol_name+'_policy_map.png')
    plt.clf()
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
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pandas as pd
from scipy.interpolate import pchip_interpolate
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import scipy.io as sio
matplotlib.use('Agg')
MODE_LIST = ['min-spe', 'max-spe', 'min-rpe', 'max-rpe', 'min-rpe-min-spe', 'max-rpe-max-spe', 'max-rpe-min-spe', 'min-rpe-max-spe']
MODE_MAP = {
    'min-spe' : ['spe', None, 'red', 'MIN_SPE'],
    'max-spe' : ['spe', None, 'mediumseagreen', 'MAX_SPE'],
    'min-rpe' : ['rpe', None, 'royalblue', 'MIN_RPE'],
    'max-rpe' : ['rpe', None, 'plum', 'MAX_RPE'],
    'min-rpe-min-spe' : ['spe', 'rpe', 'tomato', 'MIN_RPE_MIN_SPE'],
    'max-rpe-max-spe' : ['spe', 'rpe', 'dodgerblue', 'MAX_RPE_MAX_SPE'],
    'max-rpe-min-spe' : ['spe', 'rpe', 'y', 'MAX_RPE_MIN_SPE'],
    'min-rpe-max-spe' : ['spe', 'rpe', 'mediumvioletred', 'MIN_RPE_MAX_SPE']
}
VAR_MAP_LIST = ['PMB','SPE', 'RPE','RWD']
file_suffix = '20200904'#'_2020_task_whole_averaging.pkl'
mode_idf = 2 #2-min 3-max
do_anova = False
var_idf = 0
max_mode = 8
TOTAL_EPISODES=100
TRIALS_PER_EPISODE=20
#FBA=np.load('FBA.npy')
#FBA_lb = np.zeros(82)
#for ii in range(82):
#    #if FBA[ii]<np.sort(FBA)[27]:
#    if FBA[ii]<0.6:
#        FBA_lb[ii]=0
#    else:
#        FBA_lb[ii] = 1
PMB_MAP=np.zeros((max_mode,82,TOTAL_EPISODES*TRIALS_PER_EPISODE))
SPE_MAP=np.zeros((max_mode,82,TOTAL_EPISODES*TRIALS_PER_EPISODE))
RPE_MAP=np.zeros((max_mode,82,TOTAL_EPISODES*TRIALS_PER_EPISODE))
RWD_MAP=np.zeros((max_mode,82,TOTAL_EPISODES*TRIALS_PER_EPISODE))
PMB_PLOT = np.zeros((max_mode,2))
SPE_PLOT = np.zeros((max_mode,2))
RPE_PLOT = np.zeros((max_mode,2))
RWD_PLOT = np.zeros((max_mode,2))
for mode_idf in range(max_mode):
    PMB_MAP[mode_idf] = np.load(
        'history_results/SUB{0:03d}_SHUFFLE_'.format(81) + MODE_LIST[mode_idf] + file_suffix + '_PMB.npy')
    SPE_MAP[mode_idf] = np.load(
        'history_results/SUB{0:03d}_SHUFFLE_'.format(81) + MODE_LIST[mode_idf] + file_suffix + '_SPE.npy')
    RPE_MAP[mode_idf] = np.load(
        'history_results/SUB{0:03d}_SHUFFLE_'.format(81) + MODE_LIST[mode_idf] + file_suffix + '_RPE.npy')
    RWD_MAP[mode_idf] = np.load(
        'history_results/SUB{0:03d}_SHUFFLE_'.format(81) + MODE_LIST[mode_idf] + file_suffix + '_RWD.npy')
    PMB_PLOT[mode_idf][0] = np.mean(PMB_MAP[mode_idf].flatten())
    SPE_PLOT[mode_idf][0] = np.mean(SPE_MAP[mode_idf].flatten())
    RPE_PLOT[mode_idf][0] = np.mean(RPE_MAP[mode_idf].flatten())
    RWD_PLOT[mode_idf][0] = np.mean(RWD_MAP[mode_idf].flatten())
    if mode_idf < 2: RWD_PLOT[mode_idf][0] = np.mean(RWD_MAP[mode_idf].flatten()) * 40
    PMB_PLOT[mode_idf][1] = np.std(PMB_MAP[mode_idf].flatten())/np.sqrt(len(PMB_MAP[mode_idf].flatten()))
    SPE_PLOT[mode_idf][1] = np.std(SPE_MAP[mode_idf].flatten())/np.sqrt(len(SPE_MAP[mode_idf].flatten()))
    RPE_PLOT[mode_idf][1] = np.std(RPE_MAP[mode_idf].flatten())/np.sqrt(len(RPE_MAP[mode_idf].flatten()))
    RWD_PLOT[mode_idf][1] = np.std(RWD_MAP[mode_idf].flatten())/np.sqrt(len(RWD_MAP[mode_idf].flatten()))    if mode_idf < 2: RWD_PLOT[mode_idf][1] = np.std(RWD_MAP[mode_idf].flatten())/np.sqrt(len(RWD_MAP[mode_idf].flatten())) * 40

plt.bar(range(max_mode),np.transpose(PMB_PLOT)[0])
plt.errorbar(range(max_mode), np.transpose(PMB_PLOT)[0],yerr=np.transpose(PMB_PLOT)[1])
plt.xticks(range(max_mode), MODE_LIST[0:max_mode])
plt.title('PMB')
plt.savefig('history_results/PMB_pol_82_plot'+file_suffix+'.png')
plt.clf()
plt.bar(range(max_mode), np.transpose(SPE_PLOT)[0])
plt.errorbar(range(max_mode), np.transpose(SPE_PLOT)[0], yerr=np.transpose(SPE_PLOT)[1])
plt.xticks(range(max_mode), MODE_LIST[0:max_mode])
plt.title('SPE')
plt.savefig('history_results/SPE_pol_82_plot'+file_suffix+'.png')
plt.clf()
plt.bar(range(max_mode), np.transpose(RPE_PLOT)[0])
plt.errorbar(range(max_mode), np.transpose(RPE_PLOT)[0], yerr=np.transpose(RPE_PLOT)[1])
plt.xticks(range(max_mode), MODE_LIST[0:max_mode])
plt.title('RPE')
plt.savefig('history_results/RPE_pol_82_plot'+file_suffix+'.png')
plt.clf()
plt.bar(range(max_mode), np.transpose(RWD_PLOT)[0])
plt.errorbar(range(max_mode), np.transpose(RWD_PLOT)[0], yerr=np.transpose(RWD_PLOT)[1])
plt.xticks(range(max_mode), MODE_LIST[0:max_mode])
plt.title('RWD')
plt.savefig('history_results/RWD_pol_82_plot'+file_suffix+'.png')
plt.clf()

PMB_SBJ = np.zeros((8,82))
SPE_SBJ = np.zeros((8,82))
RPE_SBJ = np.zeros((8,82))
RWD_SBJ = np.zeros((8,82))
for mode_idf in range(8):
    for sbj in range(82):
        PMB_SBJ[mode_idf][sbj] = np.mean(PMB_MAP[mode_idf][sbj])
        SPE_SBJ[mode_idf][sbj] = np.mean(SPE_MAP[mode_idf][sbj])
        RPE_SBJ[mode_idf][sbj] = np.mean(RPE_MAP[mode_idf][sbj])
        RWD_SBJ[mode_idf][sbj] = np.mean(RWD_MAP[mode_idf][sbj])

    plt.scatter(SPE_SBJ[mode_idf],PMB_SBJ[mode_idf])
    plt.xlabel('SPE')
    plt.ylabel('PMB')
    plt.title('SPE-PMB')
    plt.savefig('history_results/pol_82_plot_SPE_PMB' + MODE_LIST[mode_idf] + '_' + file_suffix + '.png')
    plt.clf()
    plt.scatter(RPE_SBJ[mode_idf], PMB_SBJ[mode_idf])
    plt.xlabel('RPE')
    plt.ylabel('PMB')
    plt.title('RPE-PMB')
    plt.savefig('history_results/pol_82_plot_RPE_PMB' + MODE_LIST[mode_idf] + '_' + file_suffix + '.png')
    plt.clf()
    plt.scatter(RWD_SBJ[mode_idf], PMB_SBJ[mode_idf])
    plt.xlabel('RWD')
    plt.ylabel('PMB')
    plt.title('RWD-PMB')
    plt.savefig('history_results/pol_82_plot_RWD_PMB' + MODE_LIST[mode_idf] + '_' + file_suffix + '.png')
    plt.clf()

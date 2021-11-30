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
MODE_LIST = ['min-rpe','max-rpe','min-spe','max-spe','min-rpe-min-spe','max-rpe-max-spe','min-rpe-max-spe','max-rpe-min-spe']
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
VAR_MAP_LIST = ['PMB','SPE', 'RPE','RWD','SCR']
folderpath = '20210827'
file_suffix = '_2020_20_trials_delta_control_highest'
max_sbj = 82
#file_suffix = '_2020_delta_trials_control_highest'
#file_suffix = '_2019_delta_trials_control_highest'
# if file_suffix == '_2020_delta_trials_control' or file_suffix == '_2020_20_trials_delta_control_highest':
#    MODE_LIST = ['min-rpe', 'max-rpe']
mode_idf = 2 #2-min 3-max
do_anova = False
var_idf = 0
CONTROL_resting = 99

FBA=np.load('FBA.npy')
FBA_lb = np.zeros(max_sbj)
for ii in range(max_sbj):
    #if FBA[ii]<np.sort(FBA)[27]:
    if FBA[ii]<0.6:
        FBA_lb[ii]=0
    else:
        FBA_lb[ii] = 1

pol_sbj_map = np.zeros((max_sbj,max_sbj))
for ii in range(max_sbj):
    for jj in range(max_sbj):
            pol_sbj_map[ii][jj]=ii

NUM_EPISODES = 101
TRIALS_PER_EPISODE = 20

for mode_idf in range(len(MODE_LIST)):
    for var_idf in range(len(VAR_MAP_LIST)):
        print(VAR_MAP_LIST[var_idf] +' result in the '+ MODE_LIST[mode_idf])
        VAR_MAP=np.zeros((max_sbj,max_sbj,TRIALS_PER_EPISODE*(NUM_EPISODES+CONTROL_resting)))
        VAR_statics = np.zeros((max_sbj,max_sbj))
        MatchNonMatch = np.zeros((max_sbj*max_sbj))
        for policy_sbj_indx in range(max_sbj):
            #MatchNonMatch[policy_sbj_indx*max_sbj+policy_sbj_indx]=1
            VAR_MAP[policy_sbj_indx]=np.load('history_results/' + folderpath + '/SUB{0:03d}_SHUFFLE_'.format(policy_sbj_indx) + MODE_LIST[mode_idf] + file_suffix + '_' + VAR_MAP_LIST[var_idf] + '.npy')
            for affected_sbj_indx in range(max_sbj):
                if FBA_lb[policy_sbj_indx] == FBA_lb[affected_sbj_indx] :
                    MatchNonMatch[policy_sbj_indx*max_sbj+affected_sbj_indx]=1

                VAR_statics[policy_sbj_indx][affected_sbj_indx]=sum(VAR_MAP[policy_sbj_indx][affected_sbj_indx])/len(VAR_MAP[policy_sbj_indx][affected_sbj_indx])

        if do_anova == True:
            df = pd.DataFrame(data=VAR_statics, index=range(max_sbj), columns=range(max_sbj))
            df2 = pd.melt(df.reset_index(), id_vars=['index'], value_vars=range(max_sbj))
            df2['match'] = MatchNonMatch
            formula = 'value ~ C(match)'
            lm = ols(formula, df2).fit()
            print(anova_lm(lm))


        ori_mean = np.zeros(TRIALS_PER_EPISODE)
        ori_sem = np.zeros(TRIALS_PER_EPISODE)
        shu_mean = np.zeros(TRIALS_PER_EPISODE)
        shu_sem = np.zeros(TRIALS_PER_EPISODE)
        RWD_MAP = np.zeros((max_sbj,max_sbj,TRIALS_PER_EPISODE))
        for policy_sbj_indx in range(max_sbj):
            temp = np.load('history_results/' + folderpath + '/SUB{0:03d}_SHUFFLE_'.format(policy_sbj_indx) + MODE_LIST[mode_idf] + file_suffix + '_' + VAR_MAP_LIST[var_idf] + '.npy')
            for affected_sbj_indx in range(max_sbj):
                for episode in range(TRIALS_PER_EPISODE):
                    RWD_MAP[policy_sbj_indx][affected_sbj_indx][episode]=np.mean(temp[affected_sbj_indx][TRIALS_PER_EPISODE*(episode+CONTROL_resting):TRIALS_PER_EPISODE*(episode+CONTROL_resting+1)])
        for trials in range(TRIALS_PER_EPISODE):
            tmp_ori = np.zeros(max_sbj*NUM_EPISODES)
            tmp_shu = np.zeros((max_sbj-1)*max_sbj*NUM_EPISODES)
            tmp_ori_indx = 0
            tmp_shu_indx = 0
            for policy_sbj_indx in range(max_sbj):
                for affected_sbj_indx in range(max_sbj):
                    opt_eps = np.argmax(RWD_MAP[policy_sbj_indx][affected_sbj_indx])
                    for eps in range(NUM_EPISODES):
                        if policy_sbj_indx==affected_sbj_indx:
                            tmp_ori[tmp_ori_indx]=VAR_MAP[policy_sbj_indx][affected_sbj_indx][trials+TRIALS_PER_EPISODE*eps+CONTROL_resting*TRIALS_PER_EPISODE]
                            tmp_ori_indx += 1
                        else:
                            tmp_shu[tmp_shu_indx]=VAR_MAP[policy_sbj_indx][affected_sbj_indx][trials+TRIALS_PER_EPISODE*eps+CONTROL_resting*TRIALS_PER_EPISODE]
                            tmp_shu_indx += 1
            ori_mean[trials] = np.mean(tmp_ori)
            ori_sem[trials] = np.std(tmp_ori) / np.sqrt(len(tmp_ori))
            shu_mean[trials] = np.mean(tmp_shu)
            shu_sem[trials] = np.std(tmp_shu) / np.sqrt(len(tmp_shu))
       # print(ori_mean)
        print(np.mean(ori_mean))
        TICK_NAME_NUM = np.linspace(1, TRIALS_PER_EPISODE + 1, TRIALS_PER_EPISODE+1)[:-1]
        TICK_NAME_STR = [str(x) for x in TICK_NAME_NUM]
        smooth_x = np.linspace(0, TICK_NAME_NUM[-1], 200)
        smooth_y = pchip_interpolate(TICK_NAME_NUM, ori_mean, smooth_x)
        ax = plt.gca()
        ax.plot(smooth_x, smooth_y, label='original', color='red')
        ax.set_xlabel('Game step in one policy')
        ax.set_ylabel('original ' + VAR_MAP_LIST[var_idf] + 'control')
        smooth_sem = pd.Series(data=pchip_interpolate(TICK_NAME_NUM, ori_sem, smooth_x))
        ax.fill_between(smooth_x, smooth_y - 1.96 * smooth_sem, smooth_y + 1.96 * smooth_sem, alpha=0.2,
                        color='red')
        plt.legend(bbox_to_anchor=(1.02, 1))


        smooth_x = np.linspace(0, TICK_NAME_NUM[-1], 200)
        smooth_y = pchip_interpolate(TICK_NAME_NUM, shu_mean, smooth_x)
        ax.plot(smooth_x, smooth_y, label='shuffled policy', color='black')
        ax.set_xlabel('Game step in one policy')
        ax.set_ylabel('shuffled ' + VAR_MAP_LIST[var_idf] + ' control')
        smooth_sem = pd.Series(data=pchip_interpolate(TICK_NAME_NUM, shu_sem, smooth_x))
        ax.fill_between(smooth_x, smooth_y - 1.96 * smooth_sem, smooth_y + 1.96 * smooth_sem, alpha=0.2, color='black')
        plt.legend(bbox_to_anchor=(1.02, 1))
        #if var_idf == 2: plt.ylim(6,10)
        plt.savefig('history_results/' + folderpath + '/' + 'Shuffled policy '+ VAR_MAP_LIST[var_idf] +' result in the '+ MODE_MAP[MODE_LIST[mode_idf]][3] + file_suffix +'.png')
        plt.clf()

        plt.scatter(pol_sbj_map.flatten(),np.median(RWD_MAP,axis=2).flatten())
        plt.savefig('history_results/' + folderpath + '/' + 'Shuffled policy-specific '+ VAR_MAP_LIST[var_idf] +' result in the '+ MODE_MAP[MODE_LIST[mode_idf]][3] + file_suffix +'.png')
        plt.clf()


        sio.savemat('history_results/' + folderpath + '/' + VAR_MAP_LIST[var_idf] +' result in the '+ MODE_MAP[MODE_LIST[mode_idf]][3] +"data" + file_suffix + ".mat", {'data': VAR_MAP})




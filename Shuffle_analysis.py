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
file_suffix = '_20210601_2021_delta_control'
#file_suffix = '_20210520_2020_delta_control'
#file_suffix = '_20210616_2019_delta_control'
if file_suffix == '_20210520_2020_delta_control':
    MODE_LIST = ['min-rpe', 'max-rpe']
mode_idf = 2 #2-min 3-max
do_anova = False
var_idf = 0
CONTROL_resting = 99
FBA=np.load('FBA.npy')
FBA_lb = np.zeros(82)
for ii in range(82):
    #if FBA[ii]<np.sort(FBA)[27]:
    if FBA[ii]<0.6:
        FBA_lb[ii]=0
    else:
        FBA_lb[ii] = 1

pol_sbj_map = np.zeros((82,82))
for ii in range(82):
    for jj in range(82):
            pol_sbj_map[ii][jj]=ii

NUM_EPISODES = 101
TRIALS_PER_EPISODE = 40

for mode_idf in range(len(MODE_LIST)):
    for var_idf in range(len(VAR_MAP_LIST)):
        print(VAR_MAP_LIST[var_idf] +' result in the '+ MODE_MAP[MODE_LIST[mode_idf]][3])
        VAR_MAP=np.zeros((82,82,TRIALS_PER_EPISODE*(NUM_EPISODES+CONTROL_resting)))
        VAR_statics = np.zeros((82,82))
        MatchNonMatch = np.zeros((82*82))
        for policy_sbj_indx in range(82):
            #MatchNonMatch[policy_sbj_indx*82+policy_sbj_indx]=1
            VAR_MAP[policy_sbj_indx]=np.load('history_results/SUB{0:03d}_SHUFFLE_'.format(policy_sbj_indx) + MODE_LIST[mode_idf] + file_suffix + '_' + VAR_MAP_LIST[var_idf] + '.npy')
            for affected_sbj_indx in range(82):
                if FBA_lb[policy_sbj_indx] == FBA_lb[affected_sbj_indx] :
                    MatchNonMatch[policy_sbj_indx*82+affected_sbj_indx]=1

                VAR_statics[policy_sbj_indx][affected_sbj_indx]=sum(VAR_MAP[policy_sbj_indx][affected_sbj_indx])/len(VAR_MAP[policy_sbj_indx][affected_sbj_indx])

        if do_anova == True:
            df = pd.DataFrame(data=VAR_statics, index=range(82), columns=range(82))
            df2 = pd.melt(df.reset_index(), id_vars=['index'], value_vars=range(82))
            df2['match'] = MatchNonMatch
            formula = 'value ~ C(match)'
            lm = ols(formula, df2).fit()
            print(anova_lm(lm))


        ori_mean = np.zeros(TRIALS_PER_EPISODE)
        ori_sem = np.zeros(TRIALS_PER_EPISODE)
        shu_mean = np.zeros(TRIALS_PER_EPISODE)
        shu_sem = np.zeros(TRIALS_PER_EPISODE)
        RWD_MAP = np.zeros((82,82,TRIALS_PER_EPISODE))
        for policy_sbj_indx in range(82):
            temp = np.load('history_results/SUB{0:03d}_SHUFFLE_'.format(policy_sbj_indx) + MODE_LIST[mode_idf] + file_suffix + '_' + VAR_MAP_LIST[var_idf] + '.npy')
            for affected_sbj_indx in range(82):
                for episode in range(TRIALS_PER_EPISODE):
                    RWD_MAP[policy_sbj_indx][affected_sbj_indx][episode]=np.mean(temp[affected_sbj_indx][TRIALS_PER_EPISODE*(episode+CONTROL_resting):TRIALS_PER_EPISODE*(episode+CONTROL_resting+1)])
        for trials in range(TRIALS_PER_EPISODE):
            tmp_ori = np.zeros(82*NUM_EPISODES)
            tmp_shu = np.zeros(81*82*NUM_EPISODES)
            tmp_ori_indx = 0
            tmp_shu_indx = 0
            for policy_sbj_indx in range(82):
                for affected_sbj_indx in range(82):
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
        plt.savefig('Shuffled policy '+ VAR_MAP_LIST[var_idf] +' result in the '+ MODE_MAP[MODE_LIST[mode_idf]][3] + file_suffix +'.png')
        plt.clf()

        plt.scatter(pol_sbj_map.flatten(),np.median(RWD_MAP,axis=2).flatten())
        plt.savefig('Shuffled policy-specific '+ VAR_MAP_LIST[var_idf] +' result in the '+ MODE_MAP[MODE_LIST[mode_idf]][3] + file_suffix +'.png')
        plt.clf()


        sio.savemat(VAR_MAP_LIST[var_idf] +' result in the '+ MODE_MAP[MODE_LIST[mode_idf]][3] +"data" + file_suffix + ".mat", {'data': VAR_MAP})




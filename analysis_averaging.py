import pickle
import analysis
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
MODE_LIST = ['min-spe', 'max-spe', 'min-rpe', 'max-rpe', 'min-rpe-min-spe', 'max-rpe-max-spe', 'max-rpe-min-spe', 'min-rpe-max-spe']
VAR_LIST = ['mb_rel','mf_rel']#['rpe','spe','ctrl_reward','p_mb']
file_suffix = '_2019'
path = 'history_results'
mode_idf = 0

for mode_idf in range(8):
    file_name = path+'/Analysis-Object-'+MODE_LIST[mode_idf]+'-82'+file_suffix+'_setting_whole_averaging.pkl'

    with open(file_name,'rb') as f:
        data = pickle.load(f)

    var_idf = 2
    for var_idf in range(len(VAR_LIST)):
        VAR_MAP = data.data[MODE_LIST[mode_idf]][0][VAR_LIST[var_idf]]
        VAR_MEAN = np.zeros(1000)
        VAR_STD = np.zeros(1000)

        for episodes in range(1000):
            VAR_MEAN[episodes] = np.mean(VAR_MAP[episodes])
            VAR_STD[episodes] = np.std(VAR_MAP[episodes])/np.sqrt(82)

        plt.errorbar(range(1000),VAR_MEAN,yerr = VAR_STD)
        plt.ylabel(VAR_LIST[var_idf])
        plt.xlabel('episodes')
        plt.title(MODE_LIST[mode_idf]+'_'+VAR_LIST[var_idf])
        plt.savefig(path+'/'+MODE_LIST[mode_idf]+'_'+VAR_LIST[var_idf]+file_suffix+'_setting_whole_averaging.png')
        plt.clf()

    for var_idf in range(len(VAR_LIST)):
        VAR_MAP = data.detail[MODE_LIST[mode_idf]][0][VAR_LIST[var_idf]]
        VAR_MEAN = np.zeros(1000*20)
        VAR_STD = np.zeros(1000*20)

        for episodes in range(1000*20):
            VAR_MEAN[episodes] = np.mean(VAR_MAP[episodes])
            VAR_STD[episodes] = np.std(VAR_MAP[episodes])/np.sqrt(82)

        plt.errorbar(range(1000*20), VAR_MEAN, yerr=VAR_STD)
        plt.ylabel(VAR_LIST[var_idf])
        plt.xlabel('episodes')
        plt.title(MODE_LIST[mode_idf]+'_'+VAR_LIST[var_idf])
        plt.savefig(path+'/'+MODE_LIST[mode_idf]+'_'+VAR_LIST[var_idf]+file_suffix+'_setting_whole_averaging_trials.png')
        plt.clf()
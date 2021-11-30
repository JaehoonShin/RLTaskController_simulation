import torch
import getopt
import sys
import csv
import os
import simulation as sim
import math
from random import randint
from analysis import gData, MODE_MAP
from tqdm import tqdm
from numpy.random import choice
from torch.autograd import Variable
import pandas as pd
import analysis

import numpy as np
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt
from analysis import save_plt_figure

from mdp import MDP
from sarsa import SARSA
from forward import FORWARD
from arbitrator import BayesRelEstimator, AssocRelEstimator, Arbitrator
import dill as pickle # see https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
import random

from mdp import MDP
from ddqn import DoubleDQN
from sarsa import SARSA
from forward import FORWARD
from arbitrator import BayesRelEstimator, AssocRelEstimator, Arbitrator
from analysis import gData, RESULTS_FOLDER, COLUMNS, DETAIL_COLUMNS
from common import makedir
import analysis
from math import ceil
import scipy.io as sio
#from main import MODE_LIST
# preset constants
MDP_STAGES            = 2
TOTAL_EPISODES        = 200
TRIALS_PER_EPISODE    = 80
SPE_LOW_THRESHOLD     = 0.3#0.3
SPE_HIGH_THRESHOLD    = 0.45#0.5
RPE_LOW_THRESHOLD     = 4
RPE_HIGH_THRESHOLD    = 9 #10
MF_REL_HIGH_THRESHOLD = 0.8
MF_REL_LOW_THRESHOLD  = 0.5
MB_REL_HIGH_THRESHOLD = 0.7
MB_REL_LOW_THRESHOLD  = 0.3
CONTROL_REWARD        = 1
CONTROL_REWARD_BIAS   = 0
INIT_CTRL_INPUT       = [10, 0.5]
DEFAULT_CONTROL_MODE  = 'max-spe'
CONTROL_MODE          = DEFAULT_CONTROL_MODE
CTRL_AGENTS_ENABLED   = True
RPE_DISCOUNT_FACTOR   = 0.003
ACTION_PERIOD         = 3
STATIC_CONTROL_AGENT  = False
ENABLE_PLOT           = True
DISABLE_C_EXTENSION   = False
LEGACY_MODE           = False
MORE_CONTROL_INPUT    = True
SAVE_CTRL_RL          = False
PMB_CONTROL = False
TASK_TYPE = 2021
MF_ONLY = False
MB_ONLY = False
Reproduce_BHV = False
saved_policy_path = ''
Session_block = False
mode202010 = False
DECAY_RATE = 0.5
turn_off_tqdm = False
CONTROL_resting = 99 #Intial duration for CONTROL agent resting
max_sbj = 82
file_suffix = '_20210601_2021_delta_control'
file_suffix = '_20210520_2020_delta_control'
file_suffix = '_20210616_2019_delta_control'
RESET = False
SAVE_LOG_Q_VALUE = False
MIXED_RANDOM_MODE = False
RANDOM_MODE_LIST = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe']


error_reward_map = {
    # x should be a 4-tuple: rpe, spe, mf_rel, mb_rel
    # x should be a 5-tuple: rpe, spe, mf_rel, mb_rel, PMB - updated
    'min-rpe' : (lambda x: x[0] < RPE_LOW_THRESHOLD),
    'max-rpe' : (lambda x: x[0] > RPE_HIGH_THRESHOLD),
    'min-spe' : (lambda x: x[1] < SPE_LOW_THRESHOLD),
    'max-spe' : (lambda x: x[1] > SPE_HIGH_THRESHOLD),
    'min-mf-rel' : (lambda x: x[2] < MF_REL_LOW_THRESHOLD),
    'max-mf-rel' : (lambda x: x[2] > MF_REL_HIGH_THRESHOLD),
    'min-mb-rel' : (lambda x: x[3] < MB_REL_LOW_THRESHOLD),
    'max-mb-rel' : (lambda x: x[3] > MB_REL_HIGH_THRESHOLD),
    'min-rpe-min-spe' : lambda x: error_reward_map['min-rpe'](x) and error_reward_map['min-spe'](x),
    'max-rpe-max-spe' : lambda x: error_reward_map['max-rpe'](x) and error_reward_map['max-spe'](x),
    'min-rpe-max-spe' : lambda x: error_reward_map['min-rpe'](x) and error_reward_map['max-spe'](x),
    'max-rpe-min-spe' : lambda x: error_reward_map['max-rpe'](x) and error_reward_map['min-spe'](x),
    'random' : lambda x: 0
}


def create_lst(x):
    return [x] * TRIALS_PER_EPISODE

static_action_map = {
    'min-rpe' : create_lst(0),
    'max-rpe' : create_lst(3),
    'min-spe' : create_lst(0),
    'max-spe' : create_lst(1),
    'min-rpe-min-spe' : create_lst(0),
    'max-rpe-max-spe' : create_lst(3),
    'min-rpe-max-spe' : create_lst(1),
    'max-rpe-min-spe' : create_lst(2)
}

def error_to_reward(error, PMB=0 , mode=DEFAULT_CONTROL_MODE, bias=CONTROL_REWARD_BIAS):
    """Compute reward for the task controller. Based on the input scenario (mode), the reward function is determined from the error_reward_map dict.
        Args:
            error (float list): list with player agent's internal states. Current setting: RPE/SPE/MF-Rel/MB-Rel/PMB
            For the error argument, please check the error_reward_map
            PMB (float): PMB value of player agents. Currently duplicated with error argument.
            mode (string): type of scenario

        Return:
            action (int): action to take by human agent
        """

    if mode == 'min-rpe':
        reward = (40 - error[0]) * 3
    elif mode == 'max-rpe':
        reward = error[0] * 10
    elif mode == 'min-spe':
        reward = (1 - error[1])*150
    elif mode == 'max-spe':
        reward = error[1]*200
    elif mode == 'min-rpe-min-spe':
        reward = ((40 - error[0]) * 3 + (1 - error[1]) * 150 ) /2
    elif mode == 'max-rpe-max-spe':
        reward = ((error[0]) * 10 + (error[1]) * 100) /2
    elif mode == 'min-rpe-max-spe':
        reward = ((40 - error[0]) * 3 + (error[1]) * 200) /2
    elif mode == 'max-rpe-min-spe':
        reward = ((error[0]) * 10 + (1 - error[1]) * 150) /2
    elif mode == 'random' :
        reward = 0

    if PMB_CONTROL:
        reward = reward-60*PMB

    return reward  # -60*PMB
#    if cmp_func(error):
#        if CONTROL_REWARD < 0.5 :
#            return CONTROL_REWARD + bias
#        else :
#            return CONTROL_REWARD * ((2-PMB*2)**0.5) + bias
#            #return CONTROL_REWARD*(2-2*PMB) + bias
#    else:
#        return bias
pol_list = ['max-spe','min-spe','max-rpe','min-rpe','min-rpe-min-spe','max-rpe-max-spe','min-rpe-max-spe','max-rpe-min-spe']
if file_suffix == '_20210520_2020_delta_control':
    pol_list = ['min-rpe', 'max-rpe']
for CONTROL_MODE in pol_list:
    opt_pols=[]
    for policy_sbj_indx in range(82):
        pol = pol_list.index(CONTROL_MODE)
        params = []
        f = open('regdata.csv')
        data = f.readlines()
        for sbj in data:
            params.append(sbj.split(',')[:-1])
        f.close()
        for ii in range(len(params)):
            for jj in range(len(params[ii])):
                params[ii][jj] = float(params[ii][jj])
        pol_filename = 'history_results/Analysis-Object-'+CONTROL_MODE+'-{0:02d}'.format(policy_sbj_indx)+file_suffix+'.pkl'
        with open(pol_filename,'rb') as f:
            pol_sbj_data = pickle.load(f)
        NUM_EPISODES, NUM_FULL_FEATS_DATA = pol_sbj_data.data[pol_list[pol]][0].shape
        NUM_FULL_TRIALS, NUM_FULL_FEATS_DETAIL = pol_sbj_data.detail[pol_list[pol]][0].shape
        TRIALS_PER_EPISODE = ceil(NUM_FULL_TRIALS / NUM_EPISODES)
        PMB_shuffle=np.zeros((max_sbj,TOTAL_EPISODES*TRIALS_PER_EPISODE))
        RPE_shuffle=np.zeros((max_sbj,TOTAL_EPISODES*TRIALS_PER_EPISODE))
        SPE_shuffle = np.zeros((max_sbj, TOTAL_EPISODES*TRIALS_PER_EPISODE))
        Reward_shuffle=np.zeros((max_sbj,TOTAL_EPISODES*TRIALS_PER_EPISODE))

        opt_index = pol_sbj_data.data[pol_list[pol]][0]['ctrl_reward'].loc[
                              0.2 * len(pol_sbj_data.data[pol_list[pol]][0]):].idxmax()
        opt_pol = pol_sbj_data.detail[pol_list[pol]][0]['action'].loc[
                            opt_index * TRIALS_PER_EPISODE - TRIALS_PER_EPISODE:opt_index * TRIALS_PER_EPISODE - 1]
        opt_pols.append(opt_pol)
    sio.savemat('Policy result in the '+ CONTROL_MODE +"data" + file_suffix + ".mat", {'data': opt_pols})
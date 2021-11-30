""" Shuangyi Tong <s9tong@edu.uwaterloo.ca>
    Sept 17, 2018
"""
import torch
import numpy as np
import pandas as pd
import dill as pickle  # see https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
import random
import analysis

from tqdm import tqdm
from mdp import MDP
from ddqn import DoubleDQN
from sarsa import SARSA
from forward import FORWARD
from arbitrator import BayesRelEstimator, AssocRelEstimator, Arbitrator
from analysis import gData, RESULTS_FOLDER, COLUMNS, DETAIL_COLUMNS
from common import makedir
import getopt
import sys
import csv
import os
import torch

from analysis import gData, MODE_MAP
from tqdm import tqdm
from numpy.random import choice
# from training_layer2 import Net
from torch.autograd import Variable
# preset constants
MDP_STAGES = 2
TOTAL_EPISODES = 1000
TRIALS_PER_EPISODE = 20
SPE_LOW_THRESHOLD = 0.3  # 0.3
SPE_HIGH_THRESHOLD = 0.45  # 0.5
RPE_LOW_THRESHOLD = 4
RPE_HIGH_THRESHOLD = 9  # 10
MF_REL_HIGH_THRESHOLD = 0.8
MF_REL_LOW_THRESHOLD = 0.5
MB_REL_HIGH_THRESHOLD = 0.7
MB_REL_LOW_THRESHOLD = 0.3
CONTROL_REWARD = 1
CONTROL_REWARD_BIAS = 0
INIT_CTRL_INPUT = [10, 0.5]
DEFAULT_CONTROL_MODE = 'max-spe'
CONTROL_MODE = DEFAULT_CONTROL_MODE
CTRL_AGENTS_ENABLED = True
RPE_DISCOUNT_FACTOR = 0.003
ACTION_PERIOD = 3
STATIC_CONTROL_AGENT = False
ENABLE_PLOT = False
DISABLE_C_EXTENSION = True
LEGACY_MODE = False
MORE_CONTROL_INPUT = True
SAVE_CTRL_RL = False
TASK_TYPE = 2019
MF_ONLY = False
MB_ONLY = False

RESET = False
SAVE_LOG_Q_VALUE = False
MIXED_RANDOM_MODE = False
RANDOM_MODE_LIST = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe']

LOAD_PARAM_FILE = True
NUM_PARAMETER_SET = 82
sbj_size = NUM_PARAMETER_SET
ALL_MODE = False
ANALYSIS_OBJ = None
TO_EXCEL = None
SCENARIO = None
CROSS_MODE_PLOT = False
CROSS_COMPARE = False
OPPOSITE_CROSS_COMPARE = False
OPPOSITE_NN_CROSS_COMPARE = False
FAIR_OPPOSITE_NN_CROSS_COMPARE = False
CROSS_COMPARE_MOD = 'min-spe'
SUBJECT_A = 10  # Low MB->MF trans rate
SUBJECT_B = 17  # High MB->MF trans rate
# PARAMETER_FILE    = '82nd_subj.csv'
PARAMETER_FILE = 'regdata.csv'
TO_EXCEL_LOG_Q_VALUE = False
TO_EXCEL_RANDOM_MODE_SEQUENCE = False
TO_EXCEL_OPTIMAL_SEQUENCE = False
FILE_SUFFIX = ''
SCENARIO_MODE_MAP = {
    'boost': ['min-spe', 'min-rpe'],
    'inhibit': ['min-spe', 'max-spe'],
    'cor': ['min-rpe-min-spe', 'max-rpe-max-spe'],
    'sep': ['min-rpe-max-spe', 'max-rpe-min-spe']
}
ORIGINAL_MODE_MAP = ['min-spe', 'max-spe', 'min-rpe', 'max-rpe', 'min-rpe-min-spe', 'max-rpe-max-spe',
                     'max-rpe-min-spe', 'min-rpe-max-spe']
OPPOSITE_MODE_MAP = ['max-spe', 'min-spe', 'max-rpe', 'min-rpe', 'max-rpe-max-spe', 'min-rpe-min-spe',
                     'min-rpe-max-spe', 'max-rpe-min-spe']

MODE_LIST = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'max-rpe-max-spe', 'min-rpe-min-spe', 'max-rpe-min-spe',
             'min-rpe-max-spe', 'min-rpe-PMB', 'max-rpe-PMB', 'min-rpe-max-spe-PMB']
MODE_LIST_INPUT = [[-1, 0], [1, 0], [0, -1], [0, 1], [1, 1], [-1, -1], [1, -1], [-1, 1], [-1, 0], [1, 0]]

error_reward_map = {
    # x should be a 4-tuple: rpe, spe, mf_rel, mb_rel
    # x should be a 5-tuple: rpe, spe, mf_rel, mb_rel, PMB - updated
    'min-rpe': (lambda x: x[0] < RPE_LOW_THRESHOLD),
    'max-rpe': (lambda x: x[0] > RPE_HIGH_THRESHOLD),
    'min-spe': (lambda x: x[1] < SPE_LOW_THRESHOLD),
    'max-spe': (lambda x: x[1] > SPE_HIGH_THRESHOLD),
    'min-mf-rel': (lambda x: x[2] < MF_REL_LOW_THRESHOLD),
    'max-mf-rel': (lambda x: x[2] > MF_REL_HIGH_THRESHOLD),
    'min-mb-rel': (lambda x: x[3] < MB_REL_LOW_THRESHOLD),
    'max-mb-rel': (lambda x: x[3] > MB_REL_HIGH_THRESHOLD),
    'min-rpe-min-spe': lambda x: error_reward_map['min-rpe'](x) and error_reward_map['min-spe'](x),
    'max-rpe-max-spe': lambda x: error_reward_map['max-rpe'](x) and error_reward_map['max-spe'](x),
    'min-rpe-max-spe': lambda x: error_reward_map['min-rpe'](x) and error_reward_map['max-spe'](x),
    'max-rpe-min-spe': lambda x: error_reward_map['max-rpe'](x) and error_reward_map['min-spe'](x)
}


def create_lst(x):
    return [x] * TRIALS_PER_EPISODE


static_action_map = {
    'min-rpe': create_lst(0),
    'max-rpe': create_lst(3),
    'min-spe': create_lst(0),
    'max-spe': create_lst(1),
    'min-rpe-min-spe': create_lst(0),
    'max-rpe-max-spe': create_lst(3),
    'min-rpe-max-spe': create_lst(1),
    'max-rpe-min-spe': create_lst(2)
}


def error_to_reward(error, PMB=0, mode=DEFAULT_CONTROL_MODE, bias=CONTROL_REWARD_BIAS):
    """Compute reward for the task controller. Based on the input scenario (mode), the reward function is determined from the error_reward_map dict.
        Args:
            error (float list): list with player agent's internal states. Current setting: RPE/SPE/MF-Rel/MB-Rel/PMB
            For the error argument, please check the error_reward_map
            PMB (float): PMB value of player agents. Currently duplicated with error argument.
            mode (string): type of scenario

        Return:
            action (int): action to take by human agent
        """
#    if TASK_TYPE == 2019:
#        try:
#            cmp_func = error_reward_map[mode]
#        except KeyError:
#            print("Warning: control mode {0} not found, use default mode {1}".format(mode, DEFAULT_CONTROL_MODE))
#            cmp_func = error_reward_map[DEFAULT_CONTROL_MODE]
#
#        return cmp_func(error)
#
#    elif TASK_TYPE == 2020:
    if mode == 'min-rpe': # 35
        reward = 40-error[0]
    elif mode == 'max-rpe': # 9~10
        reward = error[0]
    elif mode == 'min-spe': # 0.8
        reward = 1-error[1]
    elif mode == 'max-spe': # 0.5
        reward = error[1]
    elif mode == 'min-rpe-min-spe':
        reward = (40-error[0]) + (1-error[1])*(35/0.8)
    elif mode == 'max-rpe-max-spe':
        reward = (error[0]) + (error[1])*(10/0.5)
    elif mode == 'min-rpe-max-spe':
        reward = (40-error[0]) + (error[1])*(35/0.5)
    elif mode == 'max-rpe-min-spe':
        reward = (error[0]) + (1-error[1])*(10/0.8)

    return reward#-60*PMB



#    if cmp_func(error):
#        if CONTROL_REWARD < 0.5 :
#            return CONTROL_REWARD + bias
#        else :
#            return CONTROL_REWARD * ((2-PMB*2)**0.5) + bias
#            #return CONTROL_REWARD*(2-2*PMB) + bias
#    else:
#        return bias

def compute_human_action(arbitrator, human_obs, model_free, model_based):
    """Compute human action by compute model-free and model-based separately
    then integrate the result by the arbitrator

    Args:
        arbitrator (any callable): arbitrator object
        human_obs (any): valid in human observation space
        model_free (any callable): model-free agent object
        model_based (any callable): model-based agent object

    Return:
        action (int): action to take by human agent
    """
    return arbitrator.action(model_free.get_Q_values(human_obs),
                             model_based.get_Q_values(human_obs))


def simulation(PARAMETER_SET='DEFAULT', return_res=False):
    """Simulate single (player agent / task controller) pair with given number of episodes. The parameters for the player agent is fixed during simulation.
        However, the task controller's parameter will be changed (actually, optimized) after each trial.
        During one trial of game, the internal variable of player gents will be collected.
        Then, collected variables will be averaged and turned into the rewards for the task controller.
        The task controller (in here, the Double DQN) will optimize its parameter by reinforcement learning.
        Various variables generated during simulation will be collected and exported to the main.py
        Here's important variable terms :
            episodes : The biggest scope. If 'Reset' variable is activated (Reset = 1), the player agent will be always resumed to the default state (clear Q-value map).
                        Cumulative variables (ex: cum_rpe) is set to 0 at the start of every episode.
                        For the episode 1 ~ 100, the task controller will generate the random actions.
                        This design aims the encouragement of exploration of task controller.
                        Also, the human agents will designed to generate random action during episode 1~99.
                        However, it is encoded in the arbitrator.py, not in the current code.
                        The game environment (task structure) will be resumed at the start of every episode, when episode > 100.

            trials : 20 trials = 1 episode in the current setting. 1 trial means one game play, so every trial starts with moving agnet to the initial point of the game.
                    At the start of trial, the task controller changes the task structure, and the storages for the values (ex: t_rpe) are set to 0.
                    At the end of each trial, the task controller will update its parameters.

            game_terminate : The smallest scope. It is boolean value.
                             2 game steps = 1 trial (game trial).
                             At the start of game step, the task player observes the current goal setting and the current state.
                             The player agent choose action with observation, and the next state for that action will be shown.
                             Player agent updates its internal states at the end of game step.

        Args:
            threshold ~ rl_learning_rate : parameters for the player agents
            performance (float) : Fitness of player agent's parameter (not in this simulation)
                                 Currently non-used
            PARAMETER_SET (string) : address for the parameters.
        Return:

    """
    print(PARAMETER_FILE)
    with open(PARAMETER_FILE) as f:
        csv_parser = csv.reader(f)
        param_list = []
        for row in csv_parser:
            param_list.append(tuple(map(float, row[:-1])))
    CHANGE_MODE_TERM = int(TOTAL_EPISODES / len(RANDOM_MODE_LIST))

    if return_res:
        res_data_df = pd.DataFrame(columns=COLUMNS)
        res_detail_df = pd.DataFrame(columns=DETAIL_COLUMNS)
    env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE, task_type=TASK_TYPE)
    if MIXED_RANDOM_MODE:
        random_mode_list = np.random.choice(RANDOM_MODE_LIST, 4, replace=False)  # order among 4 mode is random
        print('Random Mode Sequence : %s' % random_mode_list)

    # if it is mixed random mode, 'ddpn_loaded' from torch.save(model, filepath) is used instead of 'ddpn'
    ddqn = DoubleDQN(env.observation_space[MDP.CONTROL_AGENT_INDEX],
                     env.action_space[MDP.CONTROL_AGENT_INDEX],
                     torch.cuda.is_available())  # use DDQN for control agent
    gData.new_simulation()
#    for sbj in range(sbj_size):
#        print(param_list[sbj])
#        [threshold, estimator_learning_rate, amp_mb_to_mf, amp_mf_to_mb, temperature, rl_learning_rate, performance] = \
#        gData.add_human_data([amp_mf_to_mb , amp_mb_to_mf, rl_learning_rate, estimator_learning_rate, threshold, temperature, performance])

    control_obs_extra = INIT_CTRL_INPUT


    human_action_list_t = []
    Task_structure = []
    Q_value_forward_t = []
    Q_value_sarsa_t = []
    Q_value_arb_t = []
    Transition_t = []
    prev_cum_reward = 0
    env_list = [None] * sbj_size
    sarsa_list = [None] * sbj_size
    forward_list = [None] * sbj_size
    arb_list = [None] * sbj_size
    for episode in tqdm(range(TOTAL_EPISODES)):
        cum_d_p_mb = [0]*sbj_size
        cum_p_mb = [0]*sbj_size
        cum_mf_rel = [0]*sbj_size
        cum_mb_rel = [0]*sbj_size
        cum_rpe = [0]*sbj_size
        cum_spe = [0]*sbj_size
        cum_reward = [0]*sbj_size
        cum_score = [0]*sbj_size
        cum_ctrl_act = [[0]*sbj_size]*env.NUM_CONTROL_ACTION
        human_action_list_episode = []
        for trial in range(TRIALS_PER_EPISODE):
            t_d_p_mb = [0] * sbj_size
            t_p_mb = [0] * sbj_size
            t_mf_rel = [0] * sbj_size
            t_mb_rel = [0] * sbj_size
            t_rpe = [0] * sbj_size
            t_spe = [0] * sbj_size
            t_reward = [0] * sbj_size
            t_score = [0] * sbj_size
            for sbj in range(82):
                if episode == 0 and trial == 0:
                    threshold, estimator_learning_rate, amp_mb_to_mf, amp_mf_to_mb, temperature, rl_learning_rate, performance = param_list[sbj]
                    # reinitialize human agent every episode
                    sarsa = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX],
                                  learning_rate=rl_learning_rate)  # SARSA model-free learner
                    forward = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                                      env.action_space[MDP.HUMAN_AGENT_INDEX],
                                      env.state_reward_func, env.output_states_offset, env.reward_map_func,
                                      learning_rate=rl_learning_rate,
                                      disable_cforward=DISABLE_C_EXTENSION)  # forward model-based learner
                    arb = Arbitrator(AssocRelEstimator(estimator_learning_rate, env.max_rpe),
                                     BayesRelEstimator(thereshold=threshold),
                                     amp_mb_to_mf=amp_mb_to_mf, amp_mf_to_mb=amp_mf_to_mb, MB_ONLY = MB_ONLY, MF_ONLY= MF_ONLY)
                    # register in the communication controller
                    env.agent_comm_controller.register('model-based', forward)
                else:
                    sarsa = sarsa_list[sbj]
                    forward = forward_list[sbj]
                    arb = arb_list[sbj]

                if episode > 99 and trial == 0:
                    env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE, task_type=TASK_TYPE)
                elif episode == 0 and trial == 0:
                    env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE, task_type=TASK_TYPE)
                else:
                    env = env_list[sbj]

                if trial == 0:
                    env.reward_map = env.reward_map_copy.copy()
                    env.output_states = env.output_states_copy.copy()
                    arb.episode_number = episode

                human_obs, control_obs_frag = env.reset()
                control_obs = np.append(control_obs_frag, control_obs_extra)

                game_terminate = False
                """control agent choose action"""
                if STATIC_CONTROL_AGENT:
                    control_action = static_action_map[CONTROL_MODE][trial]
                else:
                    control_action = ddqn.action(control_obs)
                cum_ctrl_act[control_action][sbj] += 1
                if env.task_type == 2019:
                    if control_action == 3:
                        if env.is_flexible == 1:
                            arb.p_mb = 0.8
                            arb.p_mf = 0.2
                        else:
                            arb.p_mb = 0.2
                            arb.p_mf = 0.8

                    """control act on environment"""
                    if CTRL_AGENTS_ENABLED:  # and (trial % ACTION_PERIOD is 0 or not STATIC_CONTROL_AGENT): ## why trial % ACTION_PERIOD ??
                        _, _, _, _ = env.step([MDP.CONTROL_AGENT_INDEX, control_action])

                forward.bwd_update(env.bwd_idf, env)

                while not game_terminate:
                    """human choose action"""
                    human_action = compute_human_action(arb, human_obs, sarsa, forward)
                    # print("human action : ", human_action)
                    if SAVE_LOG_Q_VALUE:
                        human_action_list_episode.append(human_action)

                    """human act on environment"""
                    next_human_obs, human_reward, game_terminate, next_control_obs_frag \
                        = env.step((MDP.HUMAN_AGENT_INDEX, human_action))

                    """update human agent"""
                    spe = forward.optimize(human_obs, human_action, next_human_obs)
                    next_human_action = compute_human_action(arb, next_human_obs, sarsa,
                                                             forward)  # required by models like SARSA
                    if env.is_flexible == 1:  # flexible goal condition
                        rpe = sarsa.optimize(human_reward, human_action, next_human_action, human_obs, next_human_obs)
                    else:  # specific goal condition human_reward should be normalized to sarsa
                        if human_reward > 0:  # if reward is 10, 20, 40
                            rpe = sarsa.optimize(40, human_action, next_human_action, human_obs, next_human_obs)
                        else:
                            rpe = sarsa.optimize(0, human_action, next_human_action, human_obs, next_human_obs)

                    mf_rel, mb_rel, p_mb, d_p_mb = arb.add_pe(rpe, spe)
                    t_d_p_mb[sbj] += d_p_mb
                    t_p_mb[sbj] += p_mb
                    t_mf_rel[sbj] += mf_rel
                    t_mb_rel[sbj] += mb_rel
                    t_rpe[sbj] += abs(rpe)
                    t_spe[sbj] += spe
                    t_score[sbj] += human_reward  # if not the terminal state, human_reward is 0, so simply add here is fine
                    """iterators update"""
                    human_obs = next_human_obs

                # calculation after one trial
                d_p_mb, p_mb, mf_rel, mb_rel, rpe, spe = list(map(lambda x: x / MDP_STAGES, [
                    t_d_p_mb[sbj], t_p_mb[sbj], t_mf_rel[sbj], t_mb_rel[sbj], t_rpe[sbj], t_spe[sbj]]))  # map to average value
                t_d_p_mb[sbj] = d_p_mb
                t_p_mb[sbj] = p_mb
                t_mf_rel[sbj] = mf_rel
                t_mb_rel[sbj] = mb_rel
                t_rpe[sbj] = rpe
                t_spe[sbj] = spe
                t_reward[sbj] = error_to_reward((rpe, spe, mf_rel, mb_rel), p_mb, CONTROL_MODE)
                cum_d_p_mb[sbj] += d_p_mb
                cum_p_mb[sbj] += p_mb
                cum_mf_rel[sbj] += mf_rel
                cum_mb_rel[sbj] += mb_rel
                cum_rpe[sbj] += rpe
                cum_spe[sbj] += spe
                cum_score[sbj] += t_score[sbj]

                """update control agent"""
                if episode > 99:
                    cum_reward[sbj] += t_reward[sbj]
                    next_control_obs = np.append(next_control_obs_frag, [sum(t_rpe)/(2*sbj_size), sum(t_spe)/(2*sbj_size)])
                else:
                    control_action = 0
                    t_reward[sbj] = 0


                control_obs_extra = [rpe, spe]

                env_list[sbj] = env
                forward_list[sbj] = forward
                sarsa_list[sbj] = sarsa
                arb_list[sbj] = arb
            if episode>99:
                ddqn.optimize(control_obs, control_action, next_control_obs, np.mean(t_reward))
            detail_col = [t_rpe, t_spe, t_mf_rel, t_mb_rel, t_p_mb, t_d_p_mb, t_reward, t_score] + [control_action]
            if not return_res:
                gData.add_detail_res(trial + TRIALS_PER_EPISODE * episode, detail_col)
            else:
                res_detail_df.loc[trial + TRIALS_PER_EPISODE * episode] = detail_col
        for sbj in range(sbj_size):
            cum_rpe[sbj] = cum_rpe[sbj]/TRIALS_PER_EPISODE
            cum_spe[sbj] = cum_spe[sbj] / TRIALS_PER_EPISODE
            cum_mb_rel[sbj] = cum_mb_rel[sbj] / TRIALS_PER_EPISODE
            cum_mf_rel[sbj] = cum_mf_rel[sbj] / TRIALS_PER_EPISODE
            cum_p_mb[sbj] = cum_p_mb[sbj] / TRIALS_PER_EPISODE
            cum_d_p_mb[sbj] = cum_d_p_mb[sbj] / TRIALS_PER_EPISODE
            cum_reward[sbj] = cum_reward[sbj] / TRIALS_PER_EPISODE
            cum_score[sbj] = cum_score[sbj] / TRIALS_PER_EPISODE
            for action_num in range(env.NUM_CONTROL_ACTION):
                cum_ctrl_act[action_num][sbj] = cum_ctrl_act[action_num][sbj] / TRIALS_PER_EPISODE


        data_col = [[None]*sbj_size]*12
        data_col = [cum_rpe, cum_spe, cum_mf_rel, cum_mb_rel, cum_p_mb, cum_d_p_mb, cum_reward, cum_score] + list(cum_ctrl_act)

        if not return_res:
            gData.add_res(episode, data_col)
        else:
            res_data_df.loc[episode] = data_col
        if SAVE_LOG_Q_VALUE:
            human_action_list_t.append(human_action_list_episode)

#        if prev_cum_reward <= cum_reward:
#            save_NN = ddqn.eval_Q
#
#        prev_cum_reward = cum_reward

    if SAVE_LOG_Q_VALUE:
        human_action_list_t = np.array(human_action_list_t, dtype=np.int32)
        Task_structure = np.array(Task_structure)
        Transition_t = np.array(Transition_t)
        Q_value_forward_t = np.array(Q_value_forward_t)
        Q_value_sarsa_t = np.array(Q_value_sarsa_t)
        Q_value_arb_t = np.array(Q_value_arb_t)
        '''print(human_action_list_t, human_action_list_t.shape)
        print(Task_structure, Task_structure.shape)
        print(Transition_t, Transition_t.shape)
        print(Q_value_forward_t, Q_value_forward_t.shape )
        print(Q_value_sarsa_t, Q_value_sarsa_t.shape)
        print(Q_value_arb_t, Q_value_arb_t.shape)'''

        gData.add_log_Q_value(human_action_list_t, Task_structure, Transition_t, Q_value_forward_t, Q_value_sarsa_t,
                              Q_value_arb_t)

#    gData.NN = save_NN
    gData.complete_simulation()
    if return_res:
        return (res_data_df, res_detail_df)
'''data.data['min-spe'][0]['rpe'][9][81]
data.detail['min-spe'][0]['rpe'][sbj + trial*sbj_size + TRIALS_PER_EPISODE * episode *sbj_size]'''

if __name__ == '__main__':
    short_opt = "hdn:"
    long_opt = ["help", "mdp-stages=", "disable-control", "ctrl-mode=", "set-param-file=", "trials=", "episodes=",
                "all-mode", "enable-static-control",
                "disable-c-ext", "disable-detail-plot", "less-control-input", "re-analysis=", "PCA-plot",
                "learning-curve-plot", "use-confidence-interval",
                "to-excel=", "disable-action-compare", "enable-score-compare", "use-selected-subjects", "save-ctrl-rl",
                "head-tail-subjects",
                "human-data-compare", "disable-auto-max", "legacy-mode", "separate-learning-curve", "cross-mode-plot",
                "cross-compare=", "sub-A=", "sub-B=",
                "enhance-compare=", "no-reset", "save-log-Q-value", "to-excel-log-Q-value", "mixed-random-mode",
                "to-excel-random-mode-sequence",
                "to-excel-optimal-sequence", "opposite-cross-compare=", "opposite-nn-cross-compare=",
                "fair-opposite-nn-cross-compare=", "file-suffix=", "task-type="]
    try:
        opts, args = getopt.getopt(sys.argv[1:], short_opt, long_opt)
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    for o, a in opts:
        if o == "--ctrl-mode":
            CONTROL_MODE = a
        elif o == "--file-suffix":
            FILE_SUFFIX = a
            print(FILE_SUFFIX)
        elif o == "--MF_ONLY":
            MF_ONLY = bool(a)
        elif o == "--MB_ONLY":
            MB_ONLY = bool(a)
        elif o == "--task-type":
            TASK_TYPE = int(a)
        else:
            assert False, "unhandled option"

    if TASK_TYPE == 2019:
        MDP.NUM_CONTROL_ACTION = 4
    elif TASK_TYPE == 2020:
        MDP.NUM_CONTROL_ACTION = 5

    analysis.ACTION_COLUMN = ['action_' + str(action_num) for action_num in range(MDP.NUM_CONTROL_ACTION)]
    analysis.COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'd_p_mb', 'ctrl_reward', 'score'] + analysis.ACTION_COLUMN


    gData.trial_separation = TRIALS_PER_EPISODE
    gData.new_mode(CONTROL_MODE)
    print('Running mode: ' + CONTROL_MODE)
    simulation()
    gData.save_mode(CONTROL_MODE)

    # Save the whole analysis object for future reference
    pkl_file_name = 'Analysis-Object'
    if not ALL_MODE:
        pkl_file_name += '-'
        pkl_file_name += CONTROL_MODE
    pkl_file_name += '-'
    pkl_file_name += '{0:02d}'.format(NUM_PARAMETER_SET)
    with open(gData.file_name(pkl_file_name) + FILE_SUFFIX + '.pkl', 'wb') as f:
        pickle.dump(gData, f, pickle.HIGHEST_PROTOCOL)
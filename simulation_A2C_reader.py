""" Shuangyi Tong <s9tong@edu.uwaterloo.ca>
    Sept 17, 2018
"""
import getopt
import sys
import csv
import torch
import numpy as np
import pandas as pd
import dill as pickle # see https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
import random

from tqdm import tqdm
from mdp import MDP
from ddqn import DoubleDQN
from sarsa import SARSA
from forward import FORWARD
from arbitrator import BayesRelEstimator, AssocRelEstimator, Arbitrator
import analysis
analysis.IS_READER = True
analysis.HUMAN_DATA_COLUMN = analysis.HUMAN_DATA_COLUMN + ['Tag']
analysis.DETAIL_COLUMNS = analysis.DETAIL_COLUMNS + ['model_indx', 'tag']
analysis.DETAIL_COLUMNS = analysis.DETAIL_COLUMNS + ['tag_' + str(action_num) for action_num in range(4)]
analysis.COLUMNS = analysis.COLUMNS + ['model_indx', 'tag']
analysis.COLUMNS = analysis.COLUMNS + ['tag_' + str(action_num) for action_num in range(4)]
analysis.COLUMNS = analysis.COLUMNS + ['validation']
DETAIL_COLUMNS = analysis.DETAIL_COLUMNS
COLUMNS = analysis.COLUMNS
from analysis import gData, RESULTS_FOLDER
#from analysis import gData, RESULTS_FOLDER, COLUMNS, DETAIL_COLUMNS
from common import makedir
import A2C
from torch.utils.tensorboard import SummaryWriter
from main_reader import MODE_LIST
from gym import spaces
from functools import reduce
from torch import nn
from torch.nn import functional as F

# preset constants
MDP_STAGES            = 2
TOTAL_EPISODES        = 100
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
INIT_CTRL_INPUT       = np.ones(4)*0.25
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
TASK_TYPE = 2020
MF_ONLY = False
MB_ONLY = False
Reproduce_BHV = False
saved_policy_path = ''
Session_block = False
mode202010 = False
DECAY_RATE = 0.5
turn_off_tqdm = False
abs_rpe_mode = True
CONTROL_resting = 99 #Intial duration for CONTROL agent resting
TOTAL_PRE_EPISODES = 0 #100
NUM_ARBS = 1
NUM_SBJ = 82
STORE_DETAILS = True
NUM_TAGS = 4

RESET = False
SAVE_LOG_Q_VALUE = False
MIXED_RANDOM_MODE = False
RANDOM_MODE_LIST = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe']
analysis.IS_READER = True


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
    '''
    if TASK_TYPE == 2019:
        try:
            cmp_func = error_reward_map[mode]
        except KeyError:
            print("Warning: control mode {0} not found, use default mode {1}".format(mode, DEFAULT_CONTROL_MODE))
            cmp_func = error_reward_map[DEFAULT_CONTROL_MODE]

        return cmp_func(error)
    '''
    if TASK_TYPE == 2020:
        if mode == 'min-rpe':
            reward = 40 - error[0]
        elif mode == 'max-rpe':
            reward = error[0]
        elif mode == 'min-spe':
            reward = (1 - error[1])*10
        elif mode == 'max-spe':
            reward = error[1]*10
        elif mode == 'min-rpe-min-spe':
            reward = (40 - error[0]) + (1 - error[1]) * 10
        elif mode == 'max-rpe-max-spe':
            reward = (error[0]) + (error[1]) * 10
        elif mode == 'min-rpe-max-spe':
            reward = (40 - error[0]) + (error[1]) * 10
        elif mode == 'max-rpe-min-spe':
            reward = (error[0]) + (1 - error[1]) * 10
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

def _linear_size(gym_space):
    """Calculate the size of input/output based on descriptive structure (i.e.
    observation_space/action_space) defined by gym.spaces
    """
    res = 0
    if isinstance(gym_space, spaces.Tuple):
        for space in gym_space.spaces:
            res += _linear_size(space)
        return res
    elif isinstance(gym_space, spaces.MultiBinary) or \
         isinstance(gym_space, spaces.Discrete):
        return gym_space.n
    elif isinstance(gym_space, spaces.Box):
        return reduce(lambda x,y: x*y, gym_space.shape)
    else:
        raise NotImplementedError

def simulation(*param_list, PARAMETER_SET='DEFAULT',
               return_res=False):
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
                        However, it is encoded in the arbitrator.py, not in the current code.else:
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
    if Session_block:
        MODE_LIST = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'random']
    else:
        MODE_LIST = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'max-rpe-max-spe', 'min-rpe-min-spe',
                     'max-rpe-min-spe', 'min-rpe-max-spe', 'min-rpe-PMB', 'max-rpe-PMB', 'min-rpe-max-spe-PMB',
                     'random']

    CHANGE_MODE_TERM = int(TOTAL_EPISODES/len(RANDOM_MODE_LIST))
    if return_res:
        res_data_df = pd.DataFrame(columns=COLUMNS)
        res_detail_df = pd.DataFrame(columns=DETAIL_COLUMNS)
    env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE, task_type=TASK_TYPE,
              reader_mode=True, num_tags = len(INIT_CTRL_INPUT))
    MDP.DECAY_RATE = DECAY_RATE
    analysis.ACTION_COLUMN = ['action_' + str(action_num) for action_num in range(MDP.NUM_CONTROL_ACTION)]
    #analysis.COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'd_p_mb', 'ctrl_reward',
    #                    'score'] + analysis.ACTION_COLUMN
    if MIXED_RANDOM_MODE:
        random_mode_list = np.random.choice(RANDOM_MODE_LIST, 4 , replace =False) # order among 4 mode is random
        print ('Random Mode Sequence : %s' %random_mode_list)

    # if it is mixed random mode, 'ddpn_loaded' from torch.save(model, filepath) is used instead of 'ddpn'
    writer = SummaryWriter("runs/mish_activation")
    A2C.state_dim = _linear_size(env.observation_space[MDP.CONTROL_AGENT_INDEX]) - 4 + 5 + 1# remove tag_probs / add SBJ_info_for_NN & trials
    A2C.n_actions = _linear_size(env.action_space[MDP.CONTROL_AGENT_INDEX])
    actor_tag = A2C.Actor(A2C.state_dim, n_actions=NUM_TAGS, activation=A2C.Mish)
    actor_action = A2C.Actor(A2C.state_dim, n_actions=MDP.NUM_CONTROL_ACTION, activation=A2C.Mish)
    critic = A2C.Critic(A2C.state_dim, activation=A2C.Mish)
    learner = A2C.A2CLearner(actor_tag, actor_action, critic)
    state = None
    steps = 0
    episode_reward = 0
    episode_rewards = []
    # ddqn    = DoubleDQN(env.observation_space[MDP.CONTROL_AGENT_INDEX],
    #                     env.action_space[MDP.CONTROL_AGENT_INDEX],
    #                     torch.cuda.is_available(), Session_block=Session_block) # use DDQN for control agent


    gData.new_simulation()
    #gData.add_human_data([amp_mf_to_mb / amp_mb_to_mf, rl_learning_rate, estimator_learning_rate, threshold, temperature, performance])
    control_obs_extra = INIT_CTRL_INPUT
    arb_save = [None] * (NUM_ARBS*NUM_SBJ)
    sarsa_save = [None] * (NUM_ARBS*NUM_SBJ)
    forward_save = [None] * (NUM_ARBS*NUM_SBJ)
    tag_save = [None] * (NUM_ARBS*NUM_SBJ)
    validation_accuracy = [None] * (TOTAL_EPISODES)

    if Reproduce_BHV:
        saved_policy = np.load('history_results/'+saved_policy_path)
    for sbj_indx in range(NUM_SBJ):
        print('LOADING SBJ {0:00d} / {1:00d}'.format(sbj_indx+1,NUM_SBJ))
        if turn_off_tqdm == False:
            arb_list = tqdm(range(NUM_ARBS))
        else:
            arb_list = range(NUM_ARBS)
        for arb_indx in arb_list:
            filename = 'history_results/sbjs/sbj-{0:02d}-rep-{1:02d}.pkl'.format(sbj_indx, arb_indx)
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            '''
            data = {
                'MF_est_chi': MF_estimator_chi_save,
                'MB_est_categories': MB_estimator_categories_save,
                'MB_est_pe_records_counter': MB_estimator_pe_records_counter_save,
                'MB_est_pe_records': MB_estimator_pe_records_save,
                'MB_est_target_category': MB_estimator_target_category_save,
                'forwards': forward_save,
                'sarsas': sarsa_save,
                'params': param_list[sbj_indx]
            }
            '''
            estimator_learning_rate = data['params'][0]
            threshold = data['params'][1]
            amp_mb_to_mf = data['params'][2]
            amp_mf_to_mb = data['params'][3]
            rl_learning_rate = data['params'][4]
            temperature = data['params'][5]
            performance = data['params'][6]
            tag = data['params'][7] - 1
            sarsa = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX], env, learning_rate=rl_learning_rate,
                          nine_states_mode=env.nine_states_mode)  # SARSA model-free learner
            for state in range(len(sarsa.Q_sarsa)):
                sarsa.Q_sarsa[state] = data['sarsas'][state]
            forward = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                              env.action_space[MDP.HUMAN_AGENT_INDEX],
                              env.state_reward_func, env.output_states_offset, env.reward_map_func,
                              learning_rate=rl_learning_rate, disable_cforward=DISABLE_C_EXTENSION,
                              output_state_array=env.output_states_indx,
                              nine_states_mode=env.nine_states_mode)  # forward model-based learner
            forward.T = data['forwards'].copy()
            arb = Arbitrator(AssocRelEstimator(estimator_learning_rate, env.max_rpe),
                             BayesRelEstimator(thereshold=threshold), temperature=temperature,
                             amp_mb_to_mf=amp_mb_to_mf, amp_mf_to_mb=amp_mf_to_mb, MB_ONLY=MB_ONLY, MF_ONLY=MF_ONLY)
            arb.mf_rel_estimator.chi = data['MF_est_chi']
            arb.mb_rel_estimator.categories = data['MB_est_categories']
            arb.mb_rel_estimator.pe_records_counter = data['MB_est_pe_records_counter'].copy()
            arb.mb_rel_estimator.pe_records = data['MB_est_pe_records'].copy()
            arb.mb_rel_estimator.target_category = data['MB_est_target_category']
            # register in the communication controller
            env.agent_comm_controller.register('model-based', forward)
            arb_save[sbj_indx*NUM_ARBS+arb_indx] = arb
            sarsa_save[sbj_indx*NUM_ARBS+arb_indx] = sarsa
            forward_save[sbj_indx*NUM_ARBS+arb_indx] = forward
            tag_save[sbj_indx*NUM_ARBS+arb_indx] = tag


    LEN_TOT_EPS = NUM_SBJ*NUM_ARBS*TOTAL_PRE_EPISODES+NUM_SBJ*NUM_ARBS*TOTAL_EPISODES
    print(LEN_TOT_EPS)
    gData.fast_initialization_DATA(LEN_TOT_EPS)
    if STORE_DETAILS:
        gData.fast_initialization_DETAIL(LEN_TOT_EPS * TRIALS_PER_EPISODE)
    'pre-training start'
    'pre-training completed'
    memory = []


    episode_list = range(TOTAL_EPISODES)
    prev_validation_accuracy = 0
    for episode in episode_list:
        # validation_set_indx = np.add(random.choices(range(NUM_ARBS),k=NUM_SBJ), [float(x)*NUM_ARBS for x in range(NUM_SBJ)])
#        training_set_indx = np.delete(list(range(NUM_ARBS*NUM_SBJ)),validation_set_indx)
        validation_set_indx = random.choices(range(NUM_SBJ), k=1)
        training_set_indx = np.delete(list(range(NUM_ARBS * NUM_SBJ)), validation_set_indx)
        if turn_off_tqdm == False:
            training_set_indx_list = tqdm(range(len(training_set_indx)))
        else:
            training_set_indx_list = range(len(training_set_indx))
        for indx in training_set_indx_list:
            model_indx = training_set_indx[indx]
            #print('Episode ' + str(episode) + '/' + str(len(episode_list)) + '-training model '\
            #      + str(np.where(training_set_indx==model_indx)[0][0]) + '/' + str(len(training_set_indx)))
            tag = tag_save[model_indx]
            env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE, task_type=TASK_TYPE,
                      reader_mode=True, num_tags = len(INIT_CTRL_INPUT))
            env.reward_map = env.reward_map_copy.copy()
            env.output_states = env.output_states_copy.copy()
            sarsa = sarsa_save[model_indx]
            forward = forward_save[model_indx]
            arb = arb_save[model_indx]
            env.agent_comm_controller.register('model-based', forward)
            arb.episode_number = episode
            arb.CONTROL_resting = CONTROL_resting
            cum_d_p_mb = cum_p_mb = cum_mf_rel = cum_mb_rel = cum_rpe = cum_spe = cum_reward = cum_score = cum_real_rwd = 0
            cum_ctrl_act = np.zeros(MDP.NUM_CONTROL_ACTION)
            human_action_list_episode = []
            tag_prob = INIT_CTRL_INPUT
            done = False
            SBJ_info_for_NN = np.ones(5)*-1
            for trial in range(TRIALS_PER_EPISODE):
                if trial == TRIALS_PER_EPISODE - 1: done = True
                block_indx = trial // int(TRIALS_PER_EPISODE / 4)
                if trial%TRIALS_PER_EPISODE == 0:
                    #if episode > CONTROL_resting:
                    env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE,
                              task_type=TASK_TYPE, reader_mode=True, num_tags = len(INIT_CTRL_INPUT))
                    env.reward_map = env.reward_map_copy.copy()
                    env.output_states = env.output_states_copy.copy()
                #if episode <= CONTROL_resting:
                #    env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE, task_type=TASK_TYPE)
                #print(env.reward_map)
                t_d_p_mb = t_p_mb = t_mf_rel = t_mb_rel = t_rpe = t_spe = t_reward = t_score = rpe1 = rpe2  = 0
                game_terminate              = False
                human_obs, control_obs_frag = env.reset()
                #control_obs                 = np.append(control_obs_frag, control_obs_extra, SBJ_info_for_NN)

                """control agent choose action"""
                #if episode > CONTROL_resting:
                if trial == 0:
                    control_action = 0
                elif STATIC_CONTROL_AGENT:
                    control_action = static_action_map[CONTROL_MODE][trial]
                else:
                    #control_action = ddqn.action(control_obs)
                    state = control_obs
                    #print(actor(A2C.t(state)).probs)
                    control_action = actor_action(A2C.t(state)).sample().item()
                    #dist = torch.distributions.Categorical(probs=probs)
                    #control_action = dist.sample().item()
                    #control_action = probs.sample().item()
                    if trial%10 == 4: control_action = 1
                    if trial%10 == 9: control_action = 2
                cum_ctrl_act[control_action] += 1
                if TASK_TYPE == 2019:
                    if control_action == 3:
                        if env.is_flexible == 1:
                            arb.p_mb = 0.8
                            arb.p_mf = 0.2
                        else:
                            arb.p_mb = 0.2
                            arb.p_mf = 0.8
                elif TASK_TYPE == 2021:
                    if control_action == 2:
                        if env.is_flexible == 1:
                            arb.p_mb = 0.8
                            arb.p_mf = 0.2
                        else:
                            arb.p_mb = 0.2
                            arb.p_mf = 0.8
                if MB_ONLY:
                    arb.p_mb = 0.9999
                    arb.p_mf = 1-arb.p_mb
                elif MF_ONLY:
                    arb.p_mb = 0.0001
                    arb.p_mf = 1 - arb.p_mb
                """control act on environment"""
                if CTRL_AGENTS_ENABLED:  # and (trial % ACTION_PERIOD is 0 or not STATIC_CONTROL_AGENT): ## why trial % ACTION_PERIOD ??
                    _, _, _, _ = env.step([MDP.CONTROL_AGENT_INDEX, control_action])
                if TASK_TYPE == 2019:
                    if trial == 0:
                        if env.is_flexible == 1:
                            arb.p_mb = 0.8
                            arb.p_mf = 0.2
                        else:
                            arb.p_mb = 0.2
                            arb.p_mf = 0.8
                elif TASK_TYPE == 2021:
                    if trial == 0:
                        if env.is_flexible == 1:
                            arb.p_mb = 0.8
                            arb.p_mf = 0.2
                        else:
                            arb.p_mb = 0.2
                            arb.p_mf = 0.8
                forward.bwd_update(env.bwd_idf,env)
                current_game_step = 0
                while not game_terminate:
                    """human choose action"""
                    human_action = compute_human_action(arb, human_obs, sarsa, forward)
                    #print("human action : ", human_action)
                    if SAVE_LOG_Q_VALUE:
                        human_action_list_episode.append(human_action)

                    """human act on environment"""
                    next_human_obs, human_reward, game_terminate, next_control_obs_frag \
                        = env.step((MDP.HUMAN_AGENT_INDEX, human_action))

                    """update human agent"""
                    spe = forward.optimize(human_obs, human_reward, human_action, next_human_obs, env)
                    next_human_action = compute_human_action(arb, next_human_obs, sarsa, forward) # required by models like SARSA
                    if env.is_flexible == 1: #flexible goal condition
                        rpe = sarsa.optimize(human_reward, human_action, next_human_action, human_obs, next_human_obs)
                    else: # specific goal condition human_reward should be normalized to sarsa
                        if human_reward > 0: # if reward is 10, 20, 40
                            rpe = sarsa.optimize(40, human_action, next_human_action, human_obs, next_human_obs)
                        else:
                            rpe = sarsa.optimize(0, human_action, next_human_action, human_obs, next_human_obs)

                    if abs_rpe_mode: rpe = abs(rpe)
                    mf_rel, mb_rel, p_mb, d_p_mb = arb.add_pe(rpe, spe)
                    t_d_p_mb += d_p_mb
                    t_p_mb   += p_mb
                    t_mf_rel += mf_rel
                    t_mb_rel += mb_rel
                    t_rpe    += abs(rpe)
                    #t_rpe += rpe
                    t_spe    += spe
                    t_score  += human_reward # if not the terminal state, human_reward is 0, so simply add here is fine
                    """iterators update"""
                    SBJ_info_for_NN[current_game_step * 2] = human_obs
                    SBJ_info_for_NN[current_game_step * 2 + 1] = human_action
                    human_obs = next_human_obs
                    if current_game_step == 0:
                        # rpe1 = abs(rpe)
                        rpe1 = rpe
                        spe1 = spe
                    else:
                        # rpe2 = abs(rpe)
                        rpe2 = rpe
                        spe2 = spe

                    current_game_step += 1

                SBJ_info_for_NN[4] = human_obs
                #control_obs = np.append(control_obs_frag, control_obs_extra)
                control_obs = np.append(control_obs_frag, SBJ_info_for_NN)
                control_obs = np.append(control_obs, trial)

                # calculation after one trial
                d_p_mb, p_mb, mf_rel, mb_rel, rpe, spe = list(map(lambda x: x / MDP_STAGES, [
                t_d_p_mb, t_p_mb, t_mf_rel, t_mb_rel, t_rpe, t_spe])) # map to average value

                cum_d_p_mb += d_p_mb
                cum_p_mb   += p_mb
                cum_mf_rel += mf_rel
                cum_mb_rel += mb_rel
                cum_rpe    += rpe
                cum_spe    += spe
                cum_score  += t_score

                """update control agent"""

                #if episode > 99:
                #memory.append((control_action, t_reward, control_obs, next_control_obs, done, tag, tag_prob))
                # memory = [control_action, t_reward, control_obs, next_control_obs, done]
                steps += 1
                #t_reward = torch.sum(learner.learn(memory, steps, discount_rewards=False)).item()
                #loss = nn.CrossEntropyLoss()
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                #tag_vec = torch.squeeze(Variable(self.Tensor(mini_batch.tag)),1)
                tag_vec = np.zeros(NUM_TAGS)
                tag_vec[int(tag)] = 1
                tag_vec = torch.FloatTensor([tag_vec])
                tag_prob = torch.clamp(actor_tag.model(torch.FloatTensor(control_obs)), min=1e-3)
                t_reward = cos(torch.unsqueeze(tag_prob,0),tag_vec.long())

                #next_control_obs = np.append(env._make_control_observation(), tag_prob, SBJ_info_for_NN)
                if trial > 0:
                    learner.learn(prev_control_obs, control_action, control_obs, t_reward, done, tag, tag_prob, steps,
                                  discount_rewards=False)
                # = torch.sum(learner.learn(control_obs, control_action, next_control_obs, t_reward, done, tag, tag_prob, steps, discount_rewards=False)).item()
                # tag_prob = torch.distributions.Categorical(critic.model(A2C.t(control_obs))).probs.tolist()
                #print([trial] + tag_prob + [tag])
                episode_reward = t_reward
                episode_rewards.append(episode_reward)
                real_rwd = t_reward
                cum_reward = t_reward
                    #ddqn.optimize(control_obs, control_action, next_control_obs, t_reward)
                #else:
                #    control_action = 0
                #    t_reward = 0
                #    real_rwd = t_reward
                cum_real_rwd += real_rwd

                control_obs_extra = tag_prob.tolist()
                detail_col = [rpe, spe, mf_rel, mb_rel, p_mb, d_p_mb, t_reward, t_score] + [control_action] + [rpe1, rpe2] \
                            + [spe1, spe2]
                detail_col = detail_col + env.reward_map + [env.visited_goal_state]
                detail_col = detail_col + [real_rwd] + [model_indx, tag] + tag_prob.tolist()
                if STORE_DETAILS:
                    if not return_res:
                        gData.add_detail_res(trial + TRIALS_PER_EPISODE * (episode*NUM_SBJ*NUM_ARBS + indx), detail_col)
                    else:
                        res_detail_df.loc[trial + TRIALS_PER_EPISODE * (episode*NUM_SBJ*NUM_ARBS + indx)] = detail_col
                prev_control_obs = control_obs

            data_col = list(map(lambda x: x / TRIALS_PER_EPISODE,
                            [cum_rpe, cum_spe, cum_mf_rel, cum_mb_rel, cum_p_mb, cum_d_p_mb, cum_reward, cum_score]
                            + list(cum_ctrl_act) + [cum_real_rwd]  + [model_indx, tag] + tag_prob.tolist() + [tag_prob.tolist()[int(tag)]]))

            if not return_res:
                gData.add_res(episode*NUM_SBJ*NUM_ARBS + indx, data_col)
            else:
                res_data_df.loc[episode*NUM_SBJ*NUM_ARBS + indx] = data_col


        num_correct = 0
        if turn_off_tqdm == False:
            validation_set_indx_list = tqdm(range(len(validation_set_indx)))
        else:
            validation_set_indx_list = range(len(validation_set_indx))
        for indx in validation_set_indx_list:
            model_indx = int(validation_set_indx[indx])
            #print('Episode ' + str(episode) + '/' + str(len(episode_list)) + '-validation with model '\
            #      + str(np.where(validation_set_indx==model_indx)[0][0]) + '/' + str(len(validation_set_indx)))
            tag = tag_save[model_indx]
            env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE, task_type=TASK_TYPE,
                      reader_mode=True, num_tags = len(INIT_CTRL_INPUT))
            env.reward_map = env.reward_map_copy.copy()
            env.output_states = env.output_states_copy.copy()
            sarsa = sarsa_save[model_indx]
            forward = forward_save[model_indx]
            arb = arb_save[model_indx]
            env.agent_comm_controller.register('model-based', forward)
            arb.episode_number = episode
            arb.CONTROL_resting = CONTROL_resting
            cum_d_p_mb = cum_p_mb = cum_mf_rel = cum_mb_rel = cum_rpe = cum_spe = cum_reward = cum_score = cum_real_rwd = 0
            cum_ctrl_act = np.zeros(MDP.NUM_CONTROL_ACTION)
            human_action_list_episode = []
            tag_prob = INIT_CTRL_INPUT
            done = False
            SBJ_info_for_NN = np.ones(5) * -1
            for trial in range(TRIALS_PER_EPISODE):
                if trial == TRIALS_PER_EPISODE - 1: done = True
                block_indx = trial // int(TRIALS_PER_EPISODE / 4)
                if trial % TRIALS_PER_EPISODE == 0:
                    #if episode > CONTROL_resting:
                    env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE,
                              task_type=TASK_TYPE, reader_mode=True, num_tags = len(INIT_CTRL_INPUT))
                    env.reward_map = env.reward_map_copy.copy()
                    env.output_states = env.output_states_copy.copy()
                #if episode <= CONTROL_resting:
                #    env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE,task_type=TASK_TYPE)
                # print(env.reward_map)
                t_d_p_mb = t_p_mb = t_mf_rel = t_mb_rel = t_rpe = t_spe = t_reward = t_score = rpe1 = rpe2 = 0
                game_terminate = False
                human_obs, control_obs_frag = env.reset()
                """control agent choose action"""
                #if episode > CONTROL_resting:
                if trial == 0:
                    control_action = 0
                elif STATIC_CONTROL_AGENT:
                    control_action = static_action_map[CONTROL_MODE][trial]
                else:
                    # control_action = ddqn.action(control_obs)
                    state = control_obs
                    control_action = actor_action(A2C.t(state)).sample().item()
                    if trial%10 == 4: control_action = 1
                    if trial%10 == 9: control_action = 2
                    #probs = actor(A2C.t(state))
                    #dist = torch.distributions.Categorical(probs=probs)
                    #control_action = dist.sample().item()
                    #print([control_action, probs])
                cum_ctrl_act[control_action] += 1
                if TASK_TYPE == 2019:
                    if control_action == 3:
                        if env.is_flexible == 1:
                            arb.p_mb = 0.8
                            arb.p_mf = 0.2
                        else:
                            arb.p_mb = 0.2
                            arb.p_mf = 0.8
                elif TASK_TYPE == 2021:
                    if control_action == 2:
                        if env.is_flexible == 1:
                            arb.p_mb = 0.8
                            arb.p_mf = 0.2
                        else:
                            arb.p_mb = 0.2
                            arb.p_mf = 0.8
                if MB_ONLY:
                    arb.p_mb = 0.9999
                    arb.p_mf = 1 - arb.p_mb
                elif MF_ONLY:
                    arb.p_mb = 0.0001
                    arb.p_mf = 1 - arb.p_mb
                """control act on environment"""
                if CTRL_AGENTS_ENABLED:  # and (trial % ACTION_PERIOD is 0 or not STATIC_CONTROL_AGENT): ## why trial % ACTION_PERIOD ??
                    _, _, _, _ = env.step([MDP.CONTROL_AGENT_INDEX, control_action])

                if TASK_TYPE == 2019:
                    if trial == 0:
                        if env.is_flexible == 1:
                            arb.p_mb = 0.8
                            arb.p_mf = 0.2
                        else:
                            arb.p_mb = 0.2
                            arb.p_mf = 0.8
                elif TASK_TYPE == 2021:
                    if trial == 0:
                        if env.is_flexible == 1:
                            arb.p_mb = 0.8
                            arb.p_mf = 0.2
                        else:
                            arb.p_mb = 0.2
                            arb.p_mf = 0.8
                forward.bwd_update(env.bwd_idf, env)
                current_game_step = 0
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
                    spe = forward.optimize(human_obs, human_reward, human_action, next_human_obs, env)
                    next_human_action = compute_human_action(arb, next_human_obs, sarsa,
                                                             forward)  # required by models like SARSA
                    if env.is_flexible == 1:  # flexible goal condition
                        rpe = sarsa.optimize(human_reward, human_action, next_human_action, human_obs, next_human_obs)
                    else:  # specific goal condition human_reward should be normalized to sarsa
                        if human_reward > 0:  # if reward is 10, 20, 40
                            rpe = sarsa.optimize(40, human_action, next_human_action, human_obs, next_human_obs)
                        else:
                            rpe = sarsa.optimize(0, human_action, next_human_action, human_obs, next_human_obs)

                    if abs_rpe_mode: rpe = abs(rpe)
                    mf_rel, mb_rel, p_mb, d_p_mb = arb.add_pe(rpe, spe)
                    t_d_p_mb += d_p_mb
                    t_p_mb += p_mb
                    t_mf_rel += mf_rel
                    t_mb_rel += mb_rel
                    t_rpe += abs(rpe)
                    # t_rpe += rpe
                    t_spe += spe
                    t_score += human_reward  # if not the terminal state, human_reward is 0, so simply add here is fine
                    """iterators update"""
                    human_obs = next_human_obs
                    if current_game_step == 0:
                        # rpe1 = abs(rpe)
                        rpe1 = rpe
                        spe1 = spe
                    else:
                        # rpe2 = abs(rpe)
                        rpe2 = rpe
                        spe2 = spe
                    SBJ_info_for_NN[2 * current_game_step] = human_obs
                    SBJ_info_for_NN[2 * current_game_step + 1] = human_action
                    current_game_step += 1

                SBJ_info_for_NN[4] = next_human_obs
                # calculation after one trial
                d_p_mb, p_mb, mf_rel, mb_rel, rpe, spe = list(map(lambda x: x / MDP_STAGES, [
                    t_d_p_mb, t_p_mb, t_mf_rel, t_mb_rel, t_rpe, t_spe]))  # map to average value

                cum_d_p_mb += d_p_mb
                cum_p_mb += p_mb
                cum_mf_rel += mf_rel
                cum_mb_rel += mb_rel
                cum_rpe += rpe
                cum_spe += spe
                cum_score += t_score

                """update control agent"""
                steps += 1
                #control_action = 0
                #loss = nn.CrossEntropyLoss()
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                #tag_vec = torch.squeeze(Variable(self.Tensor(mini_batch.tag)),1)
                tag_vec = np.zeros(NUM_TAGS)
                tag_vec[int(tag)] = 1
                tag_vec = torch.FloatTensor([tag_vec])
                t_reward = cos(torch.unsqueeze(torch.clamp(actor_tag.model(torch.FloatTensor(control_obs)), min=1e-3),0),
                                            tag_vec.long())
                #control_obs = np.append(control_obs_frag, control_obs_extra)
                control_obs = np.append(control_obs_frag, SBJ_info_for_NN)
                control_obs = np.append(control_obs, trial)
                #if trial > 0:
                #    learner.learn(prev_control_obs, control_action, control_obs, t_reward, done, tag, tag_prob, steps,
                #                  discount_rewards=False)
                # = torch.sum(learner.learn(control_obs, control_action, next_control_obs, t_reward, done, tag, tag_prob, steps, discount_rewards=False)).item()
                real_rwd = t_reward
                cum_reward += t_reward
                episode_reward += t_reward
                episode_rewards.append(episode_reward)
                cum_real_rwd += real_rwd
                #tag_prob = torch.distributions.Categorical(critic.model(A2C.t(control_obs))).probs.tolist()
                tag_prob = torch.clamp(actor_tag.model(torch.FloatTensor(control_obs)), min=1e-3)
                t_reward = cos(torch.unsqueeze(tag_prob, 0), tag_vec.long())
                control_obs_extra = tag_prob.tolist()
                print(tag_prob)
                detail_col = [rpe, spe, mf_rel, mb_rel, p_mb, d_p_mb, t_reward, t_score] + [control_action] + [rpe1, rpe2] \
                             + [spe1, spe2]
                detail_col = detail_col + env.reward_map + [env.visited_goal_state]
                detail_col = detail_col + [real_rwd] + [model_indx, tag] + tag_prob.tolist()
                if STORE_DETAILS:
                    if not return_res:
                        gData.add_detail_res(trial + TRIALS_PER_EPISODE * (episode*NUM_SBJ*NUM_ARBS+ indx + len(training_set_indx_list)), detail_col)
                    else:
                        res_detail_df.loc[trial + TRIALS_PER_EPISODE * (episode*NUM_SBJ*NUM_ARBS+ indx + len(training_set_indx_list))] = detail_col
                prev_control_obs = control_obs

            # if tag == random.choice(range(4), weights = tag_prob, k=1):
            # num_correct += 1
            # num_correct += tag_prob[int(tag)] / len(validation_set_indx)
            num_correct = tag_prob.tolist()[int(tag)]
            data_col = list(map(lambda x: x / TRIALS_PER_EPISODE,
                                [cum_rpe, cum_spe, cum_mf_rel, cum_mb_rel, cum_p_mb, cum_d_p_mb, cum_reward, cum_score]
                                + list(cum_ctrl_act) + [cum_real_rwd] + [model_indx*TRIALS_PER_EPISODE, tag*TRIALS_PER_EPISODE] + tag_prob.tolist() +[num_correct*TRIALS_PER_EPISODE]))

            if not return_res:
                gData.add_res(episode*NUM_SBJ*NUM_ARBS+ indx + len(training_set_indx_list), data_col)
            else:
                res_data_df.loc[episode*NUM_SBJ*NUM_ARBS + indx + len(training_set_indx_list)] = data_col


        validation_accuracy[episode] = num_correct
        if prev_validation_accuracy <= validation_accuracy[episode]:
            save_NN_AT = actor_tag.model
            save_NN_AA = actor_action.model
            save_NN_C = critic.model
            prev_validation_accuracy = validation_accuracy[episode]
        print("episode : " + str(episode+1) + "/"+ str(TOTAL_EPISODES) + ", validation_accuracy: " + str(validation_accuracy[episode]) + "at tag : " +str(tag))

    if SAVE_CTRL_RL and not MIXED_RANDOM_MODE:
        makedir(RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE)
        # torch.save(ddqn.eval_Q.state_dict(), RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE + '/MLP_OBJ_Subject' + PARAMETER_SET) # save model as dictionary
        torch.save(actor_tag.model,
                   RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE + '/MLP_OBJ_Subject' + PARAMETER_SET)  # save entire model
        torch.save(actor_action.model,
                   RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE + '/MLP_OBJ_Subject' + PARAMETER_SET)  # save entire model
        torch.save(critic.model,
                   RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE + '/MLP_OBJ_Subject' + PARAMETER_SET)  # save entire model
    if ENABLE_PLOT:
        gData.plot_all_human_param(CONTROL_MODE + ' Human Agent State - parameter set: ' + PARAMETER_SET)
        gData.plot_pe(CONTROL_MODE, CONTROL_MODE + ' - parameter set: ' + PARAMETER_SET)
        gData.plot_action_effect(CONTROL_MODE, CONTROL_MODE + ' Action Summary - parameter set: ' + PARAMETER_SET)
    gData.NN = [save_NN_AA, save_NN_AT, save_NN_C]

    if SAVE_CTRL_RL and not MIXED_RANDOM_MODE:
        makedir(RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE)
        # torch.save(ddqn.eval_Q.state_dict(), RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE + '/MLP_OBJ_Subject' + PARAMETER_SET) # save model as dictionary
        torch.save(actor_action.model,
                   RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE + '/MLP_OBJ_Subject' + PARAMETER_SET)  # save entire model
        torch.save(actor_tag.model,
                   RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE + '/MLP_OBJ_Subject' + PARAMETER_SET)  # save entire model
        torch.save(critic.model,
                   RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE + '/MLP_OBJ_Subject' + PARAMETER_SET)  # save entire model
    if ENABLE_PLOT:
        gData.plot_all_human_param(CONTROL_MODE + ' Human Agent State - parameter set: ' + PARAMETER_SET)
        gData.plot_pe(CONTROL_MODE, CONTROL_MODE + ' - parameter set: ' + PARAMETER_SET)
        gData.plot_action_effect(CONTROL_MODE, CONTROL_MODE + ' Action Summary - parameter set: ' + PARAMETER_SET)

    gData.complete_simulation()

    if return_res:
        return (res_data_df, res_detail_df)



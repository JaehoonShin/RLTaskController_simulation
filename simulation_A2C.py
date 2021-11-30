""" Shuangyi Tong <s9tong@edu.uwaterloo.ca>
    Sept 17, 2018
"""
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
from analysis import gData, RESULTS_FOLDER, COLUMNS, DETAIL_COLUMNS
from common import makedir
import pg_2act as pg
from torch.utils.tensorboard import SummaryWriter
import analysis
from main_A2C import MODE_LIST
from gym import spaces
from functools import reduce

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
TASK_TYPE = 2020
MF_ONLY = False
MB_ONLY = False
Reproduce_BHV = False
saved_policy_path = ''
Session_block = False
mode202010 = False
DECAY_RATE = 0.75
turn_off_tqdm = False
abs_rpe_mode = True

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
    if TASK_TYPE == 2019:
        try:
            cmp_func = error_reward_map[mode]
        except KeyError:
            print("Warning: control mode {0} not found, use default mode {1}".format(mode, DEFAULT_CONTROL_MODE))
            cmp_func = error_reward_map[DEFAULT_CONTROL_MODE]

        return cmp_func(error)
    elif TASK_TYPE == 2020:
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

def simulation(threshold=BayesRelEstimator.THRESHOLD, estimator_learning_rate=AssocRelEstimator.LEARNING_RATE,
               amp_mb_to_mf=Arbitrator.AMPLITUDE_MB_TO_MF, amp_mf_to_mb=Arbitrator.AMPLITUDE_MF_TO_MB,
               temperature=Arbitrator.SOFTMAX_TEMPERATURE, rl_learning_rate=SARSA.LEARNING_RATE, performance=300, PARAMETER_SET='DEFAULT',
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
    env     = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE, task_type=TASK_TYPE)
    MDP.DECAY_RATE = DECAY_RATE
    analysis.ACTION_COLUMN = ['action_' + str(action_num) for action_num in range(MDP.NUM_CONTROL_ACTION)]
    analysis.COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'd_p_mb', 'ctrl_reward',
                        'score'] + analysis.ACTION_COLUMN
    if MIXED_RANDOM_MODE:
        random_mode_list = np.random.choice(RANDOM_MODE_LIST, 4 , replace =False) # order among 4 mode is random
        print ('Random Mode Sequence : %s' %random_mode_list)

    # if it is mixed random mode, 'ddpn_loaded' from torch.save(model, filepath) is used instead of 'ddpn'
    writer = SummaryWriter("runs/mish_activation")

    pg.state_dim = _linear_size(env.observation_space[MDP.CONTROL_AGENT_INDEX][0]) + 4
    pg.n_actions = _linear_size(env.action_space[MDP.CONTROL_AGENT_INDEX])
    actor1 = pg.Actor_cate(pg.state_dim, pg.n_actions, activation=pg.Mish)
    actor2 = pg.Actor_cont(pg.state_dim, 1, activation=pg.Mish)
    critic = pg.Critic(pg.state_dim, activation=pg.Mish)
    learner = pg.A2CLearner(actor1, actor2, critic)
    state = None
    done = True
    steps = 0
    episode_reward = 0
    episode_rewards = []
    # ddqn    = DoubleDQN(env.observation_space[MDP.CONTROL_AGENT_INDEX],
    #                     env.action_space[MDP.CONTROL_AGENT_INDEX],
    #                     torch.cuda.is_available(), Session_block=Session_block) # use DDQN for control agent


    gData.new_simulation()
    gData.add_human_data([amp_mf_to_mb / amp_mb_to_mf, rl_learning_rate, estimator_learning_rate, threshold, temperature, performance])
    control_obs_extra = INIT_CTRL_INPUT

    if Reproduce_BHV:
        saved_policy = np.load('history_results/'+saved_policy_path)
    if not RESET:
        # initialize human agent one time
        sarsa   = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX], learning_rate=rl_learning_rate) # SARSA model-free learner
        forward = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                        env.action_space[MDP.HUMAN_AGENT_INDEX],
                        env.state_reward_func, env.output_states_offset, env.reward_map_func,
                            learning_rate=rl_learning_rate, disable_cforward=DISABLE_C_EXTENSION) # forward model-based learner
        arb     = Arbitrator(AssocRelEstimator(estimator_learning_rate, env.max_rpe),
                            BayesRelEstimator(thereshold=threshold),
                            amp_mb_to_mf=amp_mb_to_mf, amp_mf_to_mb=amp_mf_to_mb, temperature=temperature, MB_ONLY = MB_ONLY, MF_ONLY= MF_ONLY)
        # register in the communication controller
        env.agent_comm_controller.register('model-based', forward)

    human_action_list_t = []
    Task_structure = []
    Q_value_forward_t = []
    Q_value_sarsa_t = []
    Q_value_arb_t = []
    Transition_t = []
    prev_cum_reward = 0
    if turn_off_tqdm == False:
        episode_list = tqdm(range(TOTAL_EPISODES))
    else:
        episode_list = range(TOTAL_EPISODES)

    for episode in episode_list:
        #if episode > 99:
        env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE, task_type=TASK_TYPE)
        env.reward_map = env.reward_map_copy.copy()
        env.output_states = env.output_states_copy.copy()
        if RESET:
            # reinitialize human agent every episode
            sarsa   = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX], learning_rate=rl_learning_rate) # SARSA model-free learner
            forward = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                        env.action_space[MDP.HUMAN_AGENT_INDEX],
                        env.state_reward_func, env.output_states_offset, env.reward_map_func,
                            learning_rate=rl_learning_rate, disable_cforward=DISABLE_C_EXTENSION) # forward model-based learner
            arb     = Arbitrator(AssocRelEstimator(estimator_learning_rate, env.max_rpe),
                            BayesRelEstimator(thereshold=threshold),
                            amp_mb_to_mf=amp_mb_to_mf, amp_mf_to_mb=amp_mf_to_mb, MB_ONLY = MB_ONLY, MF_ONLY= MF_ONLY)
            # register in the communication controller
            env.agent_comm_controller.register('model-based', forward)

        if MIXED_RANDOM_MODE and episode % CHANGE_MODE_TERM ==0: # Load ddqn model from exist torch.save every CHANGE_MODE_TERM
            random_mode_index = int(episode/CHANGE_MODE_TERM)
            CONTROL_MODE_temp = random_mode_list[random_mode_index]
            print ('Load DDQN model. Current model : %s Current episode : %s' %(CONTROL_MODE_temp, episode))
            ddqn_loaded = torch.load('ControlRL/' + CONTROL_MODE_temp + '/MLP_OBJ_Subject' + PARAMETER_SET)
            ddqn_loaded.eval() # evaluation mode
            ddqn.eval_Q = ddqn_loaded

        arb.episode_number = episode
        cum_d_p_mb = cum_p_mb = cum_mf_rel = cum_mb_rel = cum_rpe = cum_spe = cum_reward = cum_score = 0
        cum_ctrl_act = np.zeros(MDP.NUM_CONTROL_ACTION)
        human_action_list_episode = []
        memory = []
        for trial in range(TRIALS_PER_EPISODE):
            block_indx = trial // int(TRIALS_PER_EPISODE / 4)
            if trial%TRIALS_PER_EPISODE == 0:
                if episode > 99:
                    env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE,
                              task_type=TASK_TYPE)
                env.reward_map = env.reward_map_copy.copy()
                env.output_states = env.output_states_copy.copy()
            #print(env.reward_map)
            t_d_p_mb = t_p_mb = t_mf_rel = t_mb_rel = t_rpe = t_spe = t_reward = t_score = rpe1 = rpe2  = 0
            game_terminate              = False
            human_obs, control_obs_frag = env.reset()
            control_obs                 = np.append(control_obs_frag, control_obs_extra)
            if Session_block:
                control_obs = np.append(control_obs, int(str(CONTROL_MODES_LIST)[block_indx]))
            """control agent choose action"""
            if episode > 1:
                if Reproduce_BHV:
                    CONTROL_MODE = CONTROL_MODES_LIST
                    CONTROL_MODE_indx = MODE_LIST.index(CONTROL_MODE)
                    control_action = int(saved_policy[CONTROL_MODE_indx][trial])
                elif trial == 0:
                    control_action = [1, 0.95]
                elif Session_block:
                    if int(str(CONTROL_MODES_LIST)[block_indx]) == 5:
                        control_action = random.randrange(0,MDP.NUM_CONTROL_ACTION)
                elif STATIC_CONTROL_AGENT:
                    control_action = static_action_map[CONTROL_MODE][trial]
                else:
                    #control_action = ddqn.action(control_obs)
                    state = control_obs
                    probs = actor1(pg.t(state))
                    dist = torch.distributions.Categorical(probs=probs)
                    control_action[0] = dist.sample().item()
                    dists = actor2(pg.t(state))
                    actions = dists.sample().detach().data.numpy()
                    print([control_action[0],actions])
                    control_action[1] = np.clip(actions, 0.1, 0.9)[0]
                    print(env.reward_map)

                cum_ctrl_act[control_action[0]] += 1
                if TASK_TYPE == 2019:
                    if control_action == 3:
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
                    _, _, _, _ = env.step([MDP.CONTROL_AGENT_INDEX, control_action[0]])
                    env.decay_rate = control_action[1]
                if SAVE_LOG_Q_VALUE:
                    Task_structure.append( np.concatenate((env.reward_map, env.trans_prob, env.output_states), axis=None) )

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
                spe = forward.optimize(human_obs, human_action, next_human_obs)
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
                # t_rpe    += abs(rpe)
                t_rpe += rpe
                t_spe    += spe
                t_score  += human_reward # if not the terminal state, human_reward is 0, so simply add here is fine
                """iterators update"""
                human_obs = next_human_obs
                if current_game_step == 0:
                    # rpe1 = abs(rpe)
                    rpe1 = rpe
                else:
                    # rpe2 = abs(rpe)
                    rpe2 = rpe
                current_game_step += 1

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

            if episode > 99:
                if Session_block:
                    CONTROL_MODE=MODE_LIST[int(str(CONTROL_MODES_LIST)[block_indx])-1]
                    t_reward = error_to_reward((rpe, spe, mf_rel, mb_rel), p_mb, CONTROL_MODE)
                elif MIXED_RANDOM_MODE:
                    t_reward = error_to_reward((rpe, spe, mf_rel, mb_rel), p_mb, CONTROL_MODE_temp)
                else:
                    CONTROL_MODE = CONTROL_MODES_LIST
                    t_reward = error_to_reward((rpe, spe, mf_rel, mb_rel), p_mb, CONTROL_MODE)
                cum_reward += t_reward
                next_control_obs = np.append(next_control_obs_frag, [rpe, spe])
                if Session_block:
                    next_control_obs = np.append(next_control_obs, int(str(CONTROL_MODES_LIST)[block_indx]))
                if not MIXED_RANDOM_MODE and (not Reproduce_BHV): # if it is mixed random mode, don't train ddpn anymore. Using ddqn that is already trained before
                    memory.append((control_action[0], control_action[1], t_reward, control_obs, next_control_obs, done))
                    # memory = [control_action, t_reward, control_obs, next_control_obs, done]
                    steps += 1
                    episode_reward += t_reward
                    episode_rewards.append(episode_reward)
                    learner.learn(memory, steps, discount_rewards=False)
                    #ddqn.optimize(control_obs, control_action, next_control_obs, t_reward)
            else:
                control_action = [1, 0.9]
                t_reward = 0

            control_obs_extra = [rpe, spe]
            detail_col = [rpe, spe, mf_rel, mb_rel, p_mb, d_p_mb, t_reward, t_score] + [control_action] + [rpe1, rpe2]
            if not return_res:
                gData.add_detail_res(trial + TRIALS_PER_EPISODE * episode, detail_col)
            else:
                res_detail_df.loc[trial + TRIALS_PER_EPISODE * episode] = detail_col

            if SAVE_LOG_Q_VALUE:
                Q_value_forward = []
                Q_value_sarsa = []
                Q_value_arb = []
                Transition = []
                #print("#############Task_structure##############")
                #print(np.concatenate((env.reward_map, env.trans_prob,env.output_states), axis=None))
                #print("#############forward_Q##############")
                for state in range(5):
                    #print(state ,forward.get_Q_values(state))
                    Q_value_forward += list(forward.get_Q_values(state))
                #print("#############sarsa_Q##############")
                for state in range(5):
                    #print(state ,sarsa.get_Q_values(state))
                    Q_value_sarsa += list(sarsa.get_Q_values(state))
                #print("#############arb_Q##############")
                for state in range(5):
                    #print(state ,arb.get_Q_values(sarsa.get_Q_values(state),forward.get_Q_values(state)))
                    Q_value_arb += list(arb.get_Q_values(sarsa.get_Q_values(state),forward.get_Q_values(state)))
                #print("#############Transition##############")
                for state in range(5):
                    for action in range(2):
                        #print(state , action, forward.get_Transition(state, action), sum(forward.get_Transition(state, action)))
                        Transition += list(forward.get_Transition(state, action))

                Q_value_forward_t.append(Q_value_forward)
                Q_value_sarsa_t.append(Q_value_sarsa)
                Q_value_arb_t.append(Q_value_arb)
                Transition_t.append(Transition)

                #print(human_action_list_t, human_action_list_t.shape)
                #print(Task_structure, Task_structure.shape)
                '''print(Transition, len(Transition))
                print(Q_value_forward, len(Q_value_forward))
                print(Q_value_sarsa, len(Q_value_sarsa))
                print(Q_value_arb, len(Q_value_arb))'''

        data_col = list(map(lambda x: x / TRIALS_PER_EPISODE,
                                [cum_rpe, cum_spe, cum_mf_rel, cum_mb_rel, cum_p_mb, cum_d_p_mb, cum_reward, cum_score] + list(cum_ctrl_act)))

        if not return_res:
            gData.add_res(episode, data_col)
        else:
            res_data_df.loc[episode] = data_col
        if SAVE_LOG_Q_VALUE:
            human_action_list_t.append(human_action_list_episode)

        if prev_cum_reward <= cum_reward:
            save_NN_A1 = actor1.model
            save_NN_A2 = actor2.model
            save_NN_C = critic.model
        prev_cum_reward = cum_reward

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

        gData.add_log_Q_value(human_action_list_t, Task_structure, Transition_t, Q_value_forward_t, Q_value_sarsa_t, Q_value_arb_t )

    if MIXED_RANDOM_MODE:
        #print (list(random_mode_list))
        gData.add_random_mode_sequence(random_mode_list.tolist())

    if SAVE_CTRL_RL and not MIXED_RANDOM_MODE:
        makedir(RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE)
        #torch.save(ddqn.eval_Q.state_dict(), RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE + '/MLP_OBJ_Subject' + PARAMETER_SET) # save model as dictionary
        torch.save(actor.model, RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE + '/MLP_OBJ_Subject' + PARAMETER_SET) # save entire model
        torch.save(critic.model,
                   RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE + '/MLP_OBJ_Subject' + PARAMETER_SET)  # save entire model
    if ENABLE_PLOT:
        gData.plot_all_human_param(CONTROL_MODE + ' Human Agent State - parameter set: ' + PARAMETER_SET)
        gData.plot_pe(CONTROL_MODE, CONTROL_MODE + ' - parameter set: ' + PARAMETER_SET)
        gData.plot_action_effect(CONTROL_MODE, CONTROL_MODE + ' Action Summary - parameter set: ' + PARAMETER_SET)
    gData.NN = [save_NN_A1, save_NN_A2, save_NN_C]
    gData.complete_simulation()
    if return_res:
        return (res_data_df, res_detail_df)


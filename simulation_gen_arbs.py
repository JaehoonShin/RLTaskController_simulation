""" Shuangyi Tong <s9tong@edu.uwaterloo.ca>
    Sept 17, 2018
"""
import getopt
import sys
import csv
import numpy as np
import dill as pickle # see https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
import random

from tqdm import tqdm
from mdp import MDP

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
from gym import spaces
from functools import reduce
import pickle

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
INIT_CTRL_INPUT       = [0.25, 0.25, 0.25, 0.25]
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
TOTAL_PRE_EPISODES = 100
NUM_ARBS = 100
NUM_SBJ = 82
STORE_DETAILS = False

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

def simulation(*param_list, sbj_indx = 0, PARAMETER_SET='DEFAULT', return_res=False):
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


    env     = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE, task_type=TASK_TYPE)
    MDP.DECAY_RATE = DECAY_RATE
    analysis.ACTION_COLUMN = ['action_' + str(action_num) for action_num in range(MDP.NUM_CONTROL_ACTION)]
    #analysis.COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'd_p_mb', 'ctrl_reward',
    #                    'score'] + analysis.ACTION_COLUMN
    if MIXED_RANDOM_MODE:
        random_mode_list = np.random.choice(RANDOM_MODE_LIST, 4 , replace =False) # order among 4 mode is random
        print ('Random Mode Sequence : %s' %random_mode_list)

    #gData.new_simulation()
    #gData.add_human_data([amp_mf_to_mb / amp_mb_to_mf, rl_learning_rate, estimator_learning_rate, threshold, temperature, performance])
    control_obs_extra = INIT_CTRL_INPUT
    '''
    arb_save = [None] * NUM_ARBS
    MF_estimator_chi_save = [None] * NUM_ARBS
    MB_estimator_categories_save = [None] * NUM_ARBS
    MB_estimator_pe_records_counter_save = [None] * NUM_ARBS
    MB_estimator_pe_records_save = [None] * NUM_ARBS
    MB_estimator_target_category_save = [None] * NUM_ARBS
    sarsa_save = [None] * NUM_ARBS
    forward_save = [None] * NUM_ARBS
    '''
    if Reproduce_BHV:
        saved_policy = np.load('history_results/'+saved_policy_path)


    LEN_TOT_EPS = NUM_SBJ*NUM_ARBS*TOTAL_PRE_EPISODES+NUM_SBJ*NUM_ARBS*TOTAL_EPISODES
    #gData.fast_initialization_DATA(LEN_TOT_EPS)
    #if STORE_DETAILS:
    #    gData.fast_initialization_DETAIL(LEN_TOT_EPS * TRIALS_PER_EPISODE)
    'pre-training start'
    print(sbj_indx)
    estimator_learning_rate = param_list[sbj_indx][0]
    threshold = param_list[sbj_indx][1]
    amp_mb_to_mf = param_list[sbj_indx][2]
    amp_mf_to_mb = param_list[sbj_indx][3]
    rl_learning_rate = param_list[sbj_indx][4]
    temperature = param_list[sbj_indx][5]
    performance = param_list[sbj_indx][6]
    tag = param_list[sbj_indx][7]
    for arb_indx in range(NUM_ARBS):
        print('pre-training sbj: ' + str(sbj_indx) + ' with rep ' + str(arb_indx))
        if not RESET:
            # initialize human agent one time
            sarsa = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX], env, learning_rate=rl_learning_rate,
                        nine_states_mode=env.nine_states_mode)  # SARSA model-free learner
            forward = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                              env.action_space[MDP.HUMAN_AGENT_INDEX],
                              env.state_reward_func, env.output_states_offset, env.reward_map_func,
                              learning_rate=rl_learning_rate, disable_cforward=DISABLE_C_EXTENSION,
                              output_state_array=  env.output_states_indx, nine_states_mode = env.nine_states_mode)
                            # forward model-based learner
            arb = Arbitrator(AssocRelEstimator(estimator_learning_rate, env.max_rpe),
                             BayesRelEstimator(thereshold=threshold),
                             amp_mb_to_mf=amp_mb_to_mf, amp_mf_to_mb=amp_mf_to_mb, temperature=temperature,
                             MB_ONLY=MB_ONLY, MF_ONLY=MF_ONLY)
            # register in the communication controller
            env.agent_comm_controller.register('model-based', forward)

        human_action_list_t = []

        if turn_off_tqdm == False:
            episode_list_pre = tqdm(range(TOTAL_PRE_EPISODES))
        else:
            episode_list_pre = range(TOTAL_PRE_EPISODES)

        for episode in episode_list_pre:
            #if episode > 99:
            env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE, task_type=TASK_TYPE)
            env.reward_map = env.reward_map_copy.copy()
            env.output_states = env.output_states_copy.copy()
            if RESET:
                # reinitialize human agent every episode
                sarsa   = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX], env, learning_rate=rl_learning_rate,
                        nine_states_mode=env.nine_states_mode) # SARSA model-free learner
                forward = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                            env.action_space[MDP.HUMAN_AGENT_INDEX],
                            env.state_reward_func, env.output_states_offset, env.reward_map_func,
                            learning_rate=rl_learning_rate, disable_cforward=DISABLE_C_EXTENSION,
                            output_state_array=  env.output_states_indx, nine_states_mode = env.nine_states_mode)# forward model-based learner
                arb     = Arbitrator(AssocRelEstimator(estimator_learning_rate, env.max_rpe),
                                BayesRelEstimator(thereshold=threshold), temperature=temperature,
                                amp_mb_to_mf=amp_mb_to_mf, amp_mf_to_mb=amp_mf_to_mb, MB_ONLY = MB_ONLY, MF_ONLY= MF_ONLY)
                # register in the communication controller
                env.agent_comm_controller.register('model-based', forward)

            arb.episode_number = episode
            arb.CONTROL_resting = CONTROL_resting
            cum_d_p_mb = cum_p_mb = cum_mf_rel = cum_mb_rel = cum_rpe = cum_spe = cum_reward = cum_score = cum_real_rwd = 0
            cum_ctrl_act = np.zeros(MDP.NUM_CONTROL_ACTION)
            human_action_list_episode = []
            memory = []
            for trial in range(TRIALS_PER_EPISODE):
                block_indx = trial // int(TRIALS_PER_EPISODE / 4)
                if episode <= CONTROL_resting:
                    env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE,
                              task_type=TASK_TYPE)
                #print(env.reward_map)
                t_d_p_mb = t_p_mb = t_mf_rel = t_mb_rel = t_rpe = t_spe = t_reward = t_score = rpe1 = rpe2  = 0
                game_terminate              = False
                human_obs, control_obs_frag = env.reset()
                control_obs                 = np.append(control_obs_frag, control_obs_extra)

                """control agent choose action"""

                forward.bwd_update(env.bwd_idf,env)
                current_game_step = 0
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

                control_action = random.randrange(1,MDP.NUM_CONTROL_ACTION+1)
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
                t_reward = 0
                real_rwd = t_reward
                cum_real_rwd += real_rwd

                control_obs_extra = INIT_CTRL_INPUT
            if episode == TOTAL_PRE_EPISODES - 1:
                #arb_save[arb_indx] = arb
                MF_estimator_chi_save = arb.mf_rel_estimator.chi
                MB_estimator_categories_save = arb.mb_rel_estimator.categories
                MB_estimator_pe_records_counter_save = arb.mb_rel_estimator.pe_records_counter
                MB_estimator_pe_records_save = arb.mb_rel_estimator.pe_records
                MB_estimator_target_category_save = arb.mb_rel_estimator.target_category
                Q_sarsa_tmp = np.zeros((len(sarsa.Q_sarsa),sarsa.num_actions))
                for states in range(len(sarsa.Q_sarsa)):
                    Q_sarsa_tmp[states] = sarsa.Q_sarsa[states]
                sarsa_save = Q_sarsa_tmp
                forward_save = forward.T
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
                filename = 'history_results/sbjs/sbj-{0:02d}-rep-{1:02d}.pkl'.format(sbj_indx,arb_indx)
                with open(filename,'wb') as f:
                    pickle.dump(data,f)

    'pre-training completed'


if __name__ == '__main__':
    short_opt = "hdn:"
    long_opt = ["help", "trials=", "episodes=", "file-suffix=", "task-type=",
                "PMB_CONTROL=", "Reproduce_BHV=", 'delta-control=', 'ablation']
    try:
        opts, args = getopt.getopt(sys.argv[1:], short_opt, long_opt)
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    LOAD_PARAM_FILE = True
    PARAMETER_FILE = 'regdata_lb.csv'
    ENABLE_PLOT = False
    for o, a in opts:
        if o in ("-h", "--help"):
            sys.exit()
        elif o == "--episodes":
            TOTAL_PRE_EPISODES = int(a)
        elif o == "--trials":
            TRIALS_PER_EPISODE = int(a)
        elif o == "-n":
            NUM_PARAMETER_SET = int(a)
        elif o == "--file-suffix":
            FILE_SUFFIX = a
            print(FILE_SUFFIX)
        elif o == "--task-type":
            TASK_TYPE = int(a)
        elif o == "--PMB_CONTROL":
            PMB_CONTROL = bool(int(a))
        elif o == "--Reproduce_BHV":
            Reproduce_BHV = bool(int(a))
            if Reproduce_BHV:
                saved_policy_path = 'optimal_policy' + FILE_SUFFIX + '.npy'
                FILE_SUFFIX = FILE_SUFFIX + '_repro'
        elif o == "--delta-control":  # RPE control task, can change hyperparameter delta for task controller
            DECAY_RATE = float(a)
            FILE_SUFFIX = FILE_SUFFIX + '_delta_control'
        else:
            assert False, "unhandled option"

    if LOAD_PARAM_FILE:
        print(PARAMETER_FILE)
        with open(PARAMETER_FILE) as f:
            csv_parser = csv.reader(f)
            param_list = []
            for row in csv_parser:
                param_list.append(tuple(map(float, row)))
            print(TRIALS_PER_EPISODE)
            #            for index in range(NUM_PARAMETER_SET):
            index = NUM_PARAMETER_SET
            simulation(*(param_list), sbj_indx=NUM_PARAMETER_SET, PARAMETER_SET=str(index))
            #            gData.generate_summary(sim.CONTROL_MODE)



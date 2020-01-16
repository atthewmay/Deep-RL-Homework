"""
Original code from John Schulman for CS294 Deep Reinforcement Learning Spring 2017
Adapted for CS294-112 Fall 2017 by Abhishek Gupta and Joshua Achiam
Adapted for CS294-112 Fall 2018 by Michael Chang and Soroush Nasiriany
"""
import numpy as np
import gym
import logz
import os
import time
import inspect
from multiprocessing import Process
import multiprocessing
import math as m
import ipdb as pdb
import sys
sys.path.append("../../../gravity_ball_game/")
parent_dir = (os.path.abspath('../../../gravity_ball_game'))
os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")
from gravity_ball_game_training_simulator import GB_game
from forked_pdb import ForkedPdb
import tensorflow as tf
import psutil
from parallel_actor import Parallel_Actor, Counter
from train_pg_f18 import Agent, pathlength
import ray
tf.logging.set_verbosity(tf.logging.ERROR)
#============================================================================================#
# Utilities
#============================================================================================#

#========================================================================================#
#                           ----------PROBLEM 2----------
#========================================================================================#  
def build_mlp(input_placeholder, output_size, scope, n_layers, size,activation=tf.tanh, output_activation=None):
    """
        Builds a feedforward neural network
        
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            output_size: size of the output layer
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of the hidden layer
            activation: activation of the hidden layers
            output_activation: activation of the ouput layers

        returns:
            output placeholder of the network (the result of a forward pass) 

        Hint: use tf.layers.dense    
    """
    activation = [activation]
    if not isinstance(size,list):
        size = [size]

    fc_layer = input_placeholder
    with tf.variable_scope(scope):
        for i in range(n_layers-1): # Note it's only going to work for 1 layer.
            fc_layer = tf.contrib.layers.fully_connected(fc_layer, size[i],weights_regularizer=tf.contrib.layers.l2_regularizer(0.05),activation_fn=activation[i])

        output_placeholder = tf.contrib.layers.fully_connected(fc_layer, output_size,
                weights_regularizer=tf.contrib.layers.l2_regularizer(0.05),
                activation_fn=output_activation)

    # raise NotImplementedError
    return output_placeholder


def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

#============================================================================================#
# Policy Gradient
#============================================================================================#
# 
def train_PG(
        exp_name,
        env_name,
        n_iter, 
        gamma, 
        min_timesteps_per_batch, 
        max_path_length,
        learning_rate,
        baseline_lr, 
        reward_to_go, 
        animate, 
        logdir, 
        normalize_advantages,
        nn_baseline, 
        seed,
        n_layers,
        output_activation,
        size,
        save_models,
        save_best_model,
        resume_string,
        run_model_only,
        script_optimizing_dir,
        parallel,
        relative_positions,
        death_penalty,
        reward_circle,
        num_enemies,
        gb_discrete,
        gb_max_speed):

    start = time.time()
    if script_optimizing_dir is not None:
        logdir = logdir[:5]+script_optimizing_dir+'/'+logdir[5:]

    #========================================================================================#
    # Set Up Logger
    #========================================================================================#
    setup_logger(logdir, locals())

    #========================================================================================#
    # Set Up Env
    #========================================================================================#

    # Make the gym environment
    if env_name == 'GB_game':
        env = GB_game(num_char = num_enemies, reward_circle = reward_circle, death_penalty = death_penalty, relative_positions = relative_positions, discrete=gb_discrete, max_speed=gb_max_speed)
        discrete = env.discrete
        if parallel == True:
            ray.register_custom_serializer(GB_game, use_pickle=True) # amazing. I needed to use this to get it to
            put_env = ray.put(env)
    else:
        env = gym.make(env_name)
        # Is this env continuous, or self.discrete?
        discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    # pdb.set_trace()
    env.seed(seed)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps


    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    #========================================================================================#
    # Initialize Agent
    #========================================================================================#
    computation_graph_args = {
        'n_layers': n_layers,
        'output_activation': output_activation,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'discrete': discrete,
        'size': size,
        'learning_rate': learning_rate,
        'baseline_lr' : baseline_lr,
        }

    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }

    estimate_return_args = {
        'gamma': gamma,
        'reward_to_go': reward_to_go,
        'nn_baseline': nn_baseline,
        'normalize_advantages': normalize_advantages,
    }

    if parallel is True:
        num_cpus = psutil.cpu_count(logical=True)
        num_cpus = num_cpus-1
        print('the number of cpus is now' + str(num_cpus))
        ray.init(num_cpus=num_cpus,ignore_reinit_error=True)
        pathlen_counter = Counter.remote()
        parallel_actors = [Parallel_Actor.remote(computation_graph_args, sample_trajectory_args, estimate_return_args) for _
                           in range(num_cpus)]
        agent = Parallel_Actor.remote(computation_graph_args, sample_trajectory_args, estimate_return_args)
        # This is the one used for updating the weights
            
    else:
        agent = Agent(computation_graph_args, sample_trajectory_args, estimate_return_args)

        # build computation graph
        agent.build_computation_graph()

        # tensorflow: config, session, variable initialization
        agent.init_tf_sess()

    # Now we'll try to load if we are only running a model or if we are resuming training.
    if run_model_only is not None:
        agent.load_models_action(run_model_only)
        agent.running_only = True
    elif resume_string is not None:
        agent.load_models_action(resume_string)


    #setup for a parallel training loader.
    #========================================================================================#
    # Training Loop
    #========================================================================================#
    best_avg_return = -(5e10)
    total_timesteps = 0
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)
        if parallel is True:
            pathlen_counter.reset_counter.remote()
            weights_copy = agent.get_weights.remote()
            ray.get([p_agent.set_weights.remote(weights_copy) for p_agent in parallel_actors])

            weights = ray.get([p_agent.get_weights.remote() for p_agent in parallel_actors])
            for i in range(len(weights)):
                np.testing.assert_equal(weights[i], weights[0])
            print('\n \n the weights have successfully been reset!! \n \n')

            paths = []
            agent_outputs = []
            for p_agent in parallel_actors:# Note this is not parallel! yet.
                agent_outputs.append(p_agent.sample_trajectories.remote(itr, put_env, pathlen_counter))
            for output in agent_outputs:
                path_set, timesteps_this_batch = ray.get(output)#Gotta use pathset
                #Question: Would it be faster to do a self.env structure for parallel agents?
                [paths.append(path) for path in path_set]
                total_timesteps += timesteps_this_batch
                # wow so it's really helpful the paths come in contiguous segments.

        else:
            paths, timesteps_this_batch = agent.sample_trajectories(itr, env)

        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        # Note that estimate_return could also be parallelized.  
        if run_model_only is not None:
            continue
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = [path["reward"] for path in paths]
        if parallel:
            q_n, adv_n = ray.get(agent.estimate_return.remote(ob_no, re_n))
            agent.update_parameters.remote(ob_no, ac_na, q_n, adv_n)
        else:
            q_n, adv_n = agent.estimate_return(ob_no, re_n)
            agent.update_parameters(ob_no, ac_na, q_n, adv_n)

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        mean_return = np.mean(returns)
        if mean_return > best_avg_return:
            best_avg_return = mean_return
            if save_best_model==True:
                save_string = logdir[5:-2]
                if parallel:
                    agent.save_models_action.remote(save_string)
                else:
                    agent.save_models_action(save_string)
        logz.log_tabular("AverageReturn", mean_return)
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # My own
        if parallel is False:
            if hasattr(agent,'batch_baseline_loss'):
                logz.log_tabular("BaselineLoss", agent.batch_baseline_loss)
            logz.log_tabular("UnscaledLoss", agent.batch_unscaled_loss)
            logz.log_tabular("Loss", agent.batch_loss)


        logz.dump_tabular()
        logz.pickle_tf_vars()

        # if script_optimizing == True:
        #     print(np.max(returns))
        # One potential issue here is that there won't be a local for the first iteration. we must make it
        # so.

    if save_models == True and save_best_model==False:
        save_string = logdir[5:-2]
        if parallel:
            agent.save_models_action.remote(save_string)
        else:
            agent.save_models_action(save_string)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--baseline_lr', '-bllr', type=float, default=None)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--output_activation', type=str, default = None)
    parser.add_argument('--size', '-s', type=int, default=64)
    #I'm adding this one for my own edification
    parser.add_argument('--save_models', action = 'store_true')
    parser.add_argument('--save_best_model', action = 'store_true')
    parser.add_argument('--resume_string', type = str, default = None) # put the model name that you will resume training from!
    parser.add_argument('--run_model_only', type = str, default = None) # This is a string with the model savefile
    parser.add_argument('--script_optimizing_dir', type = str, default = None) # use this if doing a bash_script method
    parser.add_argument('--parallel', action = 'store_true')

    # These 3 are for my game only!
    parser.add_argument('--relative_positions', '-rp', action='store_true')
    parser.add_argument('--death_penalty', '-dp', action='store_true')
    parser.add_argument('--reward_circle', '-rc', action='store_true')
    parser.add_argument('--num_enemies', type=int, default = 1)
    parser.add_argument('--gb_discrete', action='store_true')
    parser.add_argument('--gb_max_speed', type=int, default=20)



    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    processes = []

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)

        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                baseline_lr=args.baseline_lr,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline, 
                seed=seed,
                n_layers=args.n_layers,
                output_activation = args.output_activation,
                size=args.size,
                save_models = args.save_models,
                save_best_model = args.save_best_model,
                resume_string = args.resume_string,
                run_model_only = args.run_model_only,
                script_optimizing_dir=args.script_optimizing_dir,
                parallel=args.parallel,
                relative_positions = args.relative_positions, # These 3 are only for the game!
                death_penalty=args.death_penalty,
                reward_circle=args.reward_circle,
                num_enemies=args.num_enemies,
                gb_discrete=args.gb_discrete,
                gb_max_speed=args.gb_max_speed
                )
        # # Awkward hacky process runs, because Tensorflow does not like
        # # repeatedly calling train_PG in the same thread.
    #     if args.render == False:
    #         p = Process(target=train_func, args=tuple())
    #         p.start()
    #         processes.append(p)
    #         # if you comment in the line below, then the loop will block 
    #         # until this process finishes
    #         # p.join()

    # if args.render == False:
    #     for p in processes:
    #         p.join()

    # else:
    train_func() # OH MY GOODNESS! The Render doesn't work if the above isn't commented out, and this line replacing it. Must use this line to render.
if __name__ == "__main__":
    main()


# you add new args to the code by putting a new arg in 3 different places.

def run_pg_GB(
        n_iter, 
        min_timesteps_per_batch, 
        max_path_length,
        animate, 
        logdir, 
        nn_baseline, 
        seed,
        n_layers,
        output_activation,
        size,
        save_models,
        save_best_model,
        run_model_only,
        script_optimizing_dir,
        relative_positions,
        death_penalty,
        reward_circle,
        num_enemies):

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
    env = GB_game(num_char = num_enemies, reward_circle = reward_circle, death_penalty = death_penalty, relative_positions = relative_positions)
    
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # Is this env continuous, or self.discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

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

    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_return_args)

    # build computation graph
    agent.build_computation_graph()

    # tensorflow: config, session, variable initialization
    agent.init_tf_sess()

    # Now we'll try to load...
    if run_model_only is not None:
        agent.load_models_action(run_model_only)
        agent.running_only = True
    #========================================================================================#
    # Training Loop
    #========================================================================================#
    best_avg_return = -(5e10)
    total_timesteps = 0
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        if run_model_only is not None:
            continue
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = [path["reward"] for path in paths]

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
        if hasattr(agent,'batch_baseline_loss'):
            logz.log_tabular("BaselineLoss", agent.batch_baseline_loss)
        logz.log_tabular("UnscaledLoss", agent.batch_unscaled_loss)
        logz.log_tabular("Loss", agent.batch_loss)


        logz.dump_tabular()
        logz.pickle_tf_vars()

        # if script_optimizing == True:
        #     print(np.max(returns))

    if save_models == True and save_best_model==False:
        save_string = logdir[5:-2]
        agent.save_models_action(save_string)
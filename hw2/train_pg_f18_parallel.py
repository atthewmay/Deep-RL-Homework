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
from gravity_ball_game_training_simulator import GB_game
from forked_pdb import ForkedPdb

#============================================================================================#
# Utilities
#============================================================================================#

#========================================================================================#
#                           ----------PROBLEM 2----------
#========================================================================================#  
def build_mlp(input_placeholder, output_size, scope, n_layers, size, activation='tf.tanh', output_activation=None):
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
    activation = [exec(activation)]
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

def pathlength(path):
    return len(path["reward"])

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

class Agent(object):
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_return_args):
        super(Agent, self).__init__()
        import tensorflow as tf
        self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_dim']
        self.discrete = computation_graph_args['discrete']
        self.size = computation_graph_args['size']
        self.n_layers = computation_graph_args['n_layers']
        self.output_activation = computation_graph_args['output_activation']
        self.learning_rate = computation_graph_args['learning_rate']
        self.baseline_lr = computation_graph_args['baseline_lr']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']

        self.gamma = estimate_return_args['gamma']
        self.reward_to_go = estimate_return_args['reward_to_go']
        self.nn_baseline = estimate_return_args['nn_baseline']
        self.normalize_advantages = estimate_return_args['normalize_advantages']

    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__() # equivalent to `with self.sess:`
        tf.global_variables_initializer().run() #pylint: disable=E1101

    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#
    def define_placeholders(self):
        """
            Placeholders for batch batch observations / actions / advantages in policy gradient 
            loss function.
            See Agent.build_computation_graph for notation

            returns:
                sy_ob_no: placeholder for observations
                sy_ac_na: placeholder for actions
                sy_adv_n: placeholder for advantages
        """
        # raise NotImplementedError
        sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        if self.discrete:
            sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) 
        else:
            sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32) 
        # YOUR CODE HERE
        sy_adv_n = tf.placeholder(shape = [None], name = 'adv',dtype = tf.float32)
        return sy_ob_no, sy_ac_na, sy_adv_n


    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#
    def policy_forward_pass(self, sy_ob_no):
        """ Constructs the symbolic operation for the policy network outputs,
            which are the parameters of the policy distribution p(a|s)

            arguments:
                sy_ob_no: (batch_size, self.ob_dim)

            returns:
                the parameters of the policy.

                if discrete, the parameters are the logits of a categorical distribution
                    over the actions
                    sy_logits_na: (batch_size, self.ac_dim)

                if continuous, the parameters are a tuple (mean, log_std) of a Gaussian
                    distribution over actions. log_std should just be a trainable
                    variable, not a network output.
                    sy_mean: (batch_size, self.ac_dim)
                    sy_logstd: (self.ac_dim,)

            Hint: use the 'build_mlp' function to output the logits (in the discrete case)
                and the mean (in the continuous case).
                Pass in self.n_layers for the 'n_layers' argument, and
                pass in self.size for the 'size' argument.
        """
        # raise NotImplementedError
        if self.discrete:
            # YOUR_CODE_HERE
            sy_logits_na = build_mlp(sy_ob_no, self.ac_dim, 'mlp', self.n_layers, self.size, activation=tf.tanh, output_activation=exec(self.output_activation))
            # Right, we want output activation = none, bc these are "logits", which in tf-language means the unscaled inputs to the softmax function
            return sy_logits_na
        else:
            # YOUR_CODE_HERE
            sy_mean = build_mlp(sy_ob_no, self.ac_dim, 'mlp', self.n_layers, self.size, activation=tf.tanh, output_activation=exec(self.output_activation))
            # Note this will be a vector of means of length self.ac_dim
            sy_logstd = tf.get_variable(name = 'std_vec', shape = self.ac_dim)
            # weird! We'll separately train the std, not letting it depend on the network. Seems maybe there should be a separate network for this guy.
            return (sy_mean, sy_logstd)

    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#
    def sample_action(self, policy_parameters):
        """ Constructs a symbolic operation for stochastically sampling from the policy
            distribution

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions 
                        sy_logits_na: (batch_size, self.ac_dim)
                    if continuous: (mean, log_std) of a Gaussian distribution over actions
                        sy_mean: (batch_size, self.ac_dim)
                        sy_logstd: (self.ac_dim,)

            returns:
                sy_sampled_ac: 
                    if discrete: (batch_size,)
                    if continuous: (batch_size, self.ac_dim)

            Hint: for the continuous case, use the reparameterization trick:
                 The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
        
                      mu + sigma * z,         z ~ N(0, I). ### Question: What the heck? How do I prove this? DO IT LATER. It makes intuitive sense!
        
                 This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
        """
        # raise NotImplementedError
        if self.discrete:
            sy_logits_na = policy_parameters
            # YOUR_CODE_HERE
            sy_sampled_ac = tf.squeeze(tf.multinomial(sy_logits_na,1),axis = 1) # only wanna squeeze out one dimension
            # Question: should there be a tf.log in here for these unscaled logits? I feel 
            # So what I've found out: if you do samples = tf.multinomial(tf.log([[0.4,0.5,0.1]]),100000), you get samples in proportion to the probabilities therein
            # OKAY: tf.multinomial(tf.log([[x1,x2,x3]])) = [x1/sum(x),x2/sum(x)...] probability distribution.
            # So therefore tf.multinomial([[a1,a2,a3]]) = [e^a1/sum(e^a),...] distribution. So yes, just input the raw outputs. 
            #it takes the exponential of each entry and makes a prob dist of those.



        else:
            sy_mean, sy_logstd = policy_parameters
            sy_sampled_ac = tf.add(sy_mean,tf.multiply(sy_logstd,tf.random_normal(tf.shape(sy_mean)))) # note I'm treating sy_logstd as std, not log(std)

        return sy_sampled_ac

    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#
    def get_log_prob(self, policy_parameters, sy_ac_na):
        """ Constructs a symbolic operation for computing the log probability of a set of actions
            that were actually taken according to the policy

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions 
                        sy_logits_na: (batch_size, self.ac_dim)
                    if continuous: (mean, log_std) of a Gaussian distribution over actions
                        sy_mean: (batch_size, self.ac_dim)
                        sy_logstd: (self.ac_dim,)

                sy_ac_na: 
                    if discrete: (batch_size,)
                    if continuous: (batch_size, self.ac_dim)

            returns:
                sy_logprob_n: (batch_size)

            Hint:
                For the discrete case, use the log probability under a categorical distribution.
                For the continuous case, use the log probability under a multivariate gaussian. 
                Question: Won't the probability be approaching zero for any set of actions? and thus log_prob = -inf?
                I think we are substituting pdf for probability here. Odd.

                AH! It doesn't matter! What we care about ultimately is the grad(log_prob(at|st)), 
                and dlog_p/dtheta = dlog_p/dlog_pdf * dlog_pdf/dtheta = 1 * dlog_pdf/dtheta (prove dlog_p/dlog_pdf = 1, SOME-OTHER-TIME!!!) 
                But in this case, we can just get log_pdf and later when gradient is taken it = gradient of log_p
                
                KEY: I am pretty certain that by saying sy_logstd [=] [action_space_dim, ], that means we are using an identity for covariance matrix
                So therefore we assume each dimension of the gaussian is independent from the others and thus total_pdf = pdf_dim1*pdf_dim2 * ... * pdfdim_n
                
                Or is this just saying that we don't know the covariance matrix???

                I'm pretty sure this is a messed up assumption, bc the dimensions of the output action are definitely not independent, as they share
                most of a neural network in common. Thus I think this wouldn't actually give the correct gradient to maximize the probability of the
                action taken. Must be close enough?

        Notes: I believe the probability for an action is given by the softmax function. That's how tf.multinomial interpreted those "logits" inputs
        """
        # raise NotImplementedError
        if self.discrete:
            sy_logits_na = policy_parameters
            softmaxed_logits = tf.nn.softmax(sy_logits_na) #gives [batch_size,action_space] vector 
            # For each entry in batch, select the appropriate chosen action. sy_ac_na is [batch_size,]
            indexer = tf.stack([tf.range(0,tf.shape(sy_ac_na)[0],1), sy_ac_na], axis = 1) # Makes the [[0,a0],[1,a1],...] array
            probs_of_chosen_actions = tf.gather_nd(softmaxed_logits,indexer) # gets the responsible action in each row. vector is [batch,] = [p_a1 p_a2 p_a3 ...]
            # each element of indexer ([k, a_k]) selects the k row and a_k column of softmaxed_logits
            sy_logprob_n = tf.log(probs_of_chosen_actions) # So flame...
            # But this entire method is less stable than softmax_cross_entropy_with_logits... Lunar lander before did it the same way... RIGHT?
        else:
            sy_mean, sy_logstd = policy_parameters
            # What needs to happen is I need to take these chosen sy_means, which are [batches,action_space], and get the probability of each action 
            # in each batch sample, using the sy_mean, which is [batches,action_space] and sy_logstd, which is [action_space, ]
            # I then multiple the entire row of probabilities to get the total probability, and then take the log of that. Tomorrow;)
            sigma_square = tf.square(sy_logstd)
            diff_mat = tf.subtract(sy_ac_na,sy_mean)
            two_pi = tf.constant(2*m.pi,dtype = tf.float32)
            first_term = tf.divide(tf.cast(1,tf.float32),tf.sqrt(tf.multiply(two_pi,sigma_square)))
            second_term = tf.exp(tf.negative(tf.divide(tf.square(diff_mat),tf.multiply(tf.cast(2,tf.float32),sigma_square))))
            pdf_output = tf.multiply(first_term,second_term)

            log_pdf = tf.log(pdf_output)
            sy_logprob_n = tf.reduce_sum(log_pdf,1) # we use sum, bc sum(log_prob) = log(mult(all_probs))

        return sy_logprob_n

    def build_computation_graph(self):
        """
            Notes on notation:
            
            Symbolic variables have the prefix sy_, to distinguish them from the numerical values
            that are computed later in the function
            
            Prefixes and suffixes:
            ob - observation 
            ac - action
            _no - this tensor should have shape (batch self.size /n/, observation dim)
            _na - this tensor should have shape (batch self.size /n/, action dim)
            _n  - this tensor should have shape (batch self.size /n/)
            
            Note: batch self.size /n/ is defined at runtime, and until then, the shape for that axis
            is None

            ----------------------------------------------------------------------------------
            loss: a function of self.sy_logprob_n and self.sy_adv_n that we will differentiate
                to get the policy gradient.
        """
        self.sy_ob_no, self.sy_ac_na, self.sy_adv_n = self.define_placeholders()
        # The policy takes in an observation and produces a distribution over the action space
        self.policy_parameters = self.policy_forward_pass(self.sy_ob_no)

        # We can sample actions from this action distribution.
        # This will be called in Agent.sample_trajectory() where we generate a rollout.
        self.sy_sampled_ac = self.sample_action(self.policy_parameters)

        # We can also compute the logprob of the actions that were actually taken by the policy
        # This is used in the loss function.
        self.sy_logprob_n = self.get_log_prob(self.policy_parameters, self.sy_ac_na)

        #========================================================================================#
        #                           ----------PROBLEM 2----------
        # Loss Function and Training Operation
        #========================================================================================#
        self.unscaled_loss = -tf.reduce_mean(self.sy_logprob_n) # This loss is just the log(a|s), not scaled by the r(path). 
        # This is a (+) number. 

        self.s_scaledlogprob_n = tf.multiply(self.sy_logprob_n,self.sy_adv_n) # this assumes the sy_adv_n is taking the total reward or reward to go at each timestep.
        self.loss = -tf.reduce_mean(self.s_scaledlogprob_n) # YOUR CODE HERE
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        #========================================================================================#
        #                           ----------PROBLEM 6----------
        # Optional Baseline
        #
        # Define placeholders for targets, a loss function and an update op for fitting a 
        # neural network baseline. These will be used to fit the neural network baseline. 
        # amazing that we use an entirely separate network to learn the baseline reward prediction!
        #========================================================================================#
        if self.nn_baseline:
            self.baseline_prediction = tf.squeeze(build_mlp(
                                    self.sy_ob_no, 
                                    1, 
                                    "nn_baseline",
                                    n_layers=self.n_layers,
                                    size=self.size))
            # YOUR_CODE_HERE
            self.sy_target_n = tf.placeholder(shape=[None], name="target_n", dtype=tf.int32) 

            self.baseline_loss = tf.losses.mean_squared_error(self.sy_target_n, self.baseline_prediction) # because this is a continuous value, lets just use sum of squared error. or mean of sqare err
            if self.baseline_lr is not None:
                self.baseline_update_op = tf.train.AdamOptimizer(self.baseline_lr).minimize(self.baseline_loss)
            else:
                self.baseline_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.baseline_loss)


    def parallel_sampler(self, env, temp_save_string):
        """okay, so the plan will be to spawn processes = cpu_number, then have
        2 queues. One will store paths and the other will store path lengths.

        I forsee an issue: if I use .join() before doing backprop, then the
        queue might overload. Well I guess just use that after queue.get() but
        before your backprop.

        New plan. gotta use Value() instead of q_len.:"""
        cpu_count = multiprocessing.cpu_count()
        q_path = multiprocessing.Queue()
        v_pathlen = multiprocessing.Value('i',0)
        paths = []
        processes = []
        for i in range(1):
            job = multiprocessing.Process(name='p_'+str(i),target=self.sample_t_parallel,
                                          args=(env,q_path,v_pathlen,temp_save_string))
            print('starting job on cpu ' + str(i))
            processes.append(job)
            job.start()
        
        proc_alive = True
        while proc_alive == True:#this holds it up till all the processes finish, and then we move on into
        # emptying the residual paths
            alive_list = [proc.is_alive() for proc in processes]
            proc_alive = any(alive_list) # will tell me if there's at least one process still living.
        while not q_path.empty():
            paths.append(q_path.get())
        timesteps_this_batch = v_pathlen.value
        for job in processes:
            job.join()
        return paths, timesteps_this_batch

    def sample_t_parallel(self,env,q_path,q_len,temp_save_string):
        import tensorflow as tf
        timesteps_this_batch = 0
        # First thing that needs to happen is loading the model.
        # All of this will take place within a new session
        ForkedPdb().set_trace()
        with tf.Session() as custom_sess:
            saver = tf.train.Saver()
            saver.restore(custom_sess,temp_save_string) # still this doesn't work. I suspect tensorflow is
            #imported in the wrong location! 
            print('model loaded from '+temp_save_string)

            while True:
                animate_this_episode=False # we won't animate parallel
                # We are going to use ForkedPdb on this portion to start digging into it. 
                path = self.sample_trajectory(env, animate_this_episode)
                print('path good') #apparently itn's not good!
                q_path.put(path)
                with multiprocessing.Lock():
                    v_pathlen.value += pathlength(path)
                print('the value of v_pathlen.value is '+str(v_pathlen.value) + ' on '+str(multiprocessing.current_process().name))
                
                if v_pathlen.value > self.min_timesteps_per_batch:
                    print("v_pathlen = " + str(v_pathlen.value))
                    break

    def sample_trajectories(self, itr, env):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            if hasattr(self,'running_only') and self.animate:
                animate_this_episode=True
            
            

            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch #

    def parallel_trajectory(self, env, queue):
        """ so Paths is a list of dictionaries.
        Because of this, and because advantage and summer for awards is actually competed later, you will not be able to put individual observations into the queue
        Instead we will need to add entire path dictionaries to the queue. This really shouldn't pose too much of an issue.
        
        Actually, it might be simpler to have the paralyzing outside of the sample_trajectories. """
        path = self.sample_trajectory
        queue.put(path)

    def sample_trajectory(self, env, animate_this_episode, custom_sess):
        ob = env.reset()
        obs, acs, rewards = [], [], []
        steps = 0
        while True:
            if animate_this_episode:
                # pdb.set_trace()
                env.render()
                time.sleep(0.01)
            # pdb.set_trace()
            obs.append(ob)
            #====================================================================================#
            #                           ----------PROBLEM 3----------
            #====================================================================================#
            # raise NotImplementedError
            ForkedPdb().set_trace()
            if custom_sess is not None:
                ac = custom_sess.run(self.sy_sampled_ac,feed_dict = {self.sy_ob_no:[ob]})
            else:
                ac = self.sess.run(self.sy_sampled_ac,feed_dict = {self.sy_ob_no:[ob]})
            ac = ac[0]
            acs.append(ac)
            ob, rew, done, _ = env.step(ac)
            rewards.append(rew)
            steps += 1
            if done or steps > self.max_path_length:
                break
        path = {"observation" : np.array(obs, dtype=np.float32), 
                "reward" : np.array(rewards, dtype=np.float32), 
                "action" : np.array(acs, dtype=np.float32)}
        return path

    #====================================================================================#
    #                           ----------PROBLEM 3----------
    #====================================================================================#
    def sum_of_rewards(self, re_n):
        """
            Monte Carlo estimation of the Q function.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from 
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                re_n: length: num_paths. Each element in re_n is a numpy array 
                    containing the rewards for the particular path

            returns:
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values 
                    whose length is the sum of the lengths of the paths

            ----------------------------------------------------------------------------------
            
            Your code should construct numpy arrays for Q-values which will be used to compute
            advantages (which will in turn be fed to the placeholder you defined in 
            Agent.define_placeholders). 
            
            Recall that the expression for the policy gradient PG is
            
                  PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]
            
            where 
            
                  tau=(s_0, a_0, ...) is a trajectory,
                  Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
                  and b_t is a baseline which may depend on s_t. 
            
            You will write code for two cases, controlled by the flag 'reward_to_go':
            
              Case 1: trajectory-based PG 
            
                  (reward_to_go = False)
            
                  Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over 
                  entire trajectory (regardless of which time step the Q-value should be for). 
            
                  For this case, the policy gradient estimator is
            
                      E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
            
                  where
            
                      Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.
            
                  Thus, you should compute
            
                      Q_t = Ret(tau)
            
              Case 2: reward-to-go PG 
            
                  (reward_to_go = True)
            
                  Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
                  from time step t. Thus, you should compute
            
                      Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            
            
            Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
            like the 'ob_no' and 'ac_na' above. 
        """
        # YOUR_CODE_HERE

        q_n = np.array([])
        if self.reward_to_go:
            # raise NotImplementedError
            for re_single_path in re_n:
                q_t_vector = [np.sum([(self.gamma**i)*(re_single_path[j+i]) for i in range(len(re_single_path)-j)]) for j in range(len(re_single_path))] # Seems wrong, but i guess we sum this entire thing...
                # q_t_vector = [q_t for j in range(len(re_single_path))]
                q_n = np.append(q_n,q_t_vector)
        else:
            # Loops are probably the slower way to do this. maybe even nested comprehensions are slower...
            for re_single_path in re_n:
                q_t = np.sum([(self.gamma**j)*re_single_path[j] for j in range(len(re_single_path))])  # Seems wrong, but i guess we sum this entire thing...
                q_t_vector = [q_t for j in range(len(re_single_path))]
                q_n = np.append(q_n,q_t_vector)
        return q_n

    def compute_advantage(self, ob_no, q_n):
        """
            Computes advantages by (possibly) subtracting a baseline from the estimated Q values

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from 
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values 
                    whose length is the sum of the lengths of the paths

            returns:
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated 
                    advantages whose length is the sum of the lengths of the paths
        """
        #====================================================================================#
        #                           ----------PROBLEM 6----------
        # Computing Baselines
        #====================================================================================#
        if self.nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current batch of Q-values. (Goes with Hint
            # #bl2 in Agent.update_parameters.
            # to match the statistics, lets just scale them both to have std = 1 and mean = 0. This means we'll always sorta be normalizing advantages
            # I think they maybe meant for me to just multiply the predictions by the std of the q_n and add the mean...
            b_n = self.sess.run(self.baseline_prediction, feed_dict = {self.sy_ob_no:ob_no}) 
            scale_mean = np.mean(q_n)
            scale_std = np.std(q_n)
            b_n = np.add(np.multiply(b_n,scale_std),scale_mean)


            # pdb_checker = [np.mean(q_n),np.std(q_n),np.mean(b_n),np.std(b_n)] # Note that this works quite well. Not perfectly. The mean is pretty dead on. 
            # print(pdb_checker)
            # q_n = np.divide(np.subtract(q_n,np.mean(q_n)),np.std(q_n))
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()
        return adv_n

    def estimate_return(self, ob_no, re_n):
        """
            Estimates the returns over a set of trajectories.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from 
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                re_n: length: num_paths. Each element in re_n is a numpy array 
                    containing the rewards for the particular path

            returns:
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values 
                    whose length is the sum of the lengths of the paths
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated 
                    advantages whose length is the sum of the lengths of the paths
        """
        q_n = self.sum_of_rewards(re_n)
        adv_n = self.compute_advantage(ob_no, q_n)
        #====================================================================================#
        #                           ----------PROBLEM 3----------
        # Advantage Normalization 

        # Question::: why do i think this is allowed and doesn't bias the policy gradient?
        # Also, i should double check to make sure this actually works.
        #====================================================================================#
        if self.normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            # raise NotImplementedError
            mean = np.mean(adv_n)
            std = np.std(adv_n)
            adv_n = np.divide(np.subtract(adv_n,mean),std) # YOUR_CODE_HERE
        return q_n, adv_n

    def update_parameters(self, ob_no, ac_na, q_n, adv_n):
        """ 
            Update the parameters of the policy and (possibly) the neural network baseline, 
            which is trained to approximate the value function.

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: shape: (sum_of_path_lengths). These are the actions sampled I think.
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values 
                    whose length is the sum of the lengths of the paths
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated 
                    advantages whose length is the sum of the lengths of the paths

            returns:
                nothing

        """
        #====================================================================================#
        #                           ----------PROBLEM 6----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if self.nn_baseline:
            # If a neural network baseline is used, set up the targets and the inputs for the 
            # baseline. 
            # 
            # Fit it to the current batch in order to use for the next iteration. Use the 
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the 
            # targets to have mean zero and std=1. (Goes with Hint #bl1 in 
            # Agent.compute_advantage.)

            # So this is where it actually does the fitting, and we'll fit it to the normalized q_n values.


            target_n = np.divide(np.subtract(q_n,np.mean(q_n)),np.std(q_n))
            self.batch_baseline_loss = self.sess.run(self.baseline_loss,feed_dict={self.sy_target_n : target_n, self.sy_ob_no : ob_no})
            print('the baseline loss is '+ str(self.batch_baseline_loss))
            self.sess.run(self.baseline_update_op,feed_dict={self.sy_target_n : target_n, self.sy_ob_no : ob_no})


        #====================================================================================#
        #                           ----------PROBLEM 3----------
        # Performing the Policy Update
        #====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on 
        # the current batch of rollouts.
        # 
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below. 

        # YOUR_CODE_HERE
        # pdb.set_trace()
        self.batch_loss = self.sess.run(self.loss, feed_dict = {self.sy_ob_no : ob_no, self.sy_ac_na : ac_na, self.sy_adv_n : adv_n})
        print('the loss is '+str(self.batch_loss))
        self.batch_unscaled_loss = self.sess.run(self.unscaled_loss, feed_dict = {self.sy_ob_no : ob_no, self.sy_ac_na : ac_na, self.sy_adv_n : adv_n})
        print('the unscaled loss is '+str(self.batch_unscaled_loss))
        _ = self.sess.run(self.update_op,feed_dict = {self.sy_ob_no : ob_no, self.sy_ac_na : ac_na, self.sy_adv_n : adv_n})

        # raise NotImplementedError
    def save_models_action(self,save_string):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, 'my_save_loc/'+save_string+'.ckpt')
        print('save path is '+save_path)
        return save_path

    def load_models_action(self,save_path):
        saver = tf.train.Saver()
        saver.restore(self.sess,save_path)
        print('model loaded from '+save_path)

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
    import tensorflow as tf
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
    if parallel is True:
        temp_save_string = logdir[5:-2]+'_temp'
        print(temp_save_string)
        temp_save_string = agent.save_models_action(temp_save_string) #yet another janky way to do this

    #========================================================================================#
    # Training Loop
    #========================================================================================#
    best_avg_return = -(5e10)
    total_timesteps = 0
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)
        if parallel is True:
            paths, timesteps_this_batch = agent.parallel_sampler(env,temp_save_string)
            print('so it returned it?')
        else:
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
        # One potential issue here is that there won't be a local for the first iteration. we must make it
        # so.
    if parallel is True:
        temp_save_string = logdir[5:-2]+'_temp'
        agent.save_models_action(temp_save_string)

    if save_models == True and save_best_model==False:
        save_string = logdir[5:-2]
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

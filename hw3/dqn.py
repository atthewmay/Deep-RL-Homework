import uuid
import time
import pickle
import sys
import gym.spaces
import itertools
import numpy as np
import random
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *

"""Summary Of the things I messed up on.
One. I neglected to include stop gradient on the target function. it actually works fine w/ this missing...

2!!! I did not include axis=1 in tf.reduce_sum(), which caused tensor flow to simply broadcast a single value (the best action in the entire
batch) rather than determining the best action for each sample in the batch. This makes it utter nonsense.

3. I neglected to include the (1-done_mask_ph), which shouldn't have too much effect.""" 

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

class QLearner(object):

  def __init__(
    self,
    env,
    q_func,
    optimizer_spec,
    session,
    exploration=LinearSchedule(1000000, 0.1),
    stopping_criterion=None,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000,
    grad_norm_clipping=10,
    rew_file=None,
    double_q=True,
    lander=False):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.

        Cool! So we are putting in the past frames as input channels to our model. This is probably much
        simpler than trying to use recurrent CNNs.

    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    double_q: bool
        If True, then use double Q-learning to compute target values. Otherwise, use vanilla DQN.
        https://papers.nips.cc/paper/3964-double-q-learning.pdf
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    self.target_update_freq = target_update_freq
    self.optimizer_spec = optimizer_spec
    self.batch_size = batch_size
    self.learning_freq = learning_freq
    self.learning_starts = learning_starts
    self.stopping_criterion = stopping_criterion
    self.env = env
    self.session = session
    self.exploration = exploration
    self.rew_file = str(uuid.uuid4()) + '.pkl' if rew_file is None else rew_file # This is a file w/ reward
    # info

    epsilon_greedy = True
    ###############
    # BUILD MODEL #
    ###############

    if len(self.env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = self.env.observation_space.shape
    else:
        img_h, img_w, img_c = self.env.observation_space.shape #will it ever be in color?
        input_shape = (img_h, img_w, frame_history_len * img_c)
    self.num_actions = self.env.action_space.n

    # set up placeholders
    # placeholder for current observation (or state)
    self.obs_t_ph              = tf.placeholder(
        tf.float32 if lander else tf.uint8, [None] + list(input_shape)) #why [None]+?
    """oh duh! the [None] is the dimension for batch_size and the list(input_shape) is the dimension for the
    input variables."""
    # placeholder for current action
    self.act_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    self.rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    # Note I really think this is the wrong shape, so I'm redefining it.
    self.obs_tp1_ph            = tf.placeholder( tf.float32 if lander else tf.uint8, [None] + list(input_shape))
#    self.obs_tp1_ph = tf.placeholder( tf.float32 if lander else tf.uint8, list(input_shape) + [None])

    '''# Just as a self note of comparison to PG, in that we had placeholders for obs and actions and
    then we got a sampled action by taking it out of the computation graph. We then did training by taking
    the list of actions and observations that we track as we interact w/ env, and we put in both of these
    placeholders to determine the prob of taking each action we did take. then we
    max(logprob(a)*reward(s,a))

    In contrast this is actually a little simpler. We just track states and actions along with reward and next state'''

    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    self.done_mask_ph          = tf.placeholder(tf.float32, [None])

    # casting to float on GPU ensures lower data transfer times.
    if lander:
      obs_t_float = self.obs_t_ph
      obs_tp1_float = self.obs_tp1_ph
    else:
      obs_t_float   = tf.cast(self.obs_t_ph,   tf.float32) / 255.0
      obs_tp1_float = tf.cast(self.obs_tp1_ph, tf.float32) / 255.0

    # Here, you should fill in your own code to compute the Bellman error. This requires
    # evaluating the current and next Q-values and constructing the corresponding error.
    # TensorFlow will differentiate this error for you, you just need to pass it to the
    # optimizer. See assignment text for details.
    # Your code should produce one scalar-valued tensor: total_error
    # This will be passed to the optimizer in the provided code below.
    # Your code should also produce two collections of variables:
    # q_func_vars
    # target_q_func_vars
    # These should hold all of the variables of the Q-function network and target network,
    # respectively. A convenient way to get these is to make use of TF's "scope" feature.
    # For example, you can create your Q-function network with the scope "q_func" like this:
    # <something> = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
    # And then you can obtain the variables like this:
    # q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    # Older versions of TensorFlow may require using "VARIABLES" instead of "GLOBAL_VARIABLES"
    # Tip: use huber_loss (from dqn_utils) instead of squared error when defining self.total_error
    ######

    """#OH SO WEIRD! I just learned that the q_fn will output self.action_space dimension, always, which means
    there is no action input, and we just have to select the proper output for comparison. I think all action
    spaces are discrete, so this works. Wow."""
    self.q_fn = q_func(obs_t_float,self.num_actions,scope="q_fn",reuse=False)
    #Note that self.q_fn = out, so calling q_fn runs all the other nodes that were defined.
    self.target_fn = q_func(obs_tp1_float,self.num_actions,scope="target_fn",reuse=False)

    # Now bellman error = huber_loss(Q(s,a)-y_i), where y_i is the r(s,a)+gamma*max_a'[Q(s',a')]
#    y_vect = tf.add(self.rew_t_ph,(1-self.done_mask_ph)*tf.multiply(gamma,tf.reduce_max(self.target_fn,axis = 1)))
    # y_vect = tf.add(self.rew_t_ph,tf.multiply(gamma,tf.reduce_max(self.target_fn,axis = 1)))
    """Wow, the error here was that you really need axis = 1 in your tf.reduce_max, or else it reduces it
    along the axis dimension as well! Yikes!

    The 1-done_mask basically just sets that term to zero if the episode was ending."""
    y_vect = tf.add(self.rew_t_ph,(1-self.done_mask_ph)*tf.multiply(gamma,tf.reduce_max(self.target_fn,axis = 1)))
#     y_vect = tf.Print(y_vect, [tf.reduce_max(self.target_fn, axis = 1),tf.shape(tf.reduce_max(self.target_fn, axis=1))],
#                       "axis=1 ")
#     y_vect = tf.Print(y_vect, [tf.reduce_max(self.target_fn),tf.shape(tf.reduce_max(self.target_fn))],
#                       "no_axis ")
#     # gotta take maximum valued action
    # import pdb; pdb.set_trace()
    indexer = tf.stack([tf.range(0,tf.shape(self.act_t_ph)[0],1), self.act_t_ph], axis = 1) # Makes the [[0,a0],[1,a1],...] array
    self.current_q_fn = tf.gather_nd(self.q_fn,indexer)
#    self.total_error = tf.reduce_sum(huber_loss(tf.subtract(self.current_q_fn,y_vect)))
    self.total_error = tf.reduce_mean(huber_loss(tf.subtract(self.current_q_fn,tf.stop_gradient(y_vect))))

    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_fn')
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_fn')

    ######

    # construct optimization op (with gradient clipping)
    self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)
    self.train_fn = minimize_and_clip(optimizer, self.total_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)
    """the self.train_fn = optimizer.apply_gradients(gradients)"""
    """okay cool so for this you don't need to use session bc the optimizer can automatically do its
    update"""
    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    self.update_target_fn = tf.group(*update_target_fn) # IDK what this * is doing...

    # construct the replay buffer
    self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, lander=lander)
    """Look into this later *** """
    self.replay_buffer_idx = None

    ###############
    # RUN ENV     #
    ###############
    self.model_initialized = False
    self.num_param_updates = 0
    self.mean_episode_reward      = -float('nan')
    self.best_mean_episode_reward = -float('inf')
    self.last_obs = self.env.reset()
    self.log_every_n_steps = 10000

    self.start_time = None
    self.t = 0

  def stopping_criterion_met(self):
    """I think the self.stopping_criterion(self.env,self.t) returns True/False, depending on whether the
      number of steps taken in an environment is >= num_timesteps, which is def as 2e8 in run_dqn_atari.py"""
    return self.stopping_criterion is not None and self.stopping_criterion(self.env, self.t)

  def step_env(self):
    ### 2. Step the env and store the transition
    # At this point, "self.last_obs" contains the latest observation that was
    # recorded from the simulator. Here, your code needs to store this
    # observation and its outcome (reward, next observation, etc.) into
    # the replay buffer while stepping the simulator forward one step.
    # At the end of this block of code, the simulator should have been
    # advanced one step, and the replay buffer should contain one more
    # transition.
    # Specifically, self.last_obs must point to the new latest observation.
    # Useful functions you'll need to call:
    # obs, reward, done, info = env.step(action)
    # this steps the environment forward one step
    # obs = env.reset()
    # this resets the environment if you reached an episode boundary.
    # Don't forget to call env.reset() to get a new observation if done
    # is true!!
    # Note that you cannot use "self.last_obs" directly as input
    # into your network, since it needs to be processed to include context
    # from previous frames. You should check out the replay buffer
    # implementation in dqn_utils.py to see what functionality the replay
    # buffer exposes. The replay buffer has a function called
    # encode_recent_observation that will take the latest observation
    # that you pushed into the buffer and compute the corresponding
    # input that should be given to a Q network by appending some
    # previous frames.
    # Don't forget to include epsilon greedy exploration!
    # And remember that the first time you enter this loop, the model
    # may not yet have been initialized (but of course, the first step
    # might as well be random, since you haven't trained your net...)

    #####
    self.t += 1 # The big counter on the entire training process. 
    # YOUR CODE HERE
   #since we are always on the most recent timestep of the buffer, we just call this...
   #right, first gotta store the last frame, and then we go with it. 
    frame_idx = self.replay_buffer.store_frame(self.last_obs)
    # implement epsilon greedy
#    import pdb; pdb.set_trace()
    if random.random() < self.exploration.value(self.t) or self.model_initialized == False:
        """oh gosh, it's upposed to be discrete"""
        # action = list(np.random.uniform([0,1,self.num_actions]))
        action = random.randint(0,self.num_actions-1)
    else:

#         if len(self.replay_buffer.encode_recent_observation().shape) == 1:
#             encoded_obs = self.replay_buffer.encode_recent_observation().reshape((-1,len(self.replay_buffer.encode_recent_observation())))
#         else:
#             encoded_obs = self.replay_buffer.encode_recent_observation()
        # action = self.session.run(self.q_fn,feed_dict = {self.obs_t_ph:encoded_obs})
        action = self.session.run(self.q_fn,feed_dict =
                                  {self.obs_t_ph:[self.replay_buffer.encode_recent_observation()]})
        """realize, each dimension output of action represents the predicted Q-value of that action_index at
        the given input state. The 'policy' of Q-learning is to choose the action that maximizes the Q-val,
        so we thus do max(action) thing"""
        action = np.argmax(action)
    obs, reward, done, info = self.env.step(action)
    self.replay_buffer.store_effect(frame_idx,action,reward,done)
        # Note: I worry the idx I pass in will cause error if it's off by one ***
    if done == 1:
        obs = self.env.reset()
    self.last_obs = obs

  def update_model(self):
    ### 3. Perform experience replay and train the network.
    # note that this is only done if the replay buffer contains enough samples
    # for us to learn something useful -- until then, the model will not be
    # initialized and random actions should be taken
    if (self.t > self.learning_starts and \
        self.t % self.learning_freq == 0 and \
        self.replay_buffer.can_sample(self.batch_size)):
        '''So learning_freq = 4'''

        obs_t_batch, act_batch, rew_batch, obs_tp1_batch, done_mask = self.replay_buffer.sample(self.batch_size)
      
        if self.model_initialized == False:
            initialize_interdependent_variables(self.session,tf.global_variables(), {
                self.obs_t_ph : obs_t_batch,
                self.obs_tp1_ph : obs_tp1_batch})
            self.model_initialized = True

#         total_error = self.session.run(self.total_error,feed_dict = {
#             self.obs_t_ph : obs_t_batch,
#             self.act_t_ph : act_batch,
#             self.rew_t_ph : rew_batch,
#             self.obs_tp1_ph : obs_tp1_batch,
#             self.done_mask_ph : done_mask})

        """Okay this is actually really interesting. Even tho self.train_fn() wouldn't need to be in a
        sess.run(), as the optimizer can just call 'minimize', it still needs to be in one, as we use different
        learning rates, which are placeholders. This means we pass in a different optimizer I think at each
        learning rate???"""

        """OH, so i think I can't do teh following, bc self.total_error isn't a placeholder and it won't
        accept some kind of scaler value perhaps. OH wow, and it doesn't make sense to calculate total_error
        separately from the train_fn step, bc if I just pass that scaler value in, I've broken the
        computation graph and TF can't do gradient descent i think. The graph's gotta flow in order for TF to
        know where everything came from."""
        current_lr = self.optimizer_spec.lr_schedule.value(self.t)
        # self.session.run(self.train_fn, feed_dict = {self.total_error : total_error, self.learning_rate : current_lr})
        
        self.session.run(self.train_fn, feed_dict = {
            self.learning_rate : current_lr,
            self.obs_t_ph : obs_t_batch,
            self.act_t_ph : act_batch,
            self.rew_t_ph : rew_batch,
            self.obs_tp1_ph : obs_tp1_batch,
            self.done_mask_ph : done_mask})


        if self.num_param_updates % self.target_update_freq == 0: #I guess this would update the very first time thru. 

            self.session.run(self.update_target_fn)
 
        self.num_param_updates += 1

#    self.t += 1 # IDK why you want a timestep here...

         # Here, you should perform training. Training consists of four steps:
          # 3.a: use the replay buffer to sample a batch of transitions (see the
          # replay buffer code for function definition, each batch that you sample
          # should consist of current observations, current actions, rewards,
          # next observations, and done indicator).
          # 3.b: initialize the model if it has not been initialized yet; to do
          # that, call
          #    initialize_interdependent_variables(self.session, tf.global_variables(), {
          #        self.obs_t_ph: obs_t_batch,
          #        self.obs_tp1_ph: obs_tp1_batch,
          #    })
          # where obs_t_batch and obs_tp1_batch are the batches of observations at
          # the current and next time step. The boolean variable model_initialized
          # indicates whether or not the model has been initialized.
          # Remember that you have to update the target network too (see 3.d)!
          # 3.c: train the model. To do this, you'll need to use the self.train_fn and
          # self.total_error ops that were created earlier: self.total_error is what you
          # created to compute the total Bellman error in a batch, and self.train_fn
          # will actually perform a gradient step and update the network parameters
          # to reduce total_error. When calling self.session.run on these you'll need to
          # populate the following placeholders:
          # self.obs_t_ph
          # self.act_t_ph
          # self.rew_t_ph
          # self.obs_tp1_ph
          # self.done_mask_ph
          # (this is needed for computing self.total_error)
          # self.learning_rate -- you can get this from self.optimizer_spec.lr_schedule.value(t)
          # (this is needed by the optimizer to choose the learning rate)
          # 3.d: periodically update the target network by calling
          # self.session.run(self.update_target_fn)
          # you should update every target_update_freq steps, and you may find the
          # variable self.num_param_updates useful for this (it was initialized to 0)
          #####

          # YOUR CODE HERE
  def log_progress(self):
    episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()

    if len(episode_rewards) > 0:
      self.mean_episode_reward = np.mean(episode_rewards[-100:])

    if len(episode_rewards) > 100:
      self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

    if self.t % self.log_every_n_steps == 0 and self.model_initialized:
      print("Timestep %d" % (self.t,))
      print("mean reward (100 episodes) %f" % self.mean_episode_reward)
      print("best mean reward %f" % self.best_mean_episode_reward)
      print("episodes %d" % len(episode_rewards))
      print("exploration %f" % self.exploration.value(self.t))
      print("learning_rate %f" % self.optimizer_spec.lr_schedule.value(self.t))
      if self.start_time is not None:
        print("running time %f" % ((time.time() - self.start_time) / 60.))

      self.start_time = time.time()

      sys.stdout.flush()

      with open(self.rew_file, 'wb') as f:
        pickle.dump(episode_rewards, f, pickle.HIGHEST_PROTOCOL)
    
    def initialize_tf(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__() # equivalent to `with self.sess:`
        tf.global_variables_initializer().run() #pylint: disable=E1101



def learn(*args, **kwargs):
  alg = QLearner(*args, **kwargs)
  while not alg.stopping_criterion_met():
    alg.step_env()
    # at this point, the environment should have been advanced one step (and
    # reset if done was true), and self.last_obs should point to the new latest
    # observation
    alg.update_model()
    alg.log_progress()
#    if alg.t > 520:
#        import pdb; pdb.set_trace()


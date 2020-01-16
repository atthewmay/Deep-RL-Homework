import psutil
import sys
import os


parent_dir = (os.path.abspath('../../../gravity_ball_game'))
os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")
sys.path.append("../../../gravity_ball_game/")

from gravity_ball_game_training_simulator import GB_game
import numpy as np
from random import randint
import time
import ray
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from train_pg_f18 import Agent, pathlength
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from ray.experimental.tf_utils import TensorFlowVariables
tf.logging.set_verbosity(tf.logging.WARN)

#just for the sake of testing
computation_graph_args = {
        'n_layers': 2,
        'output_activation': 'None',
        'ob_dim': 2,
        'ac_dim': 2,
        'discrete': True,
        'size': 32,
        'learning_rate': 1e-3,
        'baseline_lr' : 1e-3,
        }

sample_trajectory_args = {
        'animate': False,
        'max_path_length': 1000,
        'min_timesteps_per_batch': 2000,
    }

estimate_return_args = {
        'gamma': 0.99,
        'reward_to_go': False,
        'nn_baseline': False,
        'normalize_advantages': False,
    }

num_cpus = psutil.cpu_count(logical=False)
num_cpus=2
print('the number of cpus is ' + str(num_cpus))
ray.init(num_cpus=num_cpus)

filename = '/tmp/model'

@ray.remote
class Parallel_Actor(Agent):
    def __init__(self,computation_graph_args,sample_trajectory_args,estimate_return_args):
        super().__init__(computation_graph_args, sample_trajectory_args, estimate_return_args)
        # build computation graph
        self.build_computation_graph()

        # tensorflow: config, session, variable initialization
        self.init_tf_sess()

        self.variables = TensorFlowVariables(self.loss, self.sess)

    def set_weights(self, weights):
        self.variables.set_weights(weights)

    def get_weights(self):
        weights = self.variables.get_weights()

        return weights

    def test_method(self,i):
        return i*2

    def sample_trajectories(self, itr, env, counter_actor):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            if hasattr(self,'running_only') and self.animate:
                animate_this_episode=True

            path = self.sample_trajectory(env, animate_this_episode)
            path_len = pathlength(path)
            paths.append(path)
            timesteps_this_batch += path_len

            counter_actor.increment_counter.remote(path_len)
            current_count = ray.get(counter_actor.return_count.remote())

            if current_count > self.min_timesteps_per_batch:
                break

        return paths, timesteps_this_batch

    def sample_trajectories_fake(self, itr,env, counter_actor):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        print(env.size)
        while True:
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            if hasattr(self,'running_only') and self.animate:
                animate_this_episode=True

#            path = self.sample_trajectory(env, animate_this_episode)
#            paths.append(path)
#            timesteps_this_batch += pathlength(path)
            path_len = self.fake_sample_trajectory()
            counter_actor.increment_counter.remote(path_len)
            current_count = ray.get(counter_actor.return_count.remote())
            timesteps_this_batch += path_len
            time.sleep(0.3)
#             if timesteps_this_batch > self.min_timesteps_per_batch:
            if current_count > self.min_timesteps_per_batch:
                print('the final pathlength from this worker is ' + str(timesteps_this_batch))
                paths = 'filler'
                break
        return paths, timesteps_this_batch #


    def fake_sample_trajectory(self):
#         return randint(100)
        return 100


@ray.remote
class Counter():
    def __init__(self):
        self.count=0
    def increment_counter(self,pathlen):
        self.count+=pathlen
    def return_count(self):
        return self.count
    def reset_counter(self):
        self.count = 0

def main():

    env = GB_game(num_char = 4, reward_circle = True, death_penalty = False, relative_positions = True, discrete=True, max_speed=10)
    print('the size of env is'+str(env.size))
#     env = dummy_obj()
    ray.register_custom_serializer(GB_game, use_pickle=True) # amazing. I needed to use this to get it to
#    work!

    env = ray.put(env)
    print('\nthe put succeeded!!\n')
    actors = [Parallel_Actor.remote(computation_graph_args,sample_trajectory_args,estimate_return_args) for i in range(num_cpus)]
    CA = Counter.remote()
#     weights_copy = actors[0].get_weights.remote()
#     ray.get([actor.set_weights.remote(weights_copy) for actor in actors])
#     weights = ray.get([actor.get_weights.remote() for actor in actors])
# 
#     for i in range(len(weights)):
#         np.testing.assert_equal(weights[i], weights[0])
#     print('test passed!')
    return_array = ray.get([actor.sample_trajectories_fake.remote(10,env,CA) for actor in actors])
    print(return_array)
#for actor in actors:
#    weights = actor.get_weights.remote()
# Parallelize the evaluation of some test data.
#results = ray.get([actor.evaluate_next_batch.remote() for actor in actors])

if __name__ == "__main__":
    main()

# so the way this is working is that each worker runs once and the first one dominates. Teh sleep fixes.

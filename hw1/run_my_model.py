#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from hw_helpers import model, layers_model
import ipdb as pdb
import pprint

def run_model():
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('ls', type=list)
    parser.add_argument('my_saved_parameters_path', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()


    tf.reset_default_graph()


    with tf.Session() as sess:
        print('loading and building my model')
        ls = [11,64,32,2]

        # mymodel = layers_model(ls,sess = sess)
        # saver = tf.train.Saver()
        # saver.restore(sess = sess, save_path = args.my_saved_parameters_path)
        # print('loaded and built')
        # sess.run(tf.global_variables_initializer())

        new_saver = tf.train.import_meta_graph(args.my_saved_parameters_path+'.meta')
        new_saver.restore(sess, args.my_saved_parameters_path)
        graph = tf.get_default_graph()
        input_placeholder = graph.get_tensor_by_name('msh_input_placeholder:0')
        output = graph.get_tensor_by_name('fully_connected_2/BiasAdd:0')

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            # pdb.set_trace()
            while not done:
                # action = mymodel.make_preds(obs[None,:])
                action = sess.run(output, feed_dict= {input_placeholder:obs[None,:]})
 
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                # print(obs)
                totalr += r
                steps += 1
                # if done == True:
                #     pdb.set_trace()
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        my_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        with open(os.path.join('my_data', args.envname + '.pkl'), 'wb') as f:
           pickle.dump(my_data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    run_model()

import tensorflow as tf
import numpy as np
import pickle
# import tf_util
import gym
from sklearn.model_selection import train_test_split
import ipdb as pdb
import load_policy

def prepare_data(file_location):
    """
    load data from file and split into training and testing.

    filename: filename fo the pickle file that stores the observations and
        actions from rollouts after running the expert policy. The pickle
        file store a dictionary, with keys 'observation' and 'actions'
    
    return:
        X_train, X_test, y_train, y_tests: as list. X is observation space data,
            y is action space data.

    """
    with open(file_location, 'rb') as f:
        data = pickle.loads(f.read())
    data['actions'] = np.squeeze(data['actions'], axis=1)
    return _shuffle_and_split_data(data)


def _shuffle_and_split_data(data):
    """
    data is a json object that stores observations and actions

    """
    X = np.array(data['observations'])
    y = data['actions']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

class layers_model():
    def tf_reset(self):
        try:
            self.sess.close()
        except:
            pass
        tf.reset_default_graph()
        return tf.Session()
    def __init__(self,data_in,ls = None,activations = [tf.nn.tanh, tf.nn.tanh, None], sess = None, reg_scale = 0.1, lr = 1e-3):
        """data_in is either [X_train,y_train] or is a directory location of where to find the expert data.
        ls = layer_sizes is [input_size, h1_size, h2_size,...,hn_size, output_size]

        Note that this way of doing things means I can't ever use the model.predict thing. If I initialize it with data, then it'll only be useful for
        training bc i won't remake the graph manually, but just load the meta file."""
        if sess == None:
            self.sess = self.tf_reset()
        else:
            self.sess = sess

        if isinstance(data_in, str): # convert string of direcotry into test and train data.
            X_train, X_test, y_train, y_test=prepare_data(data_in)
            self.train_data = [X_train,y_train] # could do a self.test_data and self.train_data, but i don't want to work with test data i think. 
            self.test_data = [X_test,y_test]
        else: 
            self.train_data = data_in
        if ls == None:  
            self.ls = [self.train_data[0].shape[-1],64,32,self.train_data[-1].shape[-1]]
        else:
            self.ls = [self.train_data[0].shape[-1],ls[1],ls[2],self.train_data[-1].shape[-1]]


        self.activations = activations

        self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.ls[0]], name = 'msh_input_placeholder') # batch-size by state size
        self.output_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.ls[-1]]) # action space size

        self.fc1 = tf.contrib.layers.fully_connected(self.input_ph, self.ls[1],
            weights_regularizer=tf.contrib.layers.l2_regularizer(reg_scale),
            activation_fn=activations[0])

        self.fc2 = tf.contrib.layers.fully_connected(self.fc1, self.ls[2],
            weights_regularizer=tf.contrib.layers.l2_regularizer(reg_scale),
            activation_fn=activations[1])

        self.output_pred = tf.contrib.layers.fully_connected(self.fc2, self.ls[-1],
            weights_regularizer=tf.contrib.layers.l2_regularizer(reg_scale),
            activation_fn=activations[-1])

        self.lr = lr
        self.mse = tf.losses.mean_squared_error(self.output_ph, self.output_pred, loss_collection=tf.compat.v1.GraphKeys.LOSSES)

        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate = self.lr).minimize(self.mse)
        self.sess.run(tf.compat.v1.global_variables_initializer()) # I couldn't really move this around, bc it cam after the optimizer. 
        self.saver = tf.compat.v1.train.Saver()

    def train_model(self, iterations,  save_string = None, resume = False, lr = None, batch_size = 100, save_single_performance_data = None, envname = None):
        inputs = self.train_data[0]
        outputs = self.train_data[1]

        if save_single_performance_data is not None:
            self.save_performance_data = {'reward_means':[], 'reward_stds':[],'batches-batch-size':[iterations,batch_size]}

        if lr is not None:
            self.lr = lr

        if resume == True:
            try:
                self.saver.restore(self.sess,'my_save_loc/'+save_string+'.ckpt')
            except:
                print('couldnt load')

        for training_step in range(1,iterations+1):
            indices = np.random.randint(low=0, high=len(inputs), size=batch_size)
            input_batch = inputs[indices]
            output_batch = outputs[indices]
            _, mse_run = self.sess.run([self.opt, self.mse], feed_dict={self.input_ph: input_batch, self.output_ph: output_batch})
            if training_step % 1000 == 0:
                print('{0:04d} mse: {1:.5f}'.format(training_step, mse_run))
                if save_single_performance_data is not None:
                    self.run_model(50,render_runs = False,envname = envname)
                if save_string is not None:
                    save_path = self.saver.save(self.sess, 'my_save_loc/'+save_string+'.ckpt')
                    print('save path is '+save_path)

        if save_single_performance_data is not None:
            save_performance_filename = 'my_data/dagger_performance_data/' + save_string
            with open(save_performance_filename, 'wb') as f:
                data = pickle.dump(self.save_performance_data, f)

    def dagger_train_model(self, dagger_iter, rollouts_per_dagger, envname, expert_policy_file, save_data_agg = None, render_runs = False, iterations = 1000, save_string = None, resume = False, lr = 1e-3, batch_size = 100, save_performance_data = None):
        '''
        initially we train the model st we have a working one
        we add data to self.training_data using the observations we get from running the model, and then putting those thru the expert policy
        Initializing the model will already make a session, so as long as we have initialized the model and not reset the session or something, it should work.

        OKAY. SO WILL NEED TO WORK ON THE SAVING/RESUMING TRAINING STRUCTURE, IN CASE you want to load up a previously trained model and do dagger.'''
        # self.dagger = True
        if save_performance_data is not None:
            self.save_performance_data = {'reward_means':[], 'reward_stds':[],'batches-batch-size':[iterations,batch_size]}

        self.train_model(iterations,  save_string=save_string, resume=resume, lr=lr, batch_size=batch_size) # gives us an initial model. Resume determines whether you pick up where you left off. but that won't fully work unless we save the new data as well.

        for i in range(dagger_iter):
            print('dagger iteration number '+str(i))
            output_obs = self.run_model(rollouts_per_dagger,render_runs,envname)
            output_actions = self.query_expert(output_obs, expert_policy_file) # this doesn't have to actually run environments on experts, but just query the policy
            
            self.merge_datasets(output_obs,output_actions)
            self.train_model(iterations,  save_string=save_string, lr=lr, batch_size=batch_size) # always gotta resume for this guy.
            if save_performance_data is not None:
                save_performance_filename = 'my_data/dagger_performance_data/' + save_string
                with open(save_performance_filename, 'wb') as f:
                    data = pickle.dump(self.save_performance_data, f)
        if save_data_agg is not None:
            pass # save it to this location.
        # if save_performance_data is not None:
        #     save_performance_filename = 'my_data/dagger_performance_data/' + save_string
        #     with open(save_performance_filename, 'wb') as f:
        #         data = pickle.dump(self.save_performance_data, f)


    def test_model(self, data_in):
        inputs = data_in[0]
        outputs = data_in[1]
        mse_run = self.sess.run(self.mse, feed_dict={self.input_ph: inputs, self.output_ph: outputs})
        print('mse: '+str(mse_run))        

    def make_preds(self,inputs):
        output_preds = self.sess.run(self.output_pred, feed_dict = {self.input_ph:inputs})
        return output_preds

    def run_model(self,num_rollouts,render_runs,envname,store_data = False):
        '''Remember there's already a session going after initializing the model.
            tho model_predict is probably more flexible
            Store_data is used if you want to plot the performance over dagger iterations'''

        # graph = tf.get_default_graph()
        # input_placeholder = graph.get_tensor_by_name('msh_input_placeholder:0')
        # output = graph.get_tensor_by_name('fully_connected_2/BiasAdd:0')

        import gym
        env = gym.make(envname)
        max_steps = env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            # print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            # pdb.set_trace()
            while not done:
                # action = mymodel.make_preds(obs[None,:])
                action = self.make_preds(obs[None,:])
 
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                # print(obs)
                totalr += r
                steps += 1
                # if done == True:
                #     pdb.set_trace()
                if render_runs:
                    env.render()
                # if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        # print('returns', returns)            
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        if hasattr(self,'save_performance_data'):
            self.save_performance_data['reward_means'].append(np.mean(returns))
            self.save_performance_data['reward_stds'].append(np.std(returns))

        my_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        return my_data['observations']

    def query_expert(self,observations, expert_policy_file):
        '''we really shouldn't need gym at all or the environment. All we need is the expert policy
        note that things like the envname and save_string and expert policy file should all be dictated by the envname'''
        # ALREADY_INITIALIZED = set()
        # def initialize():
        #     new_variables = set(tf.all_variables()) - ALREADY_INITIALIZED
        #     tf.get_default_session().run(tf.initialize_variables(new_variables))
        #     ALREADY_INITIALIZED.update(new_variables)
        # print('loading and building expert policy')
        policy_fn = load_policy.load_policy(expert_policy_file)
        # print('loaded and built')

        # tf_util.initialize() # I'm suspicious of this line...
        with tf.Session() as temp_sess:# Wow! tf.get_default_session() will only work w/in a "with" block.
            actions = []
            for obs in observations:
                # pdb.set_trace()
                action = policy_fn(obs[None,:])
                actions.append(action)

            output_actions = np.array(actions)

            return output_actions

    def merge_datasets(self, output_obs, output_actions):
        # pdb.set_trace()
        self.train_data[0] = np.append(self.train_data[0],output_obs,axis = 0)
        self.train_data[1] = np.append(self.train_data[1],np.squeeze(output_actions,axis = 1),axis = 0)

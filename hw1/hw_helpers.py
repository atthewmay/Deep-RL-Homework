import tensorflow as tf
import numpy as np
import pickle
# # class model():
# #     def tf_reset(self):
# #         try:
# #             sess.close()
# #         except:
# #             pass
# #         tf.reset_default_graph()
# #         return tf.Session()
# #     def __init__(self,ls,activations = [tf.nn.tanh, tf.nn.tanh, None], in_sess_already = True):
# #         """ls = layer_sizes is [input_size, h1_size, h2_size,...,hn_size, output_size]"""
# #         self.ls = ls
# #         if in_sess_already == False:
# #             sess = self.tf_reset()

# #         self.activations = activations
# #         self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, ls[0]]) # batch-size by state size
# #         self.output_ph = tf.placeholder(dtype=tf.float32, shape=[None, ls[-1]]) # action space size
# #         self.W_dict = {}
# #         self.b_dict = {}
# #         for i in range(len(ls)-1):
# #             self.W_dict[i] = tf.get_variable(name='W'+str(i), shape=[ls[i], ls[i+1]], initializer=tf.contrib.layers.xavier_initializer())
# #             self.b_dict[i] = tf.get_variable(name='b'+str(i), shape=[ls[i+1]], initializer=tf.constant_initializer(0.))


# #         self.layer = self.input_ph
# #         print(tf.shape(self.layer))


# #         for i in range(len(self.activations)):
# #             self.layer = tf.matmul(self.layer, self.W_dict[i]) + self.b_dict[i]
# #             print(tf.shape(self.layer))
# #             if self.activations[i] is not None:
# #                 self.layer = self.activations[i](self.layer)
# #             self.output_pred = self.layer


# #     def train_model(self, expert_data_loc, save_string, iterations, resume = True, troubleshooting = False):
# #         if troubleshooting == False:
# #             with open(expert_data_loc,'rb') as f:
# #                 expert_data = pickle.load(f)

# #             inputs = expert_data['observations']
# #             outputs = expert_data['actions'][:,0,:]
# #         else: #okay, bad practice, but this is just if you wanna directly insert data for testing!
# #             inputs = expert_data_loc[0]
# #             outputs = expert_data_loc[1]


# #         mse = tf.reduce_mean(0.5 * tf.square(self.output_pred - self.output_ph))
# #         # you got to use this because you were in putting batches of data. 
# #         opt = tf.train.AdamOptimizer().minimize(mse)
# #         sess.run(tf.global_variables_initializer())
# #         saver = tf.train.Saver()
# #         if resume == True:
# #             try:
# #                 saver.restore(sess,'../homework/hw1/my_save_loc/'+save_string+'.ckpt')
# #             except:
# #                 print('couldnt load')



# #         batch_size = 32
# #         for training_step in range(iterations):
# #             indices = np.random.randint(low=0, high=len(inputs), size=batch_size)
# #             input_batch = inputs[indices]
# #             output_batch = outputs[indices]
# #             _, mse_run = sess.run([opt, mse], feed_dict={self.input_ph: input_batch, self.output_ph: output_batch})
# #             if training_step % 1000 == 0:
# #                 print('{0:04d} mse: {1:.5f}'.format(training_step, mse_run))
# #                 save_path = saver.save(sess, '../homework/hw1/my_save_loc/'+save_string+'.ckpt')
# #                 print('save path is '+save_path)

# #     def make_preds(self,inputs):
# #         output_preds = sess.run(self.output_pred, feed_dict = {self.input_ph:inputs})
# #         return output_preds

import tensorflow as tf
import numpy as np
import pickle
class model():
    def tf_reset(self):
        try:
            self.sess.close()
        except:
            pass
        tf.reset_default_graph()
        return tf.Session()
    def __init__(self,ls,activations = [tf.nn.tanh, tf.nn.tanh, None], sess = None, RL = False, lr = 1e-2):
        """ls = layer_sizes is [input_size, h1_size, h2_size,...,hn_size, output_size]"""
        self.ls = ls
        if sess == None:
            self.sess = self.tf_reset()
        else:
            self.sess = sess
        self.activations = activations
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, ls[0]]) # batch-size by state size
        self.output_ph = tf.placeholder(dtype=tf.float32, shape=[None, ls[-1]]) # action space size
        self.W_dict = {}
        self.b_dict = {}
        for i in range(len(ls)-1):
            self.W_dict[i] = tf.get_variable(name='W'+str(i), shape=[ls[i], ls[i+1]], initializer=tf.contrib.layers.xavier_initializer())
            self.b_dict[i] = tf.get_variable(name='b'+str(i), shape=[ls[i+1]], initializer=tf.constant_initializer(0.))


        self.layer = self.input_ph
        print(tf.shape(self.layer))


        for i in range(len(self.activations)):
            self.layer = tf.matmul(self.layer, self.W_dict[i]) + self.b_dict[i]
            print(tf.shape(self.layer))
            if self.activations[i] is not None:
                self.layer = self.activations[i](self.layer)
            self.output_pred = self.layer

        if RL == True: 
            with tf.name_scope('reward_holder'):
                self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        
            with tf.name_scope('get_resp_outs'):
                self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32, name = 'action_holder')
            
                self.indexes = tf.range(0, tf.shape(self.output_pred)[0]) * tf.shape(self.output_pred)[1] + self.action_holder

                self.responsible_outputs = tf.gather(tf.reshape(self.output_pred, [-1]), self.indexes, name = 'responsible_outputs')
                # out of the output vector, this will pull out the indexes
                # But i still don't understand indexes.

            # i feel like instead of going thru all of this, you could have just saved the actual outputs. I think I'll try that.
            # then for responsible outputs, you'd do tf.gather(outputs, action_holder) oh maybe it's not different than this. 
            # Maybe that's exactly what they're doing, bc action_holder is a scaler number. IDK.
            with tf.name_scope('loss'):
                self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder) #becuase reward_holder value 
            # doesn't directly change as you change the Weights, this is equivalent to multiplying the gradient by the reward.
            # when you take the gradient, you're solving for d(log*A)/dW = d(log_p)/dW * d(log_p*A)/d(log_p) = A*d(log_p)/dW. so it's equivalent to mult gradient
            # by the reward function
            tvars = tf.trainable_variables()

            with tf.name_scope('update'):
                # self.train_step = tf.train.RMSPropOptimizer(learning_rate = lr, decay = 0.99).minimize(self.loss)
                self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
            self.init = tf.global_variables_initializer()


    def train_model(self, data_in, save_string, iterations, resume = True, troubleshooting = False, lr = 1e-2):

        inputs = data_in[0]
        outputs = data_in[1][:,0,:]
        # if troubleshooting == False:
        #     with open(expert_data_loc,'rb') as f:
        #         expert_data = pickle.load(f)

        #     inputs = expert_data['observations']
        #     outputs = expert_data['actions'][:,0,:]
        # else: #okay, bad practice, but this is just if you wanna directly insert data for testing!
        #     inputs = expert_data_loc[0]
        #     outputs = expert_data_loc[1]


        self.mse = tf.reduce_mean(0.5 * tf.square(self.output_pred - self.output_ph))
        # you got to use this because you were in putting batches of data. 
        opt = tf.train.AdamOptimizer(learning_rate = lr).minimize(self.mse)
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if resume == True:
            try:
                saver.restore(self.sess,'../homework/hw1/my_save_loc/'+save_string+'.ckpt')
            except:
                print('couldnt load')



        batch_size = 100
        for training_step in range(iterations):
            indices = np.random.randint(low=0, high=len(inputs), size=batch_size)
            input_batch = inputs[indices]
            output_batch = outputs[indices]
            _, mse_run = self.sess.run([opt, self.mse], feed_dict={self.input_ph: input_batch, self.output_ph: output_batch})
            if training_step % 1000 == 0:
                print('{0:04d} mse: {1:.5f}'.format(training_step, mse_run))
                save_path = saver.save(self.sess, '../homework/hw1/my_save_loc/'+save_string+'.ckpt')
                print('save path is '+save_path)

    def test_model(self, data_in):
        inputs = data_in[0]
        outputs = data_in[1][:,0,:]
        mse_run = self.sess.run(self.mse, feed_dict={self.input_ph: inputs, self.output_ph: outputs})
        print('mse: '+str(mse_run))

    def make_preds(self,inputs):
        output_preds = self.sess.run(self.output_pred, feed_dict = {self.input_ph:inputs})
        return output_preds





class layers_model():
    def tf_reset(self):
        try:
            self.sess.close()
        except:
            pass
        tf.reset_default_graph()
        return tf.Session()
    def __init__(self,ls,activations = [tf.nn.tanh, tf.nn.tanh, None], sess = None, RL = False, lr = 1e-2, reg_scale = 0.1):
        """ls = layer_sizes is [input_size, h1_size, h2_size,...,hn_size, output_size]"""
        self.ls = ls
        if sess == None:
            self.sess = self.tf_reset()
        else:
            self.sess = sess
        self.activations = activations
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, ls[0]], name = 'msh_input_placeholder') # batch-size by state size
        self.output_ph = tf.placeholder(dtype=tf.float32, shape=[None, ls[-1]]) # action space size

        self.fc1 = tf.contrib.layers.fully_connected(self.input_ph, ls[1],
            weights_regularizer=tf.contrib.layers.l2_regularizer(reg_scale),
            activation_fn=activations[0])

        self.fc2 = tf.contrib.layers.fully_connected(self.fc1, ls[2],
            weights_regularizer=tf.contrib.layers.l2_regularizer(reg_scale),
            activation_fn=activations[1])

        self.output_pred = tf.contrib.layers.fully_connected(self.fc2, ls[-1],
            weights_regularizer=tf.contrib.layers.l2_regularizer(reg_scale),
            activation_fn=activations[-1])


        # self.W_dict = {}
        # self.b_dict = {}
        # for i in range(len(ls)-1):
        #     self.W_dict[i] = tf.get_variable(name='W'+str(i), shape=[ls[i], ls[i+1]], initializer=tf.contrib.layers.xavier_initializer())
        #     self.b_dict[i] = tf.get_variable(name='b'+str(i), shape=[ls[i+1]], initializer=tf.constant_initializer(0.))


        # self.layer = self.input_ph
        # print(tf.shape(self.layer))


        # for i in range(len(self.activations)):
        #     self.layer = tf.matmul(self.layer, self.W_dict[i]) + self.b_dict[i]
        #     print(tf.shape(self.layer))
        #     if self.activations[i] is not None:
        #         self.layer = self.activations[i](self.layer)
        #     self.output_pred = self.layer

        if RL == True: 
            with tf.name_scope('reward_holder'):
                self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        
            with tf.name_scope('get_resp_outs'):
                self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32, name = 'action_holder')
            
                self.indexes = tf.range(0, tf.shape(self.output_pred)[0]) * tf.shape(self.output_pred)[1] + self.action_holder

                self.responsible_outputs = tf.gather(tf.reshape(self.output_pred, [-1]), self.indexes, name = 'responsible_outputs')
                # out of the output vector, this will pull out the indexes
                # But i still don't understand indexes.

            # i feel like instead of going thru all of this, you could have just saved the actual outputs. I think I'll try that.
            # then for responsible outputs, you'd do tf.gather(outputs, action_holder) oh maybe it's not different than this. 
            # Maybe that's exactly what they're doing, bc action_holder is a scaler number. IDK.
            with tf.name_scope('loss'):
                self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder) #becuase reward_holder value 
            # doesn't directly change as you change the Weights, this is equivalent to multiplying the gradient by the reward.
            # when you take the gradient, you're solving for d(log*A)/dW = d(log_p)/dW * d(log_p*A)/d(log_p) = A*d(log_p)/dW. so it's equivalent to mult gradient
            # by the reward function
            tvars = tf.trainable_variables()

            with tf.name_scope('update'):
                # self.train_step = tf.train.RMSPropOptimizer(learning_rate = lr, decay = 0.99).minimize(self.loss)
                self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
            self.init = tf.global_variables_initializer()


    def train_model(self, data_in, iterations,  save_string = None, resume = False, troubleshooting = False, lr = 1e-3, batch_size = 100):
        inputs = data_in[0]
        # outputs = data_in[1][:,0,:]
        outputs = data_in[1]

        self.mse = tf.losses.mean_squared_error(self.output_ph, self.output_pred, loss_collection=tf.GraphKeys.LOSSES)

        # self.mse = tf.reduce_mean(0.5 * tf.square(self.output_pred - self.output_ph))
        # you got to use this because you were in putting batches of data. 
        opt = tf.train.AdamOptimizer(learning_rate = lr).minimize(self.mse)
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if resume == True:
            try:
                saver.restore(self.sess,'../homework/hw1/my_save_loc/'+save_string+'.ckpt')
            except:
                print('couldnt load')



        
        for training_step in range(iterations):
            indices = np.random.randint(low=0, high=len(inputs), size=batch_size)
            input_batch = inputs[indices]
            output_batch = outputs[indices]
            _, mse_run = self.sess.run([opt, self.mse], feed_dict={self.input_ph: input_batch, self.output_ph: output_batch})
            if training_step % 1000 == 0:
                print('{0:04d} mse: {1:.5f}'.format(training_step, mse_run))
                if save_string is not None:
                    save_path = saver.save(self.sess, '../homework/hw1/my_save_loc/'+save_string+'.ckpt')
                    print('save path is '+save_path)
    def test_model(self, data_in):
        inputs = data_in[0]
        outputs = data_in[1][:,0,:]
        mse_run = self.sess.run(self.mse, feed_dict={self.input_ph: inputs, self.output_ph: outputs})
        print('mse: '+str(mse_run))        

    def make_preds(self,inputs):
        output_preds = self.sess.run(self.output_pred, feed_dict = {self.input_ph:inputs})
        return output_preds
import tensorflow as tf
import numpy as np

def create_model(ls):
    """ls = layer_sizes is [input_size, h1_size, h2_size, output_size]"""
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, ls[0]]) # batch-size by state size
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None, ls[-1]]) # action space size
    
    W0 = tf.get_variable(name='W0', shape=[ls[0], ls[1]], initializer=tf.contrib.layers.xavier_initializer())
    W1 = tf.get_variable(name='W1', shape=[ls[1], ls[2]], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable(name='W2', shape=[ls[2], ls[3]], initializer=tf.contrib.layers.xavier_initializer())
    
    b0 = tf.get_variable(name='b0', shape=[ls[1]], initializer=tf.constant_initializer(0.))
    b1 = tf.get_variable(name='b1', shape=[ls[2]], initializer=tf.constant_initializer(0.))
    b2 = tf.get_variable(name='b2', shape=[ls[3]], initializer=tf.constant_initializer(0.))
    
    weights = [W0,W1,W2]
    biases = [b0,b1,b2]
    activations = [tf.nn.relu, tf.nn.relu, None]
    
    layer = input_ph
    print(tf.shape(layer))


    for W, b, activation in zip(weights, biases, activations):
        layer = tf.matmul(layer, W) + b
        print(tf.shape(layer))
        if activation is not None:
            layer = activation(layer)
        output_pred = layer
        
    return input_ph, output_ph, output_pred
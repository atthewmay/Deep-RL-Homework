import tensorflow as tf

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



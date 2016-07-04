import tensorflow as tf
from tensorflow.python.ops.constant_op import constant
#from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import os
import kenlm
from tensorflow.contrib.ctc import ctc_loss

learning_rate = 0.001   # TODO: Determine a reasonable value for this
training_iters = 100000 # TODO: Determine a reasonable value for this
batch_size = 128        # TODO: Determine a reasonable value for this
display_step = 10       # TODO: Determine a reasonable value for this
dropout_rate = 0.05
relu_clip = 20 # TODO: Validate this is a reasonable value
n_steps = 500 # TODO: Determine this programatically from the longest speech sample
n_input = 160 # TODO: Determine this programatically from the sample rate
n_context = 5 # TODO: Determine the optimal value using a validation data set

n_hidden_1 = n_input + 2*n_input*n_context # Note: This value was not specified in the original paper
n_hidden_2 = n_input + 2*n_input*n_context # Note: This value was not specified in the original paper
n_hidden_5 = n_input + 2*n_input*n_context # Note: This value was not specified in the original paper

n_cell_dim = n_input + 2*n_input*n_context # TODO: Is this a reasonable value

n_hidden_3 = 2 * n_cell_dim
n_character = 29 # TODO: Determine if this should be extended with other punctuation
n_hidden_6 = n_character

x = tf.placeholder("float", [None, n_steps, n_input + 2*n_input*n_context])
y = tf.placeholder("string", [None, 1])

istate_fw = tf.placeholder("float", [None, 2*n_cell_dim])
istate_bw = tf.placeholder("float", [None, 2*n_cell_dim])

keep_prob = tf.placeholder(tf.float32)

# Store layers weight & bias
# TODO: Is random_normal the best distribution to draw from?
weights = {
    'h1': tf.Variable(tf.random_normal([n_input + 2*n_input*n_context, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h5': tf.Variable(tf.random_normal([(2 * n_cell_dim), n_hidden_5])),
    'h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b5': tf.Variable(tf.random_normal([n_hidden_5])),
    'b6': tf.Variable(tf.random_normal([n_hidden_6]))
}

def BiRNN(_X, _istate_fw, _istate_bw, _weights, _biases):
    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    _X = tf.transpose(_X, [1, 0, 2])  # Permute n_steps and batch_size
    # Reshape to prepare input for first layer
    _X = tf.reshape(_X, [-1, n_input + 2*n_input*n_context]) # (n_steps*batch_size, n_input + 2*n_input*n_context)

    #Hidden layer with clipped RELU activation and dropout
    layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])), relu_clip)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    #Hidden layer with clipped RELU activation and dropout
    layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])), relu_clip)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    #Hidden layer with clipped RELU activation and dropout
    layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3'])), relu_clip)
    layer_3 = tf.nn.dropout(layer_3, keep_prob)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=False)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=False)

    # Split data because rnn cell needs a list of inputs for the BRNN inner loop
    layer_3 = tf.split(0, n_steps, layer_3)

    # Get lstm cell output
    outputs = tf.nn.bidirectional_rnn(lstm_fw_cell,
                                    lstm_bw_cell,
                                    layer_3,
                                    initial_state_fw=_istate_fw,
                                    initial_state_bw=_istate_bw)

    # Reshape outputs from a list of n_steps tensors each of shape [batch_size, 2*n_cell_dim]
    # to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]
    outputs = tf.pack(outputs[0])
    outputs = tf.reshape(outputs, [-1, 2*n_cell_dim])

    #Hidden layer with clipped RELU activation and dropout
    layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, _weights['h5']), _biases['b5'])), relu_clip)
    layer_5 = tf.nn.dropout(layer_5, keep_prob)
    #Hidden layer with softmax function
    layer_6 = tf.nn.softmax(tf.add(tf.matmul(layer_5, _weights['h6']), _biases['b6']))

    # Reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to a tensor of shape [batch_size, n_steps, n_hidden_6]
    layer_6 = tf.reshape(layer_6, [n_steps, batch_size, n_hidden_6])
    layer_6 = tf.transpose(layer_6, [1, 0, 2])  # Permute n_steps and batch_size

    # Return layer_6
    return layer_6

layer_6 = BiRNN(x, istate_fw, istate_bw, weights, biases)

#CTC loss

def SimpleSparseTensorFrom(x):
    x_ix = []
    x_val = []
    for batch_i, batch in enumerate(x):
        for time, val in enumerate(batch):
            x_ix.append([batch_i, time])
            x_val.append(val)
    x_shape = [len(x), np.asarray(x_ix).max(0)[1]+1]
    x_ix = tf.constant(x_ix, tf.int64)
    x_val = tf.constant(x_val, tf.int32)
    x_shape = tf.constant(x_shape, tf.int64)
    return tf.SparseTensor(x_ix, x_val, x_shape)

labels = [[0]*n_steps]*batch_size         #read from y and initialize this list?
labels = SimpleSparseTensorFrom(labels)
loss = ctc_loss(layer_6, labels, [batch_size]*n_steps)
print loss

#TODO: check least loss and send that sentence to predict score through LM

# Language Model


LM = os.path.join(os.path.dirname(__file__), 'text.arpa')
model = kenlm.Model(LM)
print('{0}-gram model'.format(model.order))

sentence = 'predicted sentence here'
print(sentence)
print(model.score(sentence))

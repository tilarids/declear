import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import numpy as np
import prettytensor as pt

class CtcModel(object):
    def __init__(self, max_time_steps, num_features, num_classes):
        self._create_graph(num_hidden=128,
                           batch_size=4,
                           max_time_steps=max_time_steps,
                           num_features=num_features,
                           conv_depth=10,
                           num_classes=num_classes)

    def _create_graph(self, num_hidden, batch_size, max_time_steps, num_features, conv_depth, num_classes):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # e.g: log filter bank or MFCC features
            # Has size [batch_size, max_time_steps, num_features], but the
            # batch_size and max_stepsize can vary along each step
            self.inputs = tf.placeholder(tf.float32, [batch_size, max_time_steps, num_features])

            # Here we use sparse_placeholder that will generate a
            # SparseTensor required by ctc_loss op.
            self.targets = tf.sparse_placeholder(tf.int32)

            # 1d array of size [batch_size]
            self.seq_len = tf.placeholder(tf.int32, [batch_size])

            # Defining the cell
            # Can be:
            #   tf.nn.rnn_cell.RNNCell
            #   tf.nn.rnn_cell.GRUCell
            cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

            inputsW = (pt.wrap(tf.expand_dims(self.inputs, -1))
                         .conv2d(3, conv_depth, activation_fn=tf.nn.relu)
                         .conv2d(3, conv_depth, activation_fn=tf.nn.relu))
            # The second output is the last state and we will no use that
            inputsW = tf.reshape(inputsW, [batch_size, max_time_steps, num_features * conv_depth])
            outputs, _ = tf.nn.dynamic_rnn(cell, inputsW, self.seq_len, dtype=tf.float32)
            # outputs, _ = tf.nn.dynamic_rnn(cell, inputs, self.seq_len, dtype=tf.float32)

            shape = tf.shape(self.inputs)
            batch_s, max_timesteps = shape[0], shape[1]

            # Reshaping to apply the same weights over the timesteps
            outputs = tf.reshape(outputs, [-1, num_hidden])

            # Truncated normal with mean 0 and stdev=0.1
            # Tip: Try another initialization
            # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
            W = tf.Variable(tf.truncated_normal([num_hidden,
                                                 num_classes],
                                                stddev=0.1))
            # Zero initialization
            # Tip: Is tf.zeros_initializer the same?
            b = tf.Variable(tf.constant(0., shape=[num_classes]))

            # Doing the affine projection
            logits = tf.matmul(outputs, W) + b

            # Reshaping back to the original shape
            logits = tf.reshape(logits, [batch_s, -1, num_classes])

            # Time major
            logits = tf.transpose(logits, (1, 0, 2))

            self.loss = tf.reduce_mean(ctc.ctc_loss(logits, self.targets, self.seq_len))
            self.logitsMaxTest = tf.slice(tf.argmax(logits, 2), [0, 0], [self.seq_len[0], 1])

            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

            self.predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits, self.seq_len)[0][0])

            self.error_rate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targets, normalize=False)) / \
                        tf.to_float(tf.size(self.targets.values))

            tf.scalar_summary('loss', self.loss)
            tf.scalar_summary('error_rate', self.error_rate)
            self.merged_summaries = tf.merge_all_summaries()


    def run_predictions(self, session, inputs, seq_len):
        return session.run(self.predictions, feed_dict={self.inputs: inputs, self.seq_len: seq_len})

    def run_train_step(self, session, inputs, seq_len, targets):
        fd = {self.inputs: inputs, self.targets: targets, self.seq_len: seq_len}
        return session.run([self.optimizer, self.loss, self.error_rate, self.logitsMaxTest, self.predictions, self.merged_summaries], feed_dict=fd)



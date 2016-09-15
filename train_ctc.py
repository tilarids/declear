# Adapted from tensorflow_CTC_example.

from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import numpy as np
import prettytensor as pt
# import cv2
import json
# from utils import load_batched_data

with open("model_input.json", "r") as fp:
    model_input = json.load(fp)

def target_list_to_sparse_tensor(targetList):
    '''make tensorflow SparseTensor from list of targets, with each element
       in the list being a list or array with the values of the target sequence
       (e.g., the integer values of a character map for an ASR target string)
       See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ctc/ctc_loss_op_test.py
       for example of SparseTensor format'''
    indices = []
    vals = []
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(0)[1]+1]
    return (np.array(indices), np.array(vals), np.array(shape))

def test_edit_distance():
    graph = tf.Graph()
    with graph.as_default():
        truth = tf.sparse_placeholder(tf.int32)
        hyp = tf.sparse_placeholder(tf.int32)
        editDist = tf.edit_distance(hyp, truth, normalize=False)

    with tf.Session(graph=graph) as session:
        truthTest = sparse_tensor_feed([[0,1,2], [0,1,2,3,4]])
        hypTest = sparse_tensor_feed([[3,4,5], [0,1,2,2]])
        feedDict = {truth: truthTest, hyp: hypTest}
        dist = session.run([editDist], feed_dict=feedDict)
        print(dist)

def data_lists_to_batches(inputList, targetList, batchSize):
    '''Takes a list of input matrices and a list of target arrays and returns
       a list of batches, with each batch being a 3-element tuple of inputs,
       targets, and sequence lengths.
       inputList: list of 2-d numpy arrays with dimensions nFeatures x timesteps
       targetList: list of 1-d arrays or lists of ints
       batchSize: int indicating number of inputs/targets per batch
       returns: dataBatches: list of batch data tuples, where each batch tuple (inputs, targets, seqLengths) consists of
                    inputs = 3-d array w/ shape batchSize x nTimeSteps x nFeatures
                    targets = tuple required as input for SparseTensor
                    seqLengths = 1-d array with int number of timesteps for each sample in batch
                maxSteps: maximum number of time steps across all samples'''
    # import pdb; pdb.set_trace()
    assert len(inputList) == len(targetList)
    nFeatures = inputList[0].shape[0]
    maxSteps = 0
    for inp in inputList:
        maxSteps = max(maxSteps, inp.shape[1])

    randIxs = np.random.permutation(len(inputList))
    start, end = (0, batchSize)
    dataBatches = []

    while end <= len(inputList):
        batchSeqLengths = np.zeros(batchSize)
        for batchI, origI in enumerate(randIxs[start:end]):
            batchSeqLengths[batchI] = inputList[origI].shape[-1]
        batchInputs = np.zeros((batchSize, maxSteps, nFeatures))
        batchTargetList = []
        for batchI, origI in enumerate(randIxs[start:end]):
            padSecs = maxSteps - inputList[origI].shape[1]
            assert padSecs == 0
            batchInputs[batchI,:,:] = inputList[origI].T
            #batchInputs[:,batchI,:] = np.pad(inputList[origI].T, ((0,padSecs),(0,0)),
            #                                 'constant', constant_values=0)
            batchTargetList.append(targetList[origI])
        dataBatches.append((batchInputs, target_list_to_sparse_tensor(batchTargetList),
                          batchSeqLengths))
        start += batchSize
        end += batchSize
    return (dataBatches, maxSteps)

def load_batched_data(model_input, batchSize):
    '''returns 3-element tuple: batched data (list), max # of time steps (int), and
       total number of samples (int)'''
    CLASSES_DICT = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '.': 10, ',': 10}

    inputs = []
    targets = []
    def add_sample(img_path, target_str):
        if not target_str:
            return
        inputs.append(np.load(img_path))
        targets.append(np.array([CLASSES_DICT[x] for x in target_str]))

    for record in model_input:
        # import cv2
        # cv2.imwrite('/tmp/dd/' + record['income'] + ".png", np.load(record['first_bin']))
        add_sample(record['first_bin'], record['income'])
        add_sample(record['second_bin'], record['family_income'])

    # import pdb; pdb.set_trace()
    # return
    return data_lists_to_batches(inputs, targets, batchSize) + (len(inputs), )

# INPUT_PATH = './sample_data/mfcc' #directory of MFCC nFeatures x nFrames 2-D array .npy files
# TARGET_PATH = './sample_data/char_y/' #directory of nCharacters 1-D array .npy files

####Learning Parameters
learningRate = 0.001
momentum = 0.9
nEpochs = 3000
batchSize = 4

####Network Parameters
nFeatures = 22 #22 is the height of the image
nHidden = 128
nClasses = 12 #10 digits plus dot plus the "blank" for CTC

####Load data
print('Loading data')
batchedData, maxTimeSteps, totalN = load_batched_data(model_input, batchSize)

####Define graph
print('Defining graph')
graph = tf.Graph()
with graph.as_default():

    # e.g: log filter bank or MFCC features
    # Has size [batch_size, max_stepsize, num_features], but the
    # batch_size and max_stepsize can vary along each step
    inputs = tf.placeholder(tf.float32, [batchSize, maxTimeSteps, nFeatures])

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32)

    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [batchSize])

    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    cell = tf.nn.rnn_cell.LSTMCell(nHidden, state_is_tuple=True)

    # Stacking rnn cells
    # num_layers = 1
    # stack = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers,
    #                                     state_is_tuple=True)

    inputsW = (pt.wrap(tf.expand_dims(inputs, -1))
                 .conv2d(3, 10, activation_fn=tf.nn.relu)
                 .conv2d(3, 10, activation_fn=tf.nn.relu))
    # The second output is the last state and we will no use that
    inputsW = tf.reshape(inputsW, [batchSize, maxTimeSteps, nFeatures * 10])
    outputs, _ = tf.nn.dynamic_rnn(cell, inputsW, seq_len, dtype=tf.float32)
    # outputs, _ = tf.nn.dynamic_rnn(cell, inputs, seq_len, dtype=tf.float32)

    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, nHidden])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.truncated_normal([nHidden,
                                         nClasses],
                                        stddev=0.1))
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[nClasses]))

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, nClasses])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    loss = tf.reduce_mean(ctc.ctc_loss(logits, targets, seq_len))
    logitsMaxTest = tf.slice(tf.argmax(logits, 2), [0, 0], [seq_len[0], 1])

    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits, seq_len)[0][0])

    # Inaccuracy: label error rate
    errorRate = tf.reduce_sum(tf.edit_distance(predictions, targets, normalize=False)) / \
                tf.to_float(tf.size(targets.values))
    # ler = tf.reduce_mean(tf.edit_distance(tf.cast(predictions[0], tf.int32),
    #                                       targets))

####Run session
with tf.Session(graph=graph) as session:
    print('Initializing')
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    if False:
        print("Restoring state...")
        saver.restore(session, './train/save')
        # import pdb; pdb.set_trace()
        batchedDataPred, _, _ = load_batched_data([{
                "income": ".",
                "family_income": ".",
                "first_bin": "/Users/tilarids/dev/decl/extract_img_data/0278d69820395cf130f098f79b46caa62023627a9a7362295e2c5489.pdf.1.png.first.bin.npy",
                "second_bin": "/Users/tilarids/dev/decl/extract_img_data/0278d69820395cf130f098f79b46caa62023627a9a7362295e2c5489.pdf.1.png.second.bin.npy"
            }] * batchSize, batchSize)
        batchInputs, batchTargetSparse, batchSeqLengths = batchedDataPred[0]
        pred = session.run(predictions, feed_dict={inputs: batchInputs, seq_len: batchSeqLengths})
        predDense = session.run(tf.sparse_to_dense(pred.indices, pred.shape, pred.values))
        # print("Prediction:", pred.values)
        print("Prediction:", predDense)
        print("Expected prediction:", batchTargetSparse[1])
        sys.exit(0)
    for epoch in range(nEpochs):
        print('Epoch', epoch+1, '...')
        batchErrors = np.zeros(len(batchedData))
        batchRandIxs = np.random.permutation(len(batchedData)) #randomize batch order
        for batch, batchOrigI in enumerate(batchRandIxs):
            batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
            # batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
            feedDict = {inputs: batchInputs, targets: batchTargetSparse, seq_len: batchSeqLengths}
            _, l, er, lmt, pr = session.run([optimizer, loss, errorRate, logitsMaxTest, predictions], feed_dict=feedDict)
            print(np.unique(lmt)) #print unique argmax values of first sample in batch; should be blank for a while, then spit out target values
            # print(pr, batchTargetSparse)
            if (batch % 1) == 0:
                print('Minibatch', batch, '/', batchOrigI, 'loss:', l)
                print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
            batchErrors[batch] = er*len(batchSeqLengths)
        epochErrorRate = batchErrors.sum() / totalN
        print('Epoch', epoch+1, 'error rate:', epochErrorRate)
        if 1 == epoch % 50:
            print("Saving state...")
            saver.save(session, './train/save')

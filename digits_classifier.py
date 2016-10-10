import json

import tensorflow as tf
import prettytensor as pt
import numpy as np

tf.app.flags.DEFINE_string(
    'save_path', None, 'Where to save the model checkpoints.')
FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = 4
SEED = 1
EPOCHS = 10

np.random.seed(SEED)
tf.set_random_seed(SEED)

def prepare_data():
    with open("digits_classifier_input.json", "r") as fp:
        digits_classifier_input = json.load(fp)
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for record in digits_classifier_input:
        if record.get('label', None) is None:
            continue
        img = np.load(record['bin_path'])
        img = img - 255.0 / 2.0
        img /= 255.0

        label = np.array([1.0, 0.0])
        if record['label']=='space':  # positive label
            label = np.array([0.0, 1.0])
        if np.random.randint(0,100) >= 80:  # split train/test 80/20
            test_images.append(img)
            test_labels.append(label)
        else:
            train_images.append(img)
            train_labels.append(label)
    if len(train_images) % BATCH_SIZE:
        train_images = train_images[:-(len(train_images) % BATCH_SIZE)]
        train_labels = train_labels[:-(len(train_labels) % BATCH_SIZE)]
    if len(test_images) % BATCH_SIZE:
        test_images = test_images[:-(len(test_images) % BATCH_SIZE)]
        test_labels = test_labels[:-(len(test_labels) % BATCH_SIZE)]
    print "Data loaded. %d train records and %d test records" % (len(train_images), len(test_images))
    # print train_images[0]
    return map(np.array, [train_images, train_labels, test_images, test_labels])

def permute_data(arrays, random_state=None):
  """Permute multiple numpy arrays with the same order."""
  if any(len(a) != len(arrays[0]) for a in arrays):
    raise ValueError('All arrays must be the same length.')
  if not random_state:
    random_state = np.random
  order = random_state.permutation(len(arrays[0]))
  return [a[order] for a in arrays]

def main(_=None):
    image_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, 22, 95])
    labels_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, 2])

    # Create our model.  The result of softmax_classifier is a namedtuple
    # that has members result.loss and result.softmax.
    images = pt.wrap(tf.expand_dims(image_placeholder, -1))
    with pt.defaults_scope(activation_fn=tf.nn.relu, l2loss=0.00001):
        result = (images
                    .conv2d(5, 20)
                    # .max_pool(2, 2)
                    .conv2d(5, 50)
                    # .max_pool(2, 2)
                    .flatten()
                    .fully_connected(500)
                    .dropout(0.5)
                    .softmax_classifier(2, labels_placeholder))

    accuracy = result.softmax.evaluate_classifier(labels_placeholder,
                                                  phase=pt.Phase.test)

    # Grab the data as numpy arrays.
    train_images, train_labels, test_images, test_labels = prepare_data()

    # Create the gradient optimizer and apply it to the graph.
    # pt.apply_optimizer adds regularization losses and sets up a step counter
    # (pt.global_step()) for you.
    optimizer = tf.train.AdamOptimizer()
    train_op = pt.apply_optimizer(optimizer, losses=[result.loss])

    # We can set a save_path in the runner to automatically checkpoint every so
    # often.  Otherwise at the end of the session, the model will be lost.
    runner = pt.train.Runner(save_path=FLAGS.save_path)
    with tf.Session():
        print('Initializing')
        tf.initialize_all_variables().run()
        for epoch in xrange(EPOCHS):
            # Shuffle the training data.
            train_images, train_labels = permute_data(
                (train_images, train_labels))

            runner.train_model(
                train_op,
                result.loss,
                len(train_images),
                feed_vars=(image_placeholder, labels_placeholder),
                feed_data=pt.train.feed_numpy(BATCH_SIZE, train_images, train_labels),
                print_every=100)
            classification_accuracy = runner.evaluate_model(
                accuracy,
                len(test_images),
                feed_vars=(image_placeholder, labels_placeholder),
                feed_data=pt.train.feed_numpy(BATCH_SIZE, test_images, test_labels))
            print('Accuracy after %d epoch %g%%' % (
                epoch + 1, classification_accuracy * 100))


if __name__ == '__main__':
  tf.app.run()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import dataset

CLASSES = 2
EGS_PER_CLASS = 2000

IMAGE_SIZE = 28
FILTER_SIZE = 20

BATCH_SIZE = 10
TRAIN_STEPS = 2000
STEPS_PER_DRAW = 50

LEARNING_RATES = np.logspace(-3, 3, num=7).ravel()
MOMENTUMS = np.linspace(0, 1, num=5).ravel()


def get_graph(lr, momentum, scope_name):
    with tf.variable_scope(scope_name):
        X = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE])
        y = tf.placeholder(tf.int32, shape=[None, CLASSES])

        W = tf.get_variable('W',
                            [FILTER_SIZE, FILTER_SIZE, 1, 1],
                            initializer=tf.random_normal_initializer(seed=1))

        conv_input = tf.reshape(X, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
        conv_output = tf.nn.sigmoid(
            tf.nn.conv2d(conv_input, W, [1, 1, 1, 1], 'SAME'))

        flat_conv_output = tf.reshape(conv_output,
                                      [-1, IMAGE_SIZE * IMAGE_SIZE])
        y_hat = tf.layers.dense(flat_conv_output, CLASSES)

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
        train_step = tf.train.MomentumOptimizer(lr, momentum).minimize(loss)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_hat, axis=1),
                                                   tf.argmax(y, axis=1)),
                                          tf.float32))

        return X, y, W, y_hat, loss, train_step, accuracy


def gen_batch(X_train, y_train):
    assert X_train.shape[0] == y_train.shape[0]
    n = X_train.shape[0]

    indices = np.random.permutation(n)
    X_train, y_train = X_train[indices], y_train[indices]

    i = 0
    while i + BATCH_SIZE <= n:
        yield X_train[i:i + BATCH_SIZE], y_train[i:i + BATCH_SIZE]
        i += BATCH_SIZE
    yield from gen_batch(X_train, y_train)

with tf.Session() as sess:
    np.random.seed(42)

    X_train, X_test, y_train, y_test = dataset.load(egs_per_class=EGS_PER_CLASS)

    all_results = []
    for lr in LEARNING_RATES:
        for momentum in MOMENTUMS:
            scope_name = 'lr{}-momentum{}'.format(lr, momentum)
            X, y, W, y_hat, loss, train_step, accuracy = get_graph(
                lr, momentum, scope_name)

            sess.run(tf.global_variables_initializer())

            results = []  # Results for this optimizer
            batches = gen_batch(X_train, y_train)
            for i in range(TRAIN_STEPS):
                X_batch, y_batch = next(batches)
                if i % STEPS_PER_DRAW == 0:
                    curr_W, curr_accuracy = sess.run(
                        [W, accuracy], feed_dict={X: X_test, y: y_test})
                    results.append(curr_W)
#                    print('Iteration {} for {} has accuracy {:.3f}'.format(
#                        i, scope_name, curr_accuracy))
                sess.run(train_step, feed_dict={X: X_batch, y: y_batch})
            all_results.append(results)

            print('Finished {}'.format(scope_name))

for draw_step, draw_step_results in enumerate(zip(*all_results)):
    fig = plt.figure()
    step = draw_step * STEPS_PER_DRAW

    i = 0
    for lr in LEARNING_RATES:
        for momentum in MOMENTUMS:
            curr_W = draw_step_results[i].reshape([FILTER_SIZE, FILTER_SIZE])
            ax = fig.add_subplot(len(LEARNING_RATES), len(MOMENTUMS), i + 1)
            ax.imshow(curr_W, cmap='Blues', interpolation='nearest')

            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            if i // len(MOMENTUMS) == 0:
                ax.xaxis.set_label_position('top')
                ax.set_xlabel(momentum, fontsize=10)
            if i % len(MOMENTUMS) == 0:
                ax.set_ylabel('{:.0E}'.format(lr), fontsize=10)

            i += 1

    fig.text(0.5, 0.98, 'Momentum Amount',
             horizontalalignment='center', verticalalignment='top')
    fig.text(0.5, 0.02, 'Iteration {}'.format(step),
             horizontalalignment='center', verticalalignment='top')
    fig.text(0.02, 0.5, 'Learning Rate',
             horizontalalignment='left', verticalalignment='center',
             rotation='vertical')

    plt.savefig('frame-{:03}.png'.format(draw_step))
    plt.close(fig)

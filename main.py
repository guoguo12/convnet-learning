import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import dataset

CLASSES = 2
EGS_PER_CLASS = 4000

IMAGE_SIZE = 28
FILTER_SIZE = 22

BATCH_SIZE = 10
TRAIN_STEPS = 2500
STEPS_PER_DRAW = 25

LEARNING_RATE = 0.1
MOMENTUM = 0.75


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

lr, momentum = LEARNING_RATE, MOMENTUM
with tf.Session() as sess:
    np.random.seed(42)

    X_train, X_test, y_train, y_test = dataset.load(egs_per_class=EGS_PER_CLASS)

    scope_name = 'lr{}-momentum{}'.format(lr, momentum)
    X, y, W, y_hat, loss, train_step, accuracy = get_graph(
        lr, momentum, scope_name)

    sess.run(tf.global_variables_initializer())

    results = []  # Results for this optimizer
    batches = gen_batch(X_train, y_train)
    for i in range(TRAIN_STEPS):
        X_batch, y_batch = next(batches)
        if i % STEPS_PER_DRAW == 0:
            results.append(
                sess.run(W, feed_dict={X: X_test, y: y_test}))
            print('Iteration {} for {}'.format(i, scope_name))
        sess.run(train_step, feed_dict={X: X_batch, y: y_batch})

    print('Finished {}'.format(scope_name))

for draw_step, curr_W in enumerate(results):
    fig = plt.figure(figsize=(4, 4))
    step = draw_step * STEPS_PER_DRAW

    curr_W = curr_W.reshape([FILTER_SIZE, FILTER_SIZE])
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.imshow(curr_W, cmap='Blues', interpolation='nearest')

    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig('frame-{:03}.png'.format(draw_step))
    plt.close(fig)

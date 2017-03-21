import tensorflow as tf
import numpy as np
from tflearn.datasets import cifar10
import matplotlib.pyplot as plt

def _init_conv(dims, name):
    return tf.get_variable(name+'_conv', shape = dims,
                           initializer = tf.contrib.layers.xavier_initializer())

def _init_weight(dims, name):
    return tf.get_variable(name+'_bias', shape = dims,
                           initializer = tf.contrib.layers.xavier_initializer())

def _conv2d(input, filter):
    return tf.nn.conv2d(input, filter,
                        strides=[1, 1, 1, 1], padding='SAME')

def _res_block(input, dims, name):
    W1 = _init_conv(dims, name+'1')
    # b1 = _init_weight([64], name+'1')
    res_1 = tf.nn.relu(_conv2d(input, W1))

    W2 = _init_conv(dims, name+'2')
    # b2 = _init_weight([64], name+'2')
    res_2 = tf.nn.relu(_conv2d(res_1, W2))

    return tf.nn.relu(res_2 + input)
    
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

def next_batch(num):
    i = 0

    while True:
        if i+num > len(X_train):
            i = 0

        i+=num
        yield (X_train[i-num:i], Y_train[i-num:i])


# sm_y = sm_y.reshape([10, 1])
# x = X_train[:10] # grab first 10
# y_ = Y_train[:10]

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.int64, shape=[None])

# Before magic
W_init = _init_weight([32, 32, 3, 16], "w_init")
init_out = tf.nn.relu(_conv2d(x, W_init))

# Residual block 1
res1 = _res_block(init_out, [32, 32, 16, 16], "16_residual")

# Residual block 2
res1_W = _init_weight([1, 1, 16, 32], "res2_W") # Match the dimensions!
res1_out = tf.nn.relu(_conv2d(res1, res1_W))
res2 = _res_block(res1_out, [32, 32, 32, 32], "32_residual")

# Residual block 3
res2_W = _init_weight([1, 1, 32, 64], "res3_W") # Match the dimensions!
res2_out = tf.nn.relu(_conv2d(res2, res2_W))
res3 = _res_block(res2_out, [32, 32, 64, 64], "64_residual")

# FC
avg1 = tf.nn.avg_pool(res3, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
avg1_reshape = tf.reshape(avg1, [-1, 16*16*64])

W_fc = _init_weight([16*16*64, 1024], "W_fc1")
b_fc = _init_weight([1024], "b_fc1")
fc1 = tf.matmul(avg1_reshape, W_fc) + b_fc

W_out = _init_weight([1024, 10], "W_out")
b_out = _init_weight([10], "b_out")
out = tf.matmul(fc1, W_out) + b_out

cross_entropy = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=out)
)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.cast(tf.argmax(out, 1), tf.int64), y_)
preds = tf.argmax(out, 1)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.Session()
sess.run(tf.global_variables_initializer())


MOD_PARAM = 1000
ITERATIONS = 1000000
BATCH_SIZE = 100

tr_accuracies = np.zeros(ITERATIONS / MOD_PARAM)
te_accuracies = np.zeros(ITERATIONS / MOD_PARAM)

batches = next_batch(BATCH_SIZE)

with sess.as_default():
    for i in xrange(ITERATIONS):

        batch_x, batch_y = batches.next()

        if i % MOD_PARAM == 0:
            train_accuracy = accuracy.eval(feed_dict = {
                x: batch_x, y_: batch_y
            })

            tr_accuracies[i / MOD_PARAM] = train_accuracy

            i = 0
            total = 0
            for n in xrange(0, len(Y_test), BATCH_SIZE):
                test_accuracy = accuracy.eval(feed_dict = {
                    x: X_test[n:n+BATCH_SIZE], y_: Y_test[n:n+BATCH_SIZE]
                })

                total += test_accuracy

            te_accuracies[i / MOD_PARAM] = total / (len(Y_test) / BATCH_SIZE)

            print "Iteration: %d - Training Accuracy: %f - Test Accuracy: %f" % (i,
                                                                                 train_accuracy,
                                                                                 test_accuracy)

        train_step.run(feed_dict={x: batch_x, y_: batch_y})

plt.plot(tr_accuracies, color = 'red', label='training')
plt.plot(te_accuracies, color = 'blue', label='test')
plt.savefig("progress.png", dpi=1000)

from datetime import datetime
import numpy as np
import math
import time
from random import randint

import tensorflow.python.platform
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,"""Batch size.""")
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_string('data_format', 'NHWC',"""Data format for conv""")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

param_servers = ["10.40.1.138:2222"]
workers = ["10.40.1.138:2223","10.40.1.140:2222"]
#            "10.40.1.140:2222","10.40.1.140:2223"]

cluster = tf.train.ClusterSpec({"ps":param_servers, "worker":workers})

server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

def _conv(inpOp, nIn, nOut, kH, kW, dH, dW, padType):
#    with tf.name_scope(name) as scope:
    kernel = tf.Variable(tf.truncated_normal([kH, kW, nIn, nOut],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    if FLAGS.data_format == 'NCHW':
      strides = [1, 1, dH, dW]
    else:
      strides = [1, dH, dW, 1]
    conv = tf.nn.conv2d(inpOp, kernel, strides, padding=padType,
                        data_format=FLAGS.data_format)
    biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias)
    return conv1

def _affine(inpOp, nIn, nOut):
#    with tf.name_scope(name) as scope:
    kernel = tf.Variable(tf.truncated_normal([nIn, nOut],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                         trainable=True, name='biases')
    affine1 = tf.nn.relu_layer(inpOp, kernel, biases)
    return affine1

def _mpool(inpOp, kH, kW, dH, dW):
    if FLAGS.data_format == 'NCHW':
      ksize = [1, 1, kH, kW]
      strides = [1, 1, dH, dW]
    else:
      ksize = [1, kH, kW, 1]
      strides = [1, dH, dW, 1]
    return tf.nn.max_pool(inpOp,
                          ksize=ksize,
                          strides=strides,
                          padding='VALID',
                          data_format=FLAGS.data_format)

def loss(logits, labels):
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(
        concated, tf.pack([batch_size, 1000]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                            onehot_labels,
                                                            name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

def inference(images):
    conv1 = _conv (images, 3, 96, 11, 11, 4, 4, 'VALID')
    pool1 = _mpool(conv1,  2, 2, 2, 2)
    conv2 = _conv(pool1, 96, 256, 5, 5, 1, 1, 'VALID')
    pool2 = _mpool(conv2,  2, 2, 2, 2)
    conv3 = _conv (pool2,  256, 512, 3, 3, 1, 1, 'SAME')
    conv4 = _conv (conv3,  512, 1024, 3, 3, 1, 1, 'SAME')
    conv5 = _conv (conv4,  1024, 1024, 3, 3, 1, 1, 'SAME')
    pool5 = _mpool(conv5,  2, 2, 2, 2)
    resh1 = tf.reshape(pool5, [-1, 1024 * 6 * 6])
    affn1 = _affine(resh1, 1024 * 6 * 6, 3072)
    affn2 = _affine(affn1, 3072, 4096)
    affn3 = _affine(affn2, 4096, 1000)

    return affn3

def fake_data(batch_size):
    """Generate a fake dataset that matches the dimensions of ImageNet."""

    data = np.random.rand(batch_size,231,231,3)
    labels = np.zeros((batch_size,1000),dtype=np.int)
    for i in xrange(batch_size):
        for j in xrange(randint(0,1000)):
            labels[i][j] = 1

    return data, labels

def run_benchmark():
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
        # Generate some dummy images.
            image_size = 231 
            # Note that our padding definition is slightly different the cuda-convnet.
            # In order to force the model to start with the same activations sizes,
            # we add 3 to the image_size and employ VALID padding above.
            image_shape = [FLAGS.batch_size, image_size, image_size, 3]

            images = tf.placeholder(tf.float32, [None, 231, 231, 3])
            labels = tf.placeholder(tf.float32, [None, 1000])

            global_step = tf.Variable(0)
            pred = inference(images)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(pred,
                                                                    labels,
                                                                    name='xentropy')
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss,
                                                global_step=global_step)
#
            X,Y = fake_data(FLAGS.batch_size)
            print (X.shape)
            print (Y.shape)

            init = tf.initialize_all_variables()
            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0), 
                                            global_step=global_step,
                                            init_op = init)

            with sv.managed_session(server.target) as sess:
                step = 0
                begin_time = time.time()
                while step < 50:
                    start = time.time()
                    _,step = sess.run([optimizer,global_step], feed_dict={images: X, labels: Y})
                    elapsed = time.time() - start
#                    if step % 100 == 0:
                    print ("Step %d" % step, "Elapsed %.4f" % (elapsed*1000.0))
                print ("Total time %.4f" % ((time.time() - begin_time) * 1000.0))
                sv.stop()

def main(_):
  run_benchmark()

if __name__ == '__main__':
  tf.app.run()

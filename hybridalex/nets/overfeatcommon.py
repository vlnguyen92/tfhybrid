from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def overfeat_loss(logits, labels, scope=None):
    """Build objective function"""
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels,
                                                                   name='cross_entropy_batch')
    total_loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    tf.add_to_collection(tf.GraphKeys.LOSSES, total_loss)
    return total_loss


def overfeat_part_conv(builder, images):
    net = builder.conv('conv1', images, 96, [11, 11], stride=4, padding='VALID',
                       activation=tf.nn.relu)
    net = builder.max_pool('pool1', net, 2, stride=2)

    net = builder.conv('conv2', net, 256, [5, 5], stride=1, padding='VALID',
                       activation=tf.nn.relu)
    net = builder.max_pool('pool2', net, 2, stride=2)

    net = builder.conv('conv3', net, 512, [3, 3], stride=1, padding='SAME',
                       activation=tf.nn.relu)
    net = builder.conv('conv4', net, 1024, [3, 3], stride=1, padding='SAME',
                       activation=tf.nn.relu)
    net = builder.conv('conv5', net, 1024, [3, 3], stride=1, padding='SAME',
                       activation=tf.nn.relu)
    net = builder.max_pool('pool5', net, 2, stride=2)

    return net


def overfeat_inference(builder, images, labels, num_classes, scope=None):
    net = overfeat_part_conv(builder, images)

    net = builder.fc('fc3072', net, 3072, activation=tf.nn.relu)

    net = builder.fc('fc4096', net, 4096, activation=tf.nn.relu)

    net = builder.fc('fc1000', net, num_classes, activation=tf.nn.relu)

    logits = net

    with tf.name_scope('probs'):
        net = tf.nn.softmax(net)

    return net, logits, overfeat_loss(logits, labels, scope)


def overfeat_eval(probs, labels):
    """Evaluate, returns number of correct images"""
    with tf.name_scope('evaluation'):
        correct = tf.nn.in_top_k(probs, labels, k=1)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

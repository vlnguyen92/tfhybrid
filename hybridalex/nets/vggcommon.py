from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def vgg_loss(logits, labels, scope=None):
    """Build objective function"""
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels,
                                                                   name='cross_entropy_batch')
    total_loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    tf.add_to_collection(tf.GraphKeys.LOSSES, total_loss)
    return total_loss


def vgg_part_conv(builder, images):
    net = builder.conv('conv1', images, 64, [3, 3], stride=1, padding='SAME',
                       activation=tf.nn.relu)
    net = builder.max_pool('pool1', net, 2, stride=2)

    net = builder.conv('conv2', net, 128, [3, 3], stride=1, padding='SAME',
                       activation=tf.nn.relu)
    net = builder.max_pool('pool2', net, 2, stride=2)

    net = builder.conv('conv3', net, 256, [3, 3], stride=1, padding='SAME',
                       activation=tf.nn.relu)
    net = builder.conv('conv4', net, 256, [3, 3], stride=1, padding='SAME',
                       activation=tf.nn.relu)
    net = builder.max_pool('pool4', net, 2, stride=2)

    net = builder.conv('conv5', net, 512, [3, 3], stride=1, padding='SAME',
                       activation=tf.nn.relu)
    net = builder.conv('conv6', net, 512, [3, 3], stride=1, padding='SAME',
                       activation=tf.nn.relu)
    net = builder.max_pool('pool6', net, 2, stride=2)

    net = builder.conv('conv7', net, 512, [3, 3], stride=1, padding='SAME',
                       activation=tf.nn.relu)
    net = builder.conv('conv8', net, 512, [3, 3], stride=1, padding='SAME',
                       activation=tf.nn.relu)
    net = builder.max_pool('pool8', net, 2, stride=2)

    return net


def vgg_inference(builder, images, labels, num_classes, scope=None):
    net = vgg_part_conv(builder, images)

    net = builder.fc('fc4096a', net, 4096,
                     activation=tf.nn.relu)

    net = builder.fc('fc4096b', net, 4096,
                     activation=tf.nn.relu)

    net = builder.fc('fc1000', net, num_classes,
                     activation=tf.nn.relu)

    logits = net

    with tf.name_scope('probs'):
        net = tf.nn.softmax(net)

    return net, logits, vgg_loss(logits, labels, scope)


def vgg_eval(probs, labels):
    """Evaluate, returns number of correct images"""
    with tf.name_scope('evaluation'):
        correct = tf.nn.in_top_k(probs, labels, k=1)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

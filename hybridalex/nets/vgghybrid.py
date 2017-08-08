from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .common import ModelBuilder
from .vggcommon import vgg_part_conv, vgg_inference, vgg_loss, vgg_eval


def original(images, labels, num_classes, total_num_examples, devices=None, is_train=True):
    """Build inference"""
    if devices is None:
        devices = [None]

    with tf.device(devices[0]):
        builder = ModelBuilder()
        net, logits, total_loss = vgg_inference(builder, images, labels, num_classes)

        if not is_train:
            return vgg_eval(net, labels)

        global_step = builder.ensure_global_step()
        # Compute gradients
        opt = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = opt.minimize(total_loss, global_step=global_step)

    return net, logits, total_loss, train_op, global_step


def ndev_data(images, labels, num_classes, total_num_examples, devices, is_train=True):
    """Build inference, data parallelism"""
    # use the last device in list as variable device
    devices = devices[:]
    builder = ModelBuilder(devices.pop())

    if not is_train:
        with tf.variable_scope('model'):
            prob = vgg_inference(builder, images, labels, num_classes)[0]
        return vgg_eval(prob, labels)

    global_step = builder.ensure_global_step()
    opt = tf.train.AdamOptimizer(learning_rate=0.01)

    # construct each replica
    replica_grads = []
    with tf.device(builder.variable_device()):
        image_slices = tf.split(0, len(devices), images)
        label_slices = tf.split(0, len(devices), labels)
    with tf.variable_scope('model') as vsp:
        # we only want scope for variables but not operations
        with tf.name_scope(''):
            for idx in range(len(devices)):
                dev = devices[idx]
                with tf.name_scope('tower_{}'.format(idx)) as scope:
                    with tf.device(dev):
                        prob, logits, total_loss = vgg_inference(builder, image_slices[idx],
                                                                 label_slices[idx], num_classes,
                                                                 scope)
                        # calculate gradients for batch in this replica
                        grads = opt.compute_gradients(total_loss)

                replica_grads.append(grads)
                # reuse variable for next replica
                vsp.reuse_variables()

    # average gradients across replica
    with tf.device(builder.variable_device()):
        grads = builder.average_gradients(replica_grads)
        apply_grads_op = opt.apply_gradients(grads, global_step=global_step)

        train_op = tf.group(apply_grads_op, name='train')

    # simply return prob, logits, total_loss from the last replica for simple evaluation
    return prob, logits, total_loss, train_op, global_step


def ndev_model(images, labels, num_classes, total_num_examples, devices, is_train=True):
    """Build inference"""
    builder = ModelBuilder()

    with builder.parallel(devices, colocate_variables=True):
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

    with builder.parallel(devices, colocate_variables=True):
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

    with builder.device(devices[0], colocate_variables=True):
        net = builder.fc('fc4096a', net, 4096,
                         activation=tf.nn.relu)

    with builder.device(devices[0], colocate_variables=True):
        net = builder.fc('fc4096b', net, 4096,
                         activation=tf.nn.relu)

    with builder.device(devices[0], colocate_variables=True):
        net = builder.fc('fc1000', net, num_classes, activation=tf.nn.relu)

    # save the unscaled logits for training
    logits = net

    with tf.device(devices[0]):
        with tf.name_scope('probs'):
            prob = tf.nn.softmax(net)

    if not is_train:
        return vgg_eval(prob, labels)

    with tf.device(devices[0]):
        total_loss = vgg_loss(logits, labels)

    # Compute gradients
    global_step = builder.ensure_global_step()
    opt = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = opt.minimize(total_loss, colocate_gradients_with_ops=True, global_step=global_step)

    return prob, logits, total_loss, train_op, global_step


def ndev_model_data(images, labels, num_classes, total_num_examples, devices, is_train=True):
    """Build inference, 4dev, model data hybrid"""
    devices = devices[:]
    # use the last device in list as variable device
    builder = ModelBuilder(devices.pop())
    # First data parallelism, split batches
    with tf.device(builder.variable_device()):
        image_slices = tf.split(0, len(devices), images)
        label_slices = tf.split(0, len(devices), labels)

    # Create replicas for conv layers
    conv_outputs = []
    with tf.variable_scope('model') as vsp:
        # reset name scope, we only want variables in 'model'
        with tf.name_scope(''):
            for idx in range(len(devices)):
                dev = devices[idx]
                with tf.name_scope('tower_{}'.format(idx)), builder.device(dev):
                    net = vgg_part_conv(builder, image_slices[idx])
                conv_outputs.append(net)
                vsp.reuse_variables()

    # Create model paralleled fc layers, merge all batches from replicas into a big one
    with tf.variable_scope('model'):
        # reset name scope, we only want variables in 'model'
        with tf.name_scope(''):
            with builder.parallel(devices, colocate_variables=True):
                net = builder.fc('fc4096a', conv_outputs, 4096, concat_axis=0,
                                 activation=tf.nn.relu)

            with builder.parallel(devices, colocate_variables=True):
                net = builder.fc('fc4096b', net, 4096,
                                 activation=tf.nn.relu)

            with builder.device(devices[0], colocate_variables=True):
                logits = builder.fc('fc1000', net, num_classes, activation=tf.nn.relu)

                with tf.name_scope('probs'):
                    prob = tf.nn.softmax(logits)

                if not is_train:
                    return vgg_eval(prob, labels)

                # Split output again into number of replica and compute loss seperately,
                # so that we can get grad seperately
                logit_slices = tf.split(0, len(devices), logits)

            # Construct loss
            total_losses = []
            for idx in range(len(devices)):
                with builder.device(devices[idx]):
                    l = vgg_loss(logit_slices[idx], label_slices[idx])
                    total_losses.append(l)

    # Construct training
    global_step = builder.ensure_global_step()
    opt = tf.train.AdamOptimizer(learning_rate=0.01)

    with tf.name_scope('training'):
        # first compute grads at the inputs of fc layers
        trainable_fc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model/fc')
        fc_update_ops = [opt.minimize(loss, var_list=trainable_fc, colocate_gradients_with_ops=True)
                         for loss in total_losses]
        fc_update_op = tf.group(*fc_update_ops)

        # compute grads to use for conv layers
        trainable_conv = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model/conv')
        replica_grads = []
        for idx in range(len(devices)):
            dev = devices[idx]
            conv_output = conv_outputs[idx]
            # opt.compute_gradients doesn't accept non Variable as xs,
            # so we use tf.gradients directly
            [conv_init_grad] = tf.gradients(total_losses[idx], [conv_output],
                                            colocate_gradients_with_ops=True)
            with builder.device(dev):
                grads = opt.compute_gradients(conv_output, trainable_conv,
                                              grad_loss=conv_init_grad)
                replica_grads.append(grads)
        with tf.device(builder.variable_device()):
            grads = builder.average_gradients(replica_grads)
            conv_update_op = opt.apply_gradients(grads, global_step=global_step)

        train_op = tf.group(conv_update_op, fc_update_op)

    return prob, logits, tf.reduce_mean(tf.pack(total_losses), 0), train_op, global_step

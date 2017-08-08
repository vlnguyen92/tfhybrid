from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .common import ModelBuilder
from .alexnetcommon import alexnet_inference, alexnet_part_conv, alexnet_loss, alexnet_eval
from ..optimizers.momentumhybrid import HybridMomentumOptimizer


def original(images, labels, num_classes, total_num_examples, devices=None, is_train=True):
    """Build inference"""
    if devices is None:
        devices = [None]

    def configure_optimizer(global_step, total_num_steps):
        """Return a configured optimizer"""
        def exp_decay(start, tgtFactor, num_stairs):
            decay_step = total_num_steps / (num_stairs - 1)
            decay_rate = (1 / tgtFactor) ** (1 / (num_stairs - 1))
            return tf.train.exponential_decay(start, global_step, decay_step, decay_rate,
                                              staircase=True)

        def lparam(learning_rate, momentum):
            return {
                'learning_rate': learning_rate,
                'momentum': momentum
            }

        return HybridMomentumOptimizer({
            'weights': lparam(exp_decay(0.001, 250, 4), 0.9),
            'biases': lparam(exp_decay(0.002, 10, 2), 0.9),
        })

    def train(total_loss, global_step, total_num_steps):
        """Build train operations"""
        # Compute gradients
        with tf.control_dependencies([total_loss]):
            opt = configure_optimizer(global_step, total_num_steps)
            grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        with tf.control_dependencies([apply_gradient_op]):
            return tf.no_op(name='train')

    with tf.device(devices[0]):
        builder = ModelBuilder()
        net, logits, total_loss = alexnet_inference(builder, images, labels, num_classes)

        if not is_train:
            return alexnet_eval(net, labels)

        global_step = builder.ensure_global_step()
        train_op = train(total_loss, global_step, total_num_examples)
    return net, logits, total_loss, train_op, global_step


def ndev_data(images, labels, num_classes, total_num_examples, devices, is_train=True):
    """Build inference, data parallelism"""
    # use the last device in list as variable device
    devices = devices[:]
    builder = ModelBuilder(devices.pop())

    if not is_train:
        with tf.variable_scope('model'):
            prob = alexnet_inference(builder, images, labels, num_classes)[0]
        return alexnet_eval(prob, labels)

    # configure optimizer
    def configure_optimizer(global_step, total_num_steps):
        """Return a configured optimizer"""
        def exp_decay(start, tgtFactor, num_stairs):
            decay_step = total_num_steps / (num_stairs - 1)
            decay_rate = (1 / tgtFactor) ** (1 / (num_stairs - 1))
            return tf.train.exponential_decay(start, global_step, decay_step, decay_rate,
                                              staircase=True)

        def lparam(learning_rate, momentum):
            return {
                'learning_rate': learning_rate,
                'momentum': momentum
            }

        return HybridMomentumOptimizer({
            'weights': lparam(exp_decay(0.001, 250, 4), 0.9),
            'biases': lparam(exp_decay(0.002, 10, 2), 0.9),
        })

    with tf.device(builder.variable_device()):
        global_step = builder.ensure_global_step()
    opt = configure_optimizer(global_step, total_num_examples)

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
                        prob, logits, total_loss = alexnet_inference(builder, image_slices[idx],
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
        net = builder.conv('conv1', images, 64, [11, 11], stride=4, padding='VALID',
                           weight_stddev=0.01, weight_decay=0.0005)
        net = builder.max_pool('pool1', net, 3, stride=2, activation=tf.nn.relu)
        net = builder.conv('conv2', net, 192, [5, 5], stride=1, padding='SAME',
                           weight_stddev=0.01, bias_mean=1, weight_decay=0.0005,
                           activation=tf.nn.relu)
        net = builder.max_pool('pool2', net, 3, stride=2)

    with builder.parallel(devices, colocate_variables=True):
        net = builder.conv('conv3', net, 384, [3, 3], stride=1, padding='SAME',
                           weight_stddev=0.03, weight_decay=0.0005)
        net = builder.conv('conv4', net, 256, [3, 3], stride=1, padding='SAME',
                           weight_stddev=0.03, weight_decay=0.0005,
                           activation=tf.nn.relu)
        net = builder.conv('conv5', net, 256, [3, 3], stride=1, padding='SAME',
                           weight_stddev=0.03, bias_mean=1, weight_decay=0.0005)
        net = builder.max_pool('pool3', net, 3, stride=2, activation=tf.nn.relu)

    with builder.parallel(devices, colocate_variables=True):
        net = builder.fc('fc4096a', net, 4096,
                         weight_stddev=0.01, bias_mean=1.0, weight_decay=0.0005,
                         activation=tf.nn.relu)
        net = builder.dropout('dropout1', net, 0.75)

    with builder.parallel(devices, colocate_variables=True):
        net = builder.fc('fc4096b', net, 4096,
                         weight_stddev=0.01, bias_mean=1.0, weight_decay=0.0005,
                         activation=tf.nn.relu)
        net = builder.dropout('dropout2', net, 0.5)

    with builder.device(devices[0], colocate_variables=True):
        net = builder.fc('fc1000', net, num_classes,
                         weight_stddev=0.01, weight_decay=0.0005)

    # save the unscaled logits for training
    logits = net

    with tf.device(devices[0]):
        with tf.name_scope('probs'):
            prob = tf.nn.softmax(net)

    if not is_train:
        return alexnet_eval(prob, labels)

    def configure_optimizer(global_step, total_num_steps):
        """Return a configured optimizer"""
        def exp_decay(start, tgtFactor, num_stairs):
            decay_step = total_num_steps / (num_stairs - 1)
            decay_rate = (1 / tgtFactor) ** (1 / (num_stairs - 1))
            return tf.train.exponential_decay(start, global_step, decay_step, decay_rate,
                                              staircase=True)

        def lparam(learning_rate, momentum):
            return {
                'learning_rate': learning_rate,
                'momentum': momentum
            }

        return HybridMomentumOptimizer({
            'weights': lparam(exp_decay(0.002, 250, 4), 0.9),
            'biases': lparam(exp_decay(0.004, 25, 2), 0.9),
        })

    def train(total_loss, global_step, total_num_steps):
        """Build train operations"""
        # Compute gradients
        with tf.control_dependencies([total_loss]):
            opt = configure_optimizer(global_step, total_num_steps)
            grads = opt.compute_gradients(total_loss, colocate_gradients_with_ops=True)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        return tf.group(apply_gradient_op, name='train')

    with tf.device(devices[0]):
        total_loss = alexnet_loss(logits, labels)

    global_step = builder.ensure_global_step()
    train_op = train(total_loss, global_step, total_num_examples)
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
                    net = alexnet_part_conv(builder, image_slices[idx])
                conv_outputs.append(net)
                vsp.reuse_variables()

    # Create model paralleled fc layers, merge all batches from replicas into a big one
    with tf.variable_scope('model'):
        # reset name scope, we only want variables in 'model'
        with tf.name_scope(''):
            with builder.parallel(devices, colocate_variables=True):
                net = builder.fc('fc4096', conv_outputs, 4096, concat_axis=0,
                                 weight_stddev=0.01, bias_mean=1.0, weight_decay=0.0005,
                                 activation=tf.nn.relu)
                net = builder.dropout('dropout1', net, 0.5)

            with builder.parallel(devices, colocate_variables=True):
                net = builder.fc('fc4096b', net, 4096,
                                 weight_stddev=0.01, bias_mean=1.0, weight_decay=0.0005,
                                 activation=tf.nn.relu)
                net = builder.dropout('dropout2', net, 0.5)

            with builder.device(devices[0], colocate_variables=True):
                logits = builder.fc('fc1000', net, num_classes,
                                    weight_stddev=0.01, weight_decay=0.0005)
                with tf.name_scope('probs'):
                    prob = tf.nn.softmax(logits)

                if not is_train:
                    return alexnet_eval(prob, labels)

                # Split output again into number of replica and compute loss seperately,
                # so that we can get grad seperately
                logit_slices = tf.split(0, len(devices), logits)

            # Construct loss
            total_losses = []
            for idx in range(len(devices)):
                with builder.device(devices[idx]):
                    l = alexnet_loss(logit_slices[idx], label_slices[idx])
                    total_losses.append(l)

    def configure_optimizer(global_step, total_num_steps):
        """Return a configured optimizer"""
        def exp_decay(start, tgtFactor, num_stairs):
            decay_step = total_num_steps / (num_stairs - 1)
            decay_rate = (1 / tgtFactor) ** (1 / (num_stairs - 1))
            return tf.train.exponential_decay(start, global_step, decay_step, decay_rate,
                                              staircase=True)

        def lparam(learning_rate, momentum):
            return {
                'learning_rate': learning_rate,
                'momentum': momentum
            }

        return HybridMomentumOptimizer({
            'conv./weights': lparam(exp_decay(0.002, 250, 4), 0.9),
            'conv./biases': lparam(exp_decay(0.001, 10, 2), 0.9),
            'fc.+/weights': lparam(exp_decay(0.001, 250, 4), 0.9),
            'fc.+/biases': lparam(exp_decay(0.002, 10, 2), 0.9),
        })

    # Construct training
    with tf.device(builder.variable_device()):
        global_step = builder.ensure_global_step()
    opt = configure_optimizer(global_step, total_num_examples)

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

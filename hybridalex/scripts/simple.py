from __future__ import absolute_import, division, print_function

import sys
import argparse
import numpy as np
import tensorflow as tf
from timeit import default_timer

from ..nets.common import ModelBuilder
from ..optimizers import HybridMomentumOptimizer, lparam
from ..utils import tfhelper


def fc_model(inputs, labels, devices):
    devices = devices[:]
    builder = ModelBuilder(devices.pop())
    builder.ensure_global_step()

    opt = HybridMomentumOptimizer({
            'weights': lparam(0.001, 0.9),
            'biases': lparam(0.001, 0.9),
        })

    net = inputs
    with builder.parallel(devices, colocate_variables=True):
        # for some reason, activation=tf.nn.relu is needed, otherwise TF wouldn't be
        # happy with colocate_gradients_with_ops
        net = builder.fc('fc', net, 1024, activation=tf.nn.relu)
    # net is a PartitionedTensor after the parallel, merge them back to a single tensor using
    # a dropout layer with keep_prob = 1
    with builder.device(devices[0]):
        net = builder.dropout('dropout', net, keep_prob=1)
        loss = tf.contrib.losses.mean_squared_error(net, labels)

    return opt.minimize(loss, global_step=builder.ensure_global_step(),
                        colocate_gradients_with_ops=True)


def fc_data(inputs, labels, devices):
    devices = devices[:]
    builder = ModelBuilder(devices.pop())
    builder.ensure_global_step()

    opt = HybridMomentumOptimizer({
            'weights': lparam(0.001, 0.9),
            'biases': lparam(0.001, 0.9),
        })

    replica_grads = []
    with tf.device(builder.variable_device()):
        input_slices = tf.split(0, len(devices), inputs)
        label_slices = tf.split(0, len(devices), labels)
    with tf.variable_scope('model') as vsp:
        with tf.name_scope(''):
            for idx in range(len(devices)):
                with tf.name_scope('tower_{}'.format(idx)):
                    with builder.device(devices[idx]):
                        net = builder.fc('fc', input_slices[idx], 1024)
                        loss = tf.contrib.losses.mean_squared_error(net, label_slices[idx])
                        grad = opt.compute_gradients(loss)
                        replica_grads.append(grad)
                vsp.reuse_variables()

    # average gradients across replica
    with tf.device(builder.variable_device()):
        grads = builder.average_gradients(replica_grads)
        apply_grads_op = opt.apply_gradients(grads, global_step=builder.ensure_global_step())

        train_op = tf.group(apply_grads_op, name='train')
    return train_op


def fc_none(inputs, labels, devices):
    devices = devices[:]
    builder = ModelBuilder(devices.pop())
    builder.ensure_global_step()

    opt = HybridMomentumOptimizer({
            'weights': lparam(0.001, 0.9),
            'biases': lparam(0.001, 0.9),
        })

    with builder.device(devices[0], colocate_variables=True):
        net = builder.fc('fc', inputs, 1024)
        loss = tf.contrib.losses.mean_squared_error(net, labels)
        train_op = opt.minimize(loss, name='train')

    return train_op


def conv_data(inputs, labels, devices):
    devices = devices[:]
    builder = ModelBuilder(devices.pop())
    builder.ensure_global_step()

    opt = HybridMomentumOptimizer({
            'weights': lparam(0.001, 0.9),
            'biases': lparam(0.001, 0.9),
        })

    replica_grads = []
    with tf.device(builder.variable_device()):
        input_slices = tf.split(0, len(devices), inputs)
        label_slices = tf.split(0, len(devices), labels)
    with tf.variable_scope('model') as vsp:
        with tf.name_scope(''):
            for idx in range(len(devices)):
                with tf.name_scope('tower_{}'.format(idx)):
                    with builder.device(devices[idx]):
                        net = builder.conv('conv', input_slices[idx], 96, [11, 11], stride=2)
                        loss = tf.contrib.losses.mean_squared_error(net, label_slices[idx])
                        grad = opt.compute_gradients(loss)
                        replica_grads.append(grad)
                vsp.reuse_variables()

    # average gradients across replica
    with tf.device(builder.variable_device()):
        grads = builder.average_gradients(replica_grads)
        apply_grads_op = opt.apply_gradients(grads, global_step=builder.ensure_global_step())

        train_op = tf.group(apply_grads_op, name='train')
    return train_op


def conv_model(inputs, labels, devices):
    devices = devices[:]
    builder = ModelBuilder(devices.pop())
    builder.ensure_global_step()

    opt = HybridMomentumOptimizer({
            'weights': lparam(0.001, 0.9),
            'biases': lparam(0.001, 0.9),
        })

    net = inputs
    with builder.parallel(devices, colocate_variables=True):
        # for some reason, activation=tf.nn.relu is needed, otherwise TF wouldn't be
        # happy with colocate_gradients_with_ops
        net = builder.conv('conv', net, 96, [11, 11], stride=2, activation=tf.nn.relu)
    # net is a PartitionedTensor after the parallel, merge them back to a single tensor using
    # a dropout layer with keep_prob = 1
    with builder.device(devices[0]):
        net = builder.dropout('identity', net, keep_prob=1, concat_axis=3)
        loss = tf.contrib.losses.mean_squared_error(net, labels)
    return opt.minimize(loss, global_step=builder.ensure_global_step(),
                        colocate_gradients_with_ops=True)


def conv_none(inputs, labels, devices):
    devices = devices[:]
    builder = ModelBuilder(devices.pop())
    builder.ensure_global_step()

    opt = HybridMomentumOptimizer({
            'weights': lparam(0.001, 0.9),
            'biases': lparam(0.001, 0.9),
        })

    with builder.device(devices[0], colocate_variables=True):
        net = builder.conv('conv', inputs, 96, [11, 11], stride=2)
        loss = tf.contrib.losses.mean_squared_error(net, labels)
        return opt.minimize(loss, name='train')


def train(name):
    target = 'grpc://localhost:2222'
    devices = ['/job:worker/task:0/gpu:0', '/job:worker/task:1/gpu:0', '/job:ps/task:0/cpu:0']

    log_dir = '/tmp/workspace/tflogs'

    total_steps = 50
    batch_size = 10

    network, expected_output_shape = {
        'fc_data': (fc_data, (batch_size, 1024)),
        'fc_model': (fc_model, (batch_size, 1024)),
        'fc_none': (fc_none, (batch_size, 1024)),
        'conv_data': (conv_data, (batch_size, 128, 128, 96)),
        'conv_model': (conv_model, (batch_size, 128, 128, 96)),
        'conv_none': (conv_none, (batch_size, 128, 128, 96))
    }[name]

    raw_image = np.random.randn(batch_size, 256, 256, 3)
    expected_output = np.random.randn(*expected_output_shape)

    with tf.Graph().as_default() as g:
        inputs = tf.placeholder(tf.float32, shape=raw_image.shape, name='input')
        outputs = tf.placeholder(tf.float32, shape=expected_output.shape, name='output')

        train_op = network(inputs, outputs, devices)

        summary_writer = tf.train.SummaryWriter(log_dir, g)

        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        with tf.Session(target, config=config) as sess:
            sess.run(tfhelper.initialize_op())

            speeds = []
            for step in range(total_steps):
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                start_time = default_timer()
                # run the op
                sess.run(train_op, feed_dict={inputs: raw_image, outputs: expected_output},
                         options=run_options, run_metadata=run_metadata)

                duration = default_timer() - start_time
                summary_writer.add_run_metadata(run_metadata, 'step{}'.format(step), step)

                speed = batch_size / duration
                speeds.append(speed)
                print('Step {} - {:.1f} images/sec; {:.4f} sec/batch'.format(step, speed, duration))
                sys.stdout.flush()
            print('Average {:.1f} images/sec'.format(np.mean(speeds)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()

    train(args.name)

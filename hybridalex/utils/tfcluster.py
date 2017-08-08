from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    reload
except NameError:
    # Python 3
    from importlib import reload  # noqa F401

from .tmuxwrapper import tmux

clusterEntry = 'grpc://localhost:2222'


def start_local_cluster(clusterSpec, venvname='tf'):
    initialStr = ';'.join(['import tensorflow as tf',
                           'gpu_opt = tf.GPUOptions(per_process_gpu_memory_fraction={pergpu})',
                           'cfg = tf.ConfigProto(gpu_options=gpu_opt)'
                           ])
    clusterStr = 'cluster = tf.train.ClusterSpec({})'.format(
        clusterSpec.__str__()).replace('\'', '"')
    serverStr = ('server = tf.train.Server(cluster, config=cfg, job_name="{job_name}",'
                 'task_index={task_index})')

    with tmux('tfCluster') as ts:
        for job_name, nodes in clusterSpec.items():

            for task_index in range(len(nodes)):
                window_name = '{job_name}{task_index}'.format(
                    job_name=job_name, task_index=task_index)
                ts.run('venv {}'.format(venvname), new_window=window_name, noenv=True)

                pergpu = 0.01 if job_name == 'ps' else 0.95

                iStr = initialStr.format(pergpu=pergpu)
                sStr = serverStr.format(job_name=job_name, task_index=task_index)
                ts.run("python -c '{}'".format(';'.join([iStr,
                                                         clusterStr,
                                                         sStr,
                                                         'server.join()'])),
                       new_window=window_name)


def kill_local_cluster(sessionName=None):
    if sessionName is None:
        sessionName = 'tfCluster'

    with tmux('tfCluster', destroy=True) as ts:
        ts.run('echo nothing')

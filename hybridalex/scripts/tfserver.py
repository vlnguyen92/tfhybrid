# /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf


if __name__ == '__main__':
    cldefpath = sys.argv[1]
    job_name = sys.argv[2]
    task_index = int(sys.argv[3])
    with open(cldefpath, 'rb') as f:
        cldef = tf.train.ClusterDef.FromString(f.read())

    cluster = tf.train.ClusterSpec(cldef)
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)
    server.join()

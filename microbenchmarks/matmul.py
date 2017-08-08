import tensorflow as tf
import datetime
import numpy as np

logs_path = "./logs"
# Create random large matrix
A = np.random.rand(100, 100).astype('float32')
B = np.random.rand(100, 100).astype('float32')

with tf.device('/gpu:0'):
    a = tf.placeholder(tf.float32, [100, 100])
    b = tf.placeholder(tf.float32, [100, 100])
    mul = tf.matmul(a, b)
#with tf.device('gpu:0'):

t1_1 = datetime.datetime.now()
metadata = tf.RunMetadata()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
ops = tf.GraphOptions(build_cost_model=5)

summary_op = tf.merge_all_summaries()

with tf.Session(config=tf.ConfigProto(log_device_placement=True,
    graph_options=ops)) as sess:
    sess.run(tf.initialize_all_variables())

    writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
    # Run the op.
    for i in xrange(20):
        _ = sess.run([mul], feed_dict={a:A, b:B}, options=run_options, run_metadata=metadata)
#    writer.add_summary(summary)
        print ("Cost graph " + str(metadata.cost_graph))
t2_1 = datetime.datetime.now()
print (t2_1 - t1_1)

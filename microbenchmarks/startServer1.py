import tensorflow as tf
server = tf.train.Server({"worker": ["141.212.106.204:2222",
                                     "141.212.106.20:2222"]},
                                      job_name="worker",
                                      task_index=0)
server.join()

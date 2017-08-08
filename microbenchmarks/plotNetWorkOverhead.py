#!/usr/bin/python

import matplotlib.pyplot as plt

def timeSeries(filename):
    with open(filename,'r') as file:
        data = []
        for line in file:
            data.append(float(line))
    return data

one_local = timeSeries('logs/1-machine-local.log')
one_remote = timeSeries('logs/1-machine-remote.log')
two_network = timeSeries('logs/2-machine-network.log')

X = [1000 * i for i in range(1,11)]

plt.plot(X,one_local,label='1-local')
plt.plot(X,one_remote,label='1-remote')
plt.plot(X,two_network,label='2-network')
plt.legend()
plt.show()

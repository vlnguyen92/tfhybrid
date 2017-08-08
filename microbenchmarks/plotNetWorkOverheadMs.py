#!/usr/bin/python

import matplotlib.pyplot as plt

def timeSeries(filename):
    with open(filename,'r') as file:
        data = []
        for line in file:
            data.append(float(line))
    return data

one_local = timeSeries('logs/1-machine-local.log')
two_network = timeSeries('logs/2-machine-network.log')

X = [1000 * i for i in range(1,11)]
Y = [(j-i)*1000.0 for i,j in zip(one_local,two_network)]

plt.plot(X,Y)
plt.ylabel("Network Overhead (ms)")
plt.legend()
plt.show()

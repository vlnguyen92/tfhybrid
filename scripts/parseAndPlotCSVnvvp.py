#!/usr/bin/env python

import sys
import os # OS stuff
import glob # pathname stuff
import csv # CSV
import re # Regex
from difflib import SequenceMatcher
from pprint import pprint # Pretty Print

def countRow(file):
    with open(file, 'rb') as f:
        n_row = 0
        data = csv.DictReader(f)
        for row in data:
            n_row+=1
        return n_row

import numpy as np

trace_file = "../logs/gpu_trace_alex_128.csv"
summary_file = "../logs/gpu_summary_alex_128.csv"
occupancy_file = "../logs/gpu_occupancy_alex_128.csv"

kernels = []
names = []
occupancies = []
with open(occupancy_file, 'rb') as f:
    data = csv.DictReader(f)
    for row in data:
#        print row["Kernel"]
        kernels.append(row["Kernel"])
        occupancies.append(row["Avg"])

#already sorted
percentages = []
with open(summary_file, 'rb') as f:
    data = csv.DictReader(f)
    for row in data:
        names.append(row["Name"])
        percentages.append(row["Time(%)"])

avg = 0.0
for name in names[1:]:
#    print name
    for kernel in kernels:
        if (SequenceMatcher(a=name, b=kernel).ratio() > .99):
            weighted = \
            float(percentages[names.index(name)]) * \
            float(occupancies[kernels.index(kernel)])
            avg += weighted
print avg

#$print len(kernels)
#$dfjasjdkfjaklsjdfkj;krint len(names)

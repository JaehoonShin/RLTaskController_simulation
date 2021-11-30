import dill as pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import getopt
import sys
import csv
import os

filename = 'history_results'
filename = filename + '/' + 'Analysis-Object-min-rpe-01_tst2_delta_control.pkl'

with open(filename, 'rb') as f:
    data = pickle.load(f)

plt.plot(data.data['min-rpe'][0]['validation'])
plt.savefig('history_results/reader_data_validation_plot.png')
plt.clf()

plt.plot(data.data['min-rpe'][0]['tag_prob'])
plt.savefig('history_results/reader_data_tag_prob_plot.png')
plt.clf()
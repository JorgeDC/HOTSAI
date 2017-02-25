import pandas as pd
import numpy as np
import sys

x_path = 'hots_final_hot_encoding.csv'
x = np.array(pd.read_csv(x_path))

y_path = 'hots_final_results.csv'
y = np.array(pd.read_csv(y_path))

all = np.hstack((x,y))

np.random.shuffle(all)

np.savetxt('all_results.csv', all, fmt='%i', delimiter=",")


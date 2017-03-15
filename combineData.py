import pandas as pd
import numpy as np
import sys
import math

sys.stdout.write("\rData path")
data_path = 'ReplayCharacters.csv'
#data_path = 'hotslogs_dataset.csv'
sys.stdout.write("\rLoading")
hots_pre = np.array(pd.read_csv(data_path))

data_path_replays = 'Replays.csv'
#data_path_replays = 'hotslogs_dataset_replays.csv'
replays_pre = np.array(pd.read_csv(data_path_replays))
sys.stdout.write("\rDone loading csv")

batch_size = 10
hots_rows, _ = hots_pre.shape
hots_final = []

sys.stdout.write("\rStarting preprocessing")
should_append = False

for start_i in range(0, hots_rows, batch_size):
    end_i = start_i + batch_size
    batch = hots_pre[start_i:end_i]
    for start_i_i in range(0, 10):
        #print(batch[start_i_i,:])
        #print(replays_pre[start_i,:])
        hots_final_row = np.concatenate((batch[start_i_i,:], replays_pre[math.floor(start_i/10),:]), axis=0)
        hots_final.append(hots_final_row)
        sys.stdout.write("\rOn row number: %i" % start_i)


sys.stdout.write("\rSaving file")
np.savetxt('all_data.csv', hots_final, fmt='%s', delimiter=",")
sys.stdout.write("\rdone")
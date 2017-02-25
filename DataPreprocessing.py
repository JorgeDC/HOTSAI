import numpy as np
import pandas as pd
import sys

sys.stdout.write("\rData path")
data_path = 'ReplayCharacters.csv'
#data_path = 'hotslogs_dataset.csv'
sys.stdout.write("\rLoading")
hots_pre = pd.read_csv(data_path)

data_path_replays = 'Replays.csv'
#data_path_replays = 'hotslogs_dataset_replays.csv'
replays_pre = pd.read_csv(data_path_replays)
sys.stdout.write("\rDone loading csv")

hots = np.array(hots_pre[[0, 2, 4]])
replays = np.array(replays_pre[[0, 2]])
sys.stdout.write("\rRemoving data done")

# 92999525

# 1 427 233

batch_size = 10
hots_rows, _ = hots.shape
hots_final = []
hots_final_results = []
number_heroes = 63
number_maps = 14

sys.stdout.write("\rStarting preprocessing")
should_append = False

for start_i in range(0, hots_rows, batch_size):
    end_i = start_i + batch_size
    batch = hots[start_i:end_i]

    maprow = np.zeros(number_maps)
    i = 0
    for replay_row in replays:
        if replay_row[0] == batch[0][0]:
            map_id = replay_row[1] - 1001
            maprow[map_id] = 1
            replays = np.delete(replays, i, axis=0)
        i += 1
        break

    sorted_batch = batch[batch[:, 2].argsort()]
    hots_final_row = np.array(np.zeros(number_heroes * 2))
    heroes_a_are_winners = np.remainder(start_i, 20) == 0
    heroes_a = []
    heroes_b = []
    if heroes_a_are_winners:
        heroes_b = sorted_batch[0:5]  # losers
        heroes_a = sorted_batch[5:10]  # winners
    else:
        heroes_a = sorted_batch[0:5]  # losers
        heroes_b = sorted_batch[5:10]  # winners

    for start_i_i in range(0, 5):
        hots_final_row[heroes_a[start_i_i, 1] - 1] = 1
        hots_final_row[heroes_b[start_i_i, 1] + number_heroes - 1] = 1

    hots_final_row_with_map = np.concatenate((hots_final_row, maprow), axis=0)
    hots_final.append(hots_final_row_with_map)

    hots_final_row_result = np.array(np.zeros(2))
    if heroes_a_are_winners:
        hots_final_row_result[0] = 1
    else:
        hots_final_row_result[1] = 1
    hots_final_results.append(hots_final_row_result)

    sys.stdout.write("\rOn row number: %i" % start_i)

sys.stdout.write("\rSaving file")
np.savetxt('hots_final_hot_encoding.csv', hots_final, fmt='%i', delimiter=",")
sys.stdout.write("\rSaving results")
np.savetxt('hots_final_results.csv', hots_final_results, fmt='%i', delimiter=",")
sys.stdout.write("\rDone")
import numpy as np
import pandas as pd
import sys

sys.stdout.write("\rData path")
data_path = 'training_data/all_data.csv'
sys.stdout.write("\rLoading")
hots_pre = pd.read_csv(data_path, header=None)

print(hots_pre.shape)
#game_mode is at column nr:21
#map is at column nr:22
hots = np.array(hots_pre[[0, 2, 4, 21, 22]])

batch_size = 10
hots_rows, _ = hots.shape
hots_final = []
hots_final_results = []
number_heroes = 64
number_maps = 14
number_game_mode = 4

number_of_columns_first_half = 20
number_of_columns_second_half = 5

sys.stdout.write("\rStarting preprocessing")
should_append = False

for start_i in range(0, hots_rows, batch_size):
    end_i = start_i + batch_size
    batch = hots[start_i:end_i]

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


    maprow = np.zeros(number_maps)
    map_id = sorted_batch[0,4] - 1001
    maprow[map_id] = 1

    gamemode_row = np.zeros(number_game_mode)
    game_mode_id = sorted_batch[0,3] - 3
    gamemode_row[game_mode_id] = 1

    hots_final_row_result = np.array(np.zeros(2))
    if heroes_a_are_winners:
        hots_final_row_result[0] = 1
    else:
        hots_final_row_result[1] = 1



    # hots_final_row_complete = np.concatenate((hots_final_row, maprow, gamemode_row, hots_final_row_result), axis=0)
    hots_final_row_complete = np.concatenate((hots_final_row, maprow, hots_final_row_result), axis=0)

    #only HL and TL
    if game_mode_id == 1 or game_mode_id == 2:
        hots_final.append(hots_final_row_complete)

    sys.stdout.write("\rOn row number: %i" % start_i)

sys.stdout.write("\rSaving file")
np.savetxt('training_data/hots_final_hot_encoding.csv', hots_final, fmt='%i', delimiter=",")
sys.stdout.write("\rDone")
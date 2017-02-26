import numpy as np
import pandas as pd
from datetime import datetime
import sys


sys.stdout.write("\rData path")
data_path = 'all_data.csv'
#data_path = 'small_example_alldata.csv'
sys.stdout.write("\rLoading")
hots_pre = pd.read_csv(data_path, header=None)

# avarage mmr...
hots_pre = hots_pre.fillna(2000)

print(hots_pre.shape)
# game_mode is at column nr:21
# map is at column nr:22
# date on column nr: 24
# mmr on column nr: 5
hots = np.array(hots_pre[[0, 2, 4, 21, 22, 24, 5]])

batch_size = 10
hots_rows, _ = hots.shape
hots_final = []
hots_final_results = []
number_heroes = 63
number_maps = 14
number_of_game_types = 0
number_of_players = 10

number_of_columns_first_half = 20
number_of_columns_second_half = 5
hotslogs_last_month_from_export_date = '1/14/2017 01:00:01 AM'
date_format = '%m/%d/%Y %I:%M:%S %p'

sys.stdout.write("\rStarting preprocessing")
should_append = False

export_date_minus_one_month = datetime.strptime(hotslogs_last_month_from_export_date, date_format)

all_mmr = hots[:, 6]
print(all_mmr)
mean = np.nanmean(all_mmr)
print(mean)
std = np.std(all_mmr)
print(std)
normalized_mmr = (all_mmr - mean) / std
print(normalized_mmr)
hots[:, 6] = normalized_mmr

for start_i in range(0, hots_rows, batch_size):
    end_i = start_i + batch_size
    batch = hots[start_i:end_i]

    match_date = datetime.strptime(batch[0, 5], date_format)
    # print(match_date)
    # if match_date > export_date_minus_one_month:


    sorted_batch = batch[batch[:, 2].argsort()]

    if sorted_batch[0, 3] == 4: #only hero league
        hots_final_row = np.array(np.zeros(number_heroes * 10))
        hots_mmr_row = np.array(np.zeros(10))
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
            hots_final_row[heroes_a[start_i_i, 1] + (start_i_i * number_heroes) - 1] = 1
            hots_final_row[heroes_b[start_i_i, 1] + (number_heroes * 5) + (start_i_i * number_heroes) - 1] = 1
            hots_mmr_row[start_i_i] = heroes_a[start_i_i, 6]
            hots_mmr_row[start_i_i + 5] = heroes_b[start_i_i, 6]

        maprow = np.zeros(number_maps)
        map_id = sorted_batch[0, 4] - 1001
        maprow[map_id] = 1

        #gamemode_row = np.zeros(number_of_game_types)
        #game_mode_id = sorted_batch[0, 3] - 3
        #gamemode_row[game_mode_id] = 1

        hots_final_row_result = np.array(np.zeros(2))
        if heroes_a_are_winners:
            hots_final_row_result[0] = 1
        else:
            hots_final_row_result[1] = 1

        hots_final_row_complete = np.concatenate(
            (hots_final_row, maprow, hots_mmr_row, hots_final_row_result), axis=0)
        hots_final.append(hots_final_row_complete)

        sys.stdout.write("\rOn row number: %i" % start_i)

sys.stdout.write("\rSaving file")

number_of_ints = (number_heroes * number_of_players) + number_of_game_types + number_maps

fmtString = ""
for index in range(number_of_ints):
    fmtString += "%i,"

fmtString += "%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%i,%i"

print(fmtString)

np.savetxt('hots_final_hot_encoding.csv', hots_final, fmt=fmtString, delimiter=",")

sys.stdout.write("\rDone")

import numpy as np
import pandas as pd
import sys

sys.stdout.write("\rData path")
# data_path = 'training_data/all_data_to_test.csv'
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
#for start_i in range(0, 20, batch_size):
    end_i = start_i + batch_size
    batch = hots[start_i:end_i]

    sorted_batch = batch[batch[:, 2].argsort()]
    first_team_are_winners = np.remainder(start_i, 20) == 0

    if first_team_are_winners:
        first_team_are_winners = 0
    else:
        first_team_are_winners = 1

    first_team = sorted_batch[0:5]
    second_team = sorted_batch[5:10]

    hots_heroes_first_team = []
    hots_heroes_second_team = []

    # for start_i_i in range(0, 5):
    #     hots_heroe_first_team = np.array(np.zeros(number_heroes))
    #     hots_heroe_first_team[first_team[start_i_i, 1] - 1] = 1
    #     hots_heroes_first_team = np.concatenate((hots_heroes_first_team, hots_heroe_first_team), axis=0)
    #
    #     hots_heroe_second_team = np.array(np.zeros(number_heroes))
    #     hots_heroe_second_team[second_team[start_i_i, 1] - 1] = 1
    #     hots_heroes_second_team = np.concatenate((hots_heroes_second_team, hots_heroe_second_team), axis=0)
    #
    # hots_final_row =  np.concatenate(hots_heroes_first_team, hots_heroes_second_team)


    hots_final_row = np.zeros(10)

    for start_i_i in range(0, 10):

        heroe_id = 0
        if start_i_i == 0:
            hots_final_row[0] = first_team[0, 1] - 1

        if start_i_i == 1:
            hots_final_row[1] = second_team[0, 1] - 1

        if start_i_i == 2:
            hots_final_row[2] = second_team[1, 1 ]- 1

        if start_i_i == 3:
            hots_final_row[3] = first_team[1, 1] - 1

        if start_i_i == 4:
            hots_final_row[4] = first_team[2, 1] - 1

        if start_i_i == 5:
            hots_final_row[5] = second_team[2, 1] - 1

        if start_i_i == 6:
            hots_final_row[6] = second_team[3, 1] - 1

        if start_i_i == 7:
            hots_final_row[7] = first_team[3, 1] - 1

        if start_i_i == 8:
            hots_final_row[8] = first_team[4, 1] - 1

        if start_i_i == 9:
            hots_final_row[9] = second_team[4, 1] - 1

        #hots_final_row = hots_final_row.append(heroe_id)
        #hots_final_row = np.concatenate((hots_final_row, heroe_id), axis=0)

    #print(hots_final_row)

    maprow = np.zeros(number_heroes) #number of heroes and padded with 0 behind the vector
    map_id = sorted_batch[0,4] - 1001
    maprow[map_id] = 1

    gamemode_row = np.zeros(number_game_mode)
    game_mode_id = sorted_batch[0,3] - 3
    gamemode_row[game_mode_id] = 1

    hots_final_row_result = np.array(np.zeros(2))
    if first_team_are_winners:
        hots_final_row_result[0] = 1
    else:
        hots_final_row_result[1] = 1


    # hots_final_row_complete = np.concatenate((hots_final_row, maprow, gamemode_row, hots_final_row_result), axis=0)
    # hots_final_row_complete = np.concatenate(([map_id], hots_final_row, hots_final_row_result), axis=0)
    hots_final_row_complete = np.concatenate((hots_final_row, hots_final_row_result), axis=0)
    # print(hots_final_row_complete)
    #only HL and TL
    if game_mode_id == 1 or game_mode_id == 2:
        hots_final.append(hots_final_row_complete)

    sys.stdout.write("\rOn row number: %i" % start_i)

sys.stdout.write("\rSaving file")
np.savetxt('training_data/hots_final_hot_encoding_rnn.csv', hots_final, fmt='%i', delimiter=",")
sys.stdout.write("\rDone")
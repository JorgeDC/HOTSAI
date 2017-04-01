import tensorflow as tf
import pandas as pd
import numpy as np
import sys

#if __name__ == "__main__":
#    blue_team = sys.argv[1]
#    red_team = sys.argv[2]
#    print(blue_team)
#    print(red_team)

save_model_path = 'saved_model/tensorflow_hots_model'
loaded_graph = tf.Graph()

number_heroes = 64
number_maps = 14
number_game_mode = 4

heroes_id_to_name = {0:"ABATHUR",
                     1 :"ANUBARAK",
                     2: "ARTHAS",
                     3: "AZMODAN",
                     4: "BRIGHTWING",
                     5: "CHEN",
                     6: "DIABLO",
                     7: "ETC",
                     8: "FALSTAD",
                     9: "GAZLOWE",
                     10: "ILLIDAN",
                     11: "JAINA",
                     12: "JOHANNA",
                     13: "KAELTHAS",
                     14: "KERRIGAN",
                     15: "KHARAZIM",
                     16: "LEORIC",
                     17: "LILI",
                     18: "MALFURION",
                     19: "MURADIN",
                     20: "MURKY",
                     21: "NAZEEBO",
                     22: "NOVA",
                     23: "RAYNOR",
                     24: "REHGAR",
                     25: "SGTHAMMER",
                     26: "SONYA",
                     27: "STITCHES",
                     28: "SYLVANAS",
                     29: "TASSADAR",
                     30: "THEBUTCHER",
                     31: "THELOSTVIKINGS",
                     32: "THRALL",
                     33: "TYCHUS",
                     34: "TYRAEL",
                     35: "TYRANDE",
                     36: "UTHER",
                     37: "VALLA",
                     38: "ZAGARA",
                     39: "ZERATUL",
                     40: "REXXAR",
                     41: "LTMORALES",
                     42: "ARTANIS",
                     43: "CHO",
                     44: "GALL",
                     45: "LUNARA",
                     46: "GREYMANE",
                     47: "LIMING",
                     48: "XUL",
                     49: "DEHAKA",
                     50: "TRACER",
                     51: "CHROMIE",
                     52: "MEDIVH",
                     53: "GULDAN",
                     54: "AURIEL",
                     55: "ALARAK",
                     56: "ZARYA",
                     57: "SAMURO",
                     58: "VARIAN",
                     59: "RAGNAROS",
                     60: "ZULJIN",
                     61: "VALEERA",
                     62: "LUCIO",
                     63: "PROBIUS"}

heroes_name_to_id = {y:x for x,y in heroes_id_to_name.items()}

map_id_to_name = {0: "BATTLEFIELD_OF_ETERNITY",
                  1: "BLACKHEARTS_BAY",
                  2: "CURSED_HOLLOW",
                  3: "DRAGON_SHIRE",
                  4: "GARDEN_OF_TERROR",
                  5: "HAUNTED_MINES",
                  6: "INFERNAL_SHRINES",
                  7: "SKY_TEMPLE",
                  8: "TOMB_OF_THE_SPIDER_QUEEN",
                  9: "TOWERS_OF_DOOM",
                  10: "LOST_CAVERN",
                  11: "BRAXIS_HOLDOUT",
                  12: "WARHEAD_JUNCTION",
                  13: "SILVER_CITY",
                  14: "BRAXIS_OUTPOST"}

map_name_to_id = {y:x for x,y in map_id_to_name.items()}

#3=Quick Match 4=Hero League 5=Team League 6=Unranked Draft
game_mode_id_to_name = {0: "QM", 1: "HL", 2: "TL", 3: "UD"}
game_mode_name_to_id = {y:x for x,y in game_mode_id_to_name.items()}

#Enemy team - your team - map - game mode

enemy_team = ["ILLIDAN"]
your_team = []
map = "CURSED_HOLLOW"
game_mode = "HL"

all_possibilities = []

enemy_team_row = np.zeros(number_heroes)
your_team_row = np.zeros(number_heroes)
map_row = np.zeros(number_maps)
game_mode_row = np.zeros(number_game_mode)

enemy_team_row[heroes_name_to_id["NOVA"]] = 1
map_row[map_name_to_id[map]] = 1
game_mode_row[game_mode_name_to_id[game_mode]] = 1
print(enemy_team_row)
print(map_row)
print(game_mode_row)

for i in range(number_heroes):
    heroe_your_team_row = np.zeros(number_heroes)
    heroe_your_team_row[heroes_name_to_id["ABATHUR"]] = 1
    heroe_your_team_row[i] = 1
    complete_row = np.concatenate((heroe_your_team_row, enemy_team_row, map_row, game_mode_row), axis=0)
    all_possibilities.append(complete_row)
    print(complete_row)

with tf.Session(graph=loaded_graph) as sess:
    # Load model
    loader = tf.train.import_meta_graph(save_model_path + '.meta')
    loader.restore(sess, save_model_path)

    # Get Tensors from loaded model
    loaded_x = loaded_graph.get_tensor_by_name('features:0')
    loaded_y = loaded_graph.get_tensor_by_name('labels:0')
    loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    loaded_logits = loaded_graph.get_tensor_by_name('softmax_logits:0')
    results = sess.run(loaded_logits, feed_dict={loaded_x: all_possibilities, loaded_keep_prob: 1.0})
    results = results[results[:, 0].argsort()]
    rows, cols = results.shape
    for i in range(rows):
        heroe = heroes_id_to_name[i]
        print(heroe, results[i])




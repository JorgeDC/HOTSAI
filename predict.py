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

print(heroes_name_to_id)

with tf.Session(graph=loaded_graph) as sess:
    # Load model
    loader = tf.train.import_meta_graph(save_model_path + '.meta')
    loader.restore(sess, save_model_path)

    # Get Tensors from loaded model
    loaded_x = loaded_graph.get_tensor_by_name('features:0')
    loaded_y = loaded_graph.get_tensor_by_name('labels:0')
    loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    loaded_logits = loaded_graph.get_tensor_by_name('softmax_logits:0')



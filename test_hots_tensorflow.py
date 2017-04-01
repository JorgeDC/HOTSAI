import tensorflow as tf
import pandas as pd
import numpy as np
import random
import math


save_model_path = 'saved_model/tensorflow_hots_model'
loaded_graph = tf.Graph()

test_features_path = 'training_data/test_data/test_x.csv'
test_labels_path = 'training_data/test_data/test_y.csv'
test_features = np.array(pd.read_csv(test_features_path))
test_labels = np.array(pd.read_csv(test_labels_path))
batch_size = 50

with tf.Session(graph=loaded_graph) as sess:
    # Load model
    loader = tf.train.import_meta_graph(save_model_path + '.meta')
    loader.restore(sess, save_model_path)

    # Get Tensors from loaded model
    loaded_x = loaded_graph.get_tensor_by_name('features:0')
    loaded_y = loaded_graph.get_tensor_by_name('labels:0')
    loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    loaded_logits = loaded_graph.get_tensor_by_name('softmax_logits:0')
    loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

    # Get accuracy in batches for memory limitations
    test_batch_acc_total = 0
    test_batch_count = 0

    #for train_feature_batch, train_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
    batch_count = int(math.ceil(len(test_features) / batch_size))
    for batch_i in range(batch_count):
        batch_start = batch_i * batch_size
        batch_features = test_features[batch_start:batch_start + batch_size]
        batch_labels = test_labels[batch_start:batch_start + batch_size]
        test_batch_acc_total += sess.run(
            loaded_acc,
            feed_dict={loaded_x: batch_features, loaded_y: batch_labels, loaded_keep_prob: 1.0})
        test_batch_count += 1

    print('Testing Accuracy: {}\n'.format(test_batch_acc_total / test_batch_count))
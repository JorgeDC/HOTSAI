import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import sys
from tensorflow.contrib import seq2seq

number_heroes = 64
number_maps = 14
number_of_game_types = 4
features_count = (number_heroes * 11)
lstm_size = 256
lstm_layers = 2

number_of_labels = 2

standard_deviation = 1 / np.sqrt(features_count)
print(standard_deviation)

#hyperparameters
number_of_hidden_nodes_layer1 = 1000

batch_size = 50
learning_rate = 0.03
epochs = 5
dropout_keep_prob = 1.0

data_all = 'training_data/hots_final_hot_encoding_rnn.csv'

sys.stdout.write("\rLoading data")
hots_all = np.array(pd.read_csv(data_all))
sys.stdout.write("\rShuffling data")
np.random.shuffle(hots_all)

hots_results = np.array(hots_all[:,[features_count, features_count+1]])
hots_features = np.array(hots_all[:,:features_count])

num_rows, num_cols = hots_results.shape

train_set_count = math.floor((num_rows / 10) * 8)
val_test_set_count = math.floor((num_rows / 10) * 1)

train_x, val_x, test_x = hots_features[:train_set_count,:], hots_features[train_set_count:train_set_count+val_test_set_count,:], hots_features[train_set_count+val_test_set_count:,:]
train_y, val_y, test_y = hots_results[:train_set_count,:], hots_results[train_set_count:train_set_count+val_test_set_count,:], hots_results[train_set_count+val_test_set_count:,:]


features = tf.placeholder(tf.float32, [None, 11, number_heroes], name="features")
labels = tf.placeholder(tf.int32, [None, 2], name="labels")
keep_prob = tf.placeholder(tf.float32, shape=(), name="keep_prob")

train_feed_dict = {features: train_x, labels: train_y, keep_prob:dropout_keep_prob}
valid_feed_dict = {features: val_x, labels: val_y, keep_prob:1.0}
test_feed_dict = {features: test_x, labels: test_y, keep_prob:1.0}

#rnn


# Your basic LSTM cell
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

# Add dropout to the cell
drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

# Stack up multiple LSTM layers, for deep learning
cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)

# Getting an initial state of all zeros
initial_state = cell.zero_state(batch_size, tf.float32)

input_data_shape = tf.shape(features)
outputs, final_state = tf.nn.dynamic_rnn(cell, features, initial_state=initial_state)

# Probabilities for generating words
probs = tf.nn.softmax(outputs, name='probs')

# Loss function
cost = seq2seq.sequence_loss(
    outputs,
    labels,
    tf.ones([input_data_shape[0], input_data_shape[1]]))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)

# Gradient Clipping
gradients = optimizer.compute_gradients(cost)
capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
train_op = optimizer.apply_gradients(capped_gradients)

# The accuracy measured against the validation set
validation_accuracy = 0.0

# Measurements use for graphing loss and accuracy
log_batch_step = 50000
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

# #with tf.Session(config=tf.ConfigProto(allow_growth=True)) as session:

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    batch_count = int(math.ceil(len(train_x) / batch_size))

    for epoch_i in range(epochs):

        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')

        # The training cycle
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i * batch_size
            batch_features = train_x[batch_start:batch_start + batch_size]
            batch_labels = train_y[batch_start:batch_start + batch_size]

            inputs = [[[]]]

            for feature in batch_features:
                feature_vector = [[]]
                for start_i in range(0, 11 * number_heroes, 11):
                    end_i = start_i + batch_size
                    cell_vector = feature[start_i:end_i]
                    feature_vector.append(cell_vector)
                inputs.append(feature_vector)

            # Run optimizer and get loss
            _, l = session.run(
                [optimizer, cost],
                feed_dict={features: inputs, labels: batch_labels, keep_prob:dropout_keep_prob})

        #     # Log every 2000 batches
        #     if not batch_i % log_batch_step:
        #         # Calculate Training and Validation accuracy
        #         training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
        #         validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)
        #
        #         # Log batches
        #         previous_batch = batches[-1] if batches else 0
        #         batches.append(log_batch_step + previous_batch)
        #         loss_batch.append(l)
        #         train_acc_batch.append(training_accuracy)
        #         valid_acc_batch.append(validation_accuracy)
        #         print(validation_accuracy)
        #
        # # Check accuracy against Validation data
        # validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

    save_model_path = 'saved_model/tensorflow_hots_model'
    saver = tf.train.Saver()
    save_path = saver.save(session, save_model_path)

# loss_plot = plt.subplot(211)
# loss_plot.set_title('Loss')
# loss_plot.plot(batches, loss_batch, 'g')
# loss_plot.set_xlim([batches[0], batches[-1]])
# acc_plot = plt.subplot(212)
# acc_plot.set_title('Accuracy')
# acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
# acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
# acc_plot.set_ylim([0, 1.0])
# acc_plot.set_xlim([batches[0], batches[-1]])
# acc_plot.legend(loc=4)
# plt.tight_layout()
# plt.show()

sys.stdout.write("\rSaving test data")
np.savetxt('training_data/test_data/test_x.csv', test_x, fmt='%i', delimiter=",")
np.savetxt('training_data/test_data/test_y.csv', test_y, fmt='%i', delimiter=",")
print('Validation accuracy at {}'.format(validation_accuracy))


# test_accuracy = 0.0
#
# with tf.Session() as session:
#     session.run(init)
#     batch_count = int(math.ceil(len(train_x) / batch_size))
#
#     for epoch_i in range(epochs):
#
#         # Progress bar
#         batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')
#
#         # The training cycle
#         for batch_i in batches_pbar:
#             # Get a batch of training features and labels
#             batch_start = batch_i * batch_size
#             batch_features = train_x[batch_start:batch_start + batch_size]
#             batch_labels = train_x[batch_start:batch_start + batch_size]
#
#             # Run optimizer
#             _ = session.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})
#
#         # Check accuracy against Test data
#         test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)
#
# assert test_accuracy >= 0.80, 'Test accuracy at {}, should be equal to or greater than 0.80'.format(test_accuracy)
# print('Nice Job! Test Accuracy is {}'.format(test_accuracy))




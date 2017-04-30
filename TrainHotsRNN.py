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
embed_size = 20

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


features = tf.placeholder(tf.int32, [None, 11, number_heroes], name="features")
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

reshaped_featues = tf.reshape(features, [batch_size*11, number_heroes])
print(reshaped_featues)
embedding = tf.Variable(tf.random_uniform([number_heroes, embed_size], -1, 1))
print(embedding)
embed = tf.nn.embedding_lookup(embedding, reshaped_featues)

print(embed)
reshaped_embed = tf.reshape(embed, [batch_size, 11, embed_size])

print(reshaped_embed)

outputs, final_state = tf.nn.dynamic_rnn(cell, reshaped_embed, initial_state=initial_state)

#print(outputs[:, -1].shape)
#print(labels.shape)

predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 2, activation_fn=tf.sigmoid)
cost = tf.losses.mean_squared_error(labels, predictions)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# The accuracy measured against the validation set
validation_accuracy = 0.0

# Measurements use for graphing loss and accuracy
log_batch_step = 50000
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

# #with tf.Session(config=tf.ConfigProto(allow_growth=True)) as session:

def getBatches(batch_x, batch_y, b_size=100):
    number_features = batch_y.shape[0]
    number_of_batches = np.floor(number_features/b_size).astype(np.int)
    print(number_of_batches)
    # The training cycle
    for batch_i in range(number_of_batches):
        # Get a batch of training features and labels
        batch_start = batch_i * b_size
        batch_features = batch_x[batch_start:batch_start + b_size]
        batch_labels = batch_y[batch_start:batch_start + b_size]

        rows, _ = batch_features.shape
        # inputs = np.empty([rows, 11, number_heroes])
        inputs = []

        # print(batch_features.shape)
        # print(batch_labels.shape)

        for feature in batch_features:
            feature_vector = np.empty([11, number_heroes])
            for start_i in range(0, 11 * number_heroes, number_heroes):
                end_i = start_i + b_size
                cell_vector = feature[start_i:end_i]
                np.append(feature_vector, cell_vector)

            # np.append(inputs, feature_vector)
            inputs.append(feature_vector)

        inputs = np.array(inputs)
        yield inputs, batch_labels

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    iteration = 1
    for epoch_i in range(epochs):

         for x, y in getBatches(train_x, train_y, batch_size):

            # Run optimizer and get loss
            #print(x.shape)
            #print(y.shape)
            #print(dropout_keep_prob)
            loss, state, _ = session.run([cost, final_state, optimizer], feed_dict={features: x, labels: y, keep_prob:dropout_keep_prob})
            print(loss)

            if iteration % 500 == 0:
                print("Epoch: {}/{}".format(epoch_i, epochs),"Iteration: {}".format(iteration), "Train loss: {:.3f}".format(loss))

            if iteration % 2500 == 0:
                val_acc = []
                val_state = session.run(cell.zero_state(batch_size, tf.float32))
                for x_val, y_val in getBatches(val_x, val_y, batch_size):
                    print(x_val.shape)
                    print(y_val.shape)
                    feed = {features: x_val, labels: y_val, keep_prob: 1.0}
                    batch_acc, val_state = session.run([accuracy, final_state], feed_dict=feed)
                    print(batch_acc)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            iteration += 1

    save_model_path = 'saved_model/tensorflow_hots_model'
    saver = tf.train.Saver()
    save_path = saver.save(session, save_model_path)


sys.stdout.write("\rSaving test data")
np.savetxt('training_data/test_data/test_x.csv', test_x, fmt='%i', delimiter=",")
np.savetxt('training_data/test_data/test_y.csv', test_y, fmt='%i', delimiter=",")
print('Validation accuracy at {}'.format(validation_accuracy))

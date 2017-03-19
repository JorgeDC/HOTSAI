import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import sys

number_heroes = 64
number_maps = 14
number_of_game_types = 4
features_count = (number_heroes * 2) + number_maps + number_of_game_types

number_of_labels = 2

#hyperparameters
number_of_hidden_nodes = 100
batch_size = 128
learning_rate = 0.003
epochs = 1

data_all = 'training_data/hots_final_hot_encoding.csv'

sys.stdout.write("\rLoading data")
hots_all = np.array(pd.read_csv(data_all))
sys.stdout.write("\rShuffling data")
np.random.shuffle(hots_all)

hots_results = np.array(hots_all[:,[features_count, features_count+1]])
hots_features = np.array(hots_all[:,:features_count])

num_rows, num_cols = hots_results.shape

train_set_count = math.floor((num_rows / 10) * 3)
val_test_set_count = math.floor((num_rows / 10) * 2)

train_x, val_x, test_x = hots_features[:train_set_count,:], hots_features[train_set_count:train_set_count+val_test_set_count,:], hots_features[train_set_count+val_test_set_count:,:]
train_y, val_y, test_y = hots_results[:train_set_count,:], hots_results[train_set_count:train_set_count+val_test_set_count,:], hots_results[train_set_count+val_test_set_count:,:]

print(train_x.shape)
print(val_x.shape)
print(test_x.shape)

features = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.float32)

weights_hidden_layer = tf.Variable(tf.truncated_normal((features_count, number_of_hidden_nodes)))
biases_hidden_layer = tf.Variable(tf.zeros(number_of_hidden_nodes))

train_feed_dict = {features: train_x, labels: train_y}
valid_feed_dict = {features: val_x, labels: val_y}
test_feed_dict = {features: test_x, labels: test_y}

logits_hidden_layer = tf.matmul(features, weights_hidden_layer) + biases_hidden_layer
hidden_layer = tf.nn.relu(logits_hidden_layer)

weights_prediction = tf.Variable(tf.truncated_normal((number_of_hidden_nodes, number_of_labels)))
biases_prediction = tf.Variable(tf.zeros(number_of_labels))

logits_prediction = tf.matmul(hidden_layer, weights_prediction) + biases_prediction
prediction = tf.nn.softmax(logits_prediction)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))

# Determine if the predictions are correct
is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
# Calculate the accuracy of the predictions
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()


# The accuracy measured against the validation set
validation_accuracy = 0.0

# Measurements use for graphing loss and accuracy
log_batch_step = 50
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

with tf.Session() as session:
    session.run(init)
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

            # Run optimizer and get loss
            _, l = session.run(
                [optimizer, cost],
                feed_dict={features: batch_features, labels: batch_labels})

            # Log every 50 batches
            if not batch_i % log_batch_step:
                # Calculate Training and Validation accuracy
                training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                # Log batches
                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)
                loss_batch.append(l)
                train_acc_batch.append(training_accuracy)
                valid_acc_batch.append(validation_accuracy)

        # Check accuracy against Validation data
        validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

loss_plot = plt.subplot(211)
loss_plot.set_title('Loss')
loss_plot.plot(batches, loss_batch, 'g')
loss_plot.set_xlim([batches[0], batches[-1]])
acc_plot = plt.subplot(212)
acc_plot.set_title('Accuracy')
acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
acc_plot.set_ylim([0, 1.0])
acc_plot.set_xlim([batches[0], batches[-1]])
acc_plot.legend(loc=4)
plt.tight_layout()
plt.show()

print('Validation accuracy at {}'.format(validation_accuracy))
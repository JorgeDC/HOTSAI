import tflearn
import pandas as pd
import numpy as np
import math

number_heroes = 63

data_path_features = 'hots_final_hot_encoding.csv'
data_path_results = 'hots_final_results.csv'
hots_features = np.array(pd.read_csv(data_path_features))
hots_results = np.array(pd.read_csv(data_path_results))

num_rows, num_cols = hots_results.shape

train_val_set_count = math.floor((num_rows / 10) * 8)

train_val_x, test_x = hots_features[:train_val_set_count,:], hots_features[train_val_set_count:,:]
train_val_y, test_y = hots_results[:train_val_set_count,:], hots_results[train_val_set_count:,:]

net = tflearn.input_data([None, 126])
net = tflearn.fully_connected(net, 25, activation='tanh')
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy')
model = tflearn.DNN(net)

model.fit(train_val_x, train_val_y, validation_set=0.1, show_metric=True, batch_size=2000, n_epoch=100)

model.save("hots_weights")

# Compare the labels that our model predicts with the actual labels
predictions = (np.array(model.predict(test_x))[:,0] >= 0.5).astype(np.int_)

# Calculate the accuracy, which is the percentage of times the predicated labels matched the actual labels
test_accuracy = np.mean(predictions == test_y[:,0], axis=0)

# Print out the result
print("Test accuracy: ", test_accuracy)
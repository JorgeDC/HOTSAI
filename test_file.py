import numpy as np
import tensorflow as tf

test_3d_tensor = tf.Variable([[[1,2,3,4], [5,6,7,8]]
                                 , [[9,10,11,12], [13,14,15,16]]])

print(test_3d_tensor)
reshaped_tensor = tf.reshape(test_3d_tensor, [2*2, 4])
print(reshaped_tensor)
#a = tf.Print(reshaped_tensor, [reshaped_tensor], message="This is a: ")


reshaped_tensor_back = tf.reshape(reshaped_tensor, [2, 2, 4])
print(reshaped_tensor_back)

#initialize the variable
init_op = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init_op)  # execute init_op
    # print the random values that we sample
    print(session.run(reshaped_tensor))
    print(session.run(reshaped_tensor_back))
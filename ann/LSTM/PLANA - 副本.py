# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import matplotlib
import os

tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


# In[2]:

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-5)


# In[3]:

# train Parameters
seq_length = 7
data_dim = 5
hidden_dim = 7
output_dim = 1
learning_rate = 0.01  # update learning rate
iterations = 3000
num_layers = 3


# In[4]:
xy = np.loadtxt('PLANA.csv', delimiter=',')
#xy0 = np.loadtxt('PLANA0.csv', delimiter=',')
#xy1 = np.loadtxt('PLANA1.csv', delimiter=',')

xy = MinMaxScaler(xy)
x = xy
y = xy[:, [3]]  # label




# In[5]:


#build a dataset
dataX0 = []
dataY0 = []
for i in range(0, len(y) - 14, 2):
    _x0 = []
    for j in range(i, i + 14, 2):
          _x0.append(x[j, :])
    _x0 = np.array(_x0)
    _x0.reshape((7,5))
    _y0 = y[i + 14]  # Next close price
    print(_x0, "->", _y0)
    dataX0.append(_x0)
    dataY0.append(_y0)




#build a dataset
dataX1 = []
dataY1 = []
for i in range(1, len(y) - 14, 2):
    _x1 = []
    for j in range(i, i + 14, 2):
          _x1.append(x[j, :])
    _x1 = np.array(_x1)
    _x1.reshape((7,5))
    _y1 = y[i + 14]  # Next close price
    print(_x1, "->", _y1)
    dataX1.append(_x1)
    dataY1.append(_y1)

# In[6]:


train_size0 = int(len(dataY0) * 0.7)

trainX00, testX00 = np.array(dataX0[0:train_size0]), np.array(
    dataX0[train_size0:len(dataX0)])
trainY00, testY00 = np.array(dataY0[0:train_size0]), np.array(
    dataY0[train_size0:len(dataY0)])

trainX10, testX10 = np.array(dataX0[0:train_size0]), np.array(dataX0[train_size0:train_size0+180])
trainY10, testY10 = np.array(dataY0[0:train_size0]), np.array(dataY0[train_size0:train_size0+180])

trainX20, testX20 = np.array(dataX0[180:train_size0+180]), np.array(dataX0[train_size0+180:train_size0+360])
trainY20, testY20 = np.array(dataY0[180:train_size0+180]), np.array(dataY0[train_size0+180:train_size0+360])

trainX30, testX30 = np.array(dataX0[360:train_size0+360]), np.array(dataX0[train_size0+360:train_size0+540])
trainY30, testY30 = np.array(dataY0[360:train_size0+360]), np.array(dataY0[train_size0+360:train_size0+540])

trainX40, testX40 = np.array(dataX0[540:train_size0+540]), np.array(dataX0[train_size0+540:len(dataY0)])
trainY40, testY40 = np.array(dataY0[540:train_size0+540]), np.array(dataY0[train_size0+540:len(dataY0)])



train_size1 = int(len(dataY1) * 0.7)
test_size1 = len(dataY1) - train_size1
trainX01, testX01 = np.array(dataX1[0:train_size1]), np.array(
    dataX1[train_size1:len(dataX1)])
trainY01, testY01 = np.array(dataY1[0:train_size1]), np.array(
    dataY1[train_size1:len(dataY1)])

trainX11, testX11 = np.array(dataX1[0:train_size1]), np.array(dataX1[train_size1:train_size1+180])
trainY11, testY11 = np.array(dataY1[0:train_size1]), np.array(dataY1[train_size1:train_size1+180])

trainX21, testX21 = np.array(dataX1[180:train_size1+180]), np.array(dataX1[train_size1+180:train_size1+360])
trainY21, testY21 = np.array(dataY1[180:train_size1+180]), np.array(dataY1[train_size1+180:train_size1+360])

trainX31, testX31 = np.array(dataX1[360:train_size1+360]), np.array(dataX1[train_size1+360:train_size1+540])
trainY31, testY31 = np.array(dataY1[360:train_size1+360]), np.array(dataY1[train_size1+360:train_size1+540])

trainX41, testX41 = np.array(dataX1[540:train_size1+540]), np.array(dataX1[train_size1+540:len(dataY1)])
trainY41, testY41 = np.array(dataY1[540:train_size1+540]), np.array(dataY1[train_size1+540:len(dataY1)])



# In[7]:

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])

# In[8]:

# build a LSTM network
# build a LSTM network
# stacked_rnn = []
# for _ in range(num_layers):
#    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
#    stacked_rnn.append(cell)
# lstm_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
# outputs, _states = tf.nn.dynamic_rnn(lstm_cell_m, X, dtype=tf.float32)

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# In[9]:

loss = tf.reduce_sum(tf.square(Y_pred - Y))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# In[10]:

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
mse = tf.reduce_mean(tf.square(predictions - targets))
rsquare = 1 - (tf.reduce_sum(tf.square(targets - predictions)) /
               tf.reduce_sum(tf.square(targets - tf.reduce_mean(targets))))
mape = tf.reduce_mean(tf.abs((targets - predictions) / targets))

# In[11]:

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)


    # Training step
    for i in range(1000):
        _, step_loss = sess.run([train, loss], feed_dict={
            X: trainX00, Y: trainY00})
        print("[step: {}] loss: {}".format(i, step_loss))

        # Test step
        test_predict00 = sess.run(Y_pred, feed_dict={X: testX00})


        '''
        rsquare_val = sess.run(rsquare, feed_dict={
            targets: b1, predictions: a1})
        print("          RSQUARE: {}".format(rsquare_val))
'''


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(1000):
        _, step_loss = sess.run([train, loss], feed_dict={
            X: trainX10, Y: trainY10})
        print("[step: {}] loss: {}".format(i, step_loss))

        # Test step
        test_predict10 = sess.run(Y_pred, feed_dict={X: testX10})


        '''
        rsquare_val = sess.run(rsquare, feed_dict={
            targets: b2, predictions: a2})
        print("          RSQUARE: {}".format(rsquare_val))
'''

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(1000):
        _, step_loss = sess.run([train, loss], feed_dict={
            X: trainX20, Y: trainY20})
        print("[step: {}] loss: {}".format(i, step_loss))

        # Test step
        test_predict20 = sess.run(Y_pred, feed_dict={X: testX20})


        '''
        rsquare_val = sess.run(rsquare, feed_dict={
            targets: b3, predictions: a3})


        print("          RSQUARE: {}".format(rsquare_val))
'''

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(1000):
        _, step_loss = sess.run([train, loss], feed_dict={
            X: trainX30, Y: trainY30})
        print("[step: {}] loss: {}".format(i, step_loss))

        # Test step
        test_predict30 = sess.run(Y_pred, feed_dict={X: testX30})



        '''
        rsquare_val = sess.run(rsquare, feed_dict={
            targets: b4, predictions: a4})


        print("          RSQUARE: {}".format(rsquare_val))

'''
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(2000):
        _, step_loss = sess.run([train, loss], feed_dict={
            X: trainX40, Y: trainY40})
        print("[step: {}] loss: {}".format(i, step_loss))

        # Test step
        test_predict40 = sess.run(Y_pred, feed_dict={X: testX40})



with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(1000):
        _, step_loss = sess.run([train, loss], feed_dict={
            X: trainX01, Y: trainY01})
        print("[step: {}] loss: {}".format(i, step_loss))

        # Test step
        test_predict01 = sess.run(Y_pred, feed_dict={X: testX01})

        '''
        rsquare_val = sess.run(rsquare, feed_dict={
            targets: b1, predictions: a1})
        print("          RSQUARE: {}".format(rsquare_val))
'''

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(1000):
        _, step_loss = sess.run([train, loss], feed_dict={
            X: trainX11, Y: trainY11})
        print("[step: {}] loss: {}".format(i, step_loss))

        # Test step
        test_predict11 = sess.run(Y_pred, feed_dict={X: testX11})

        '''
        rsquare_val = sess.run(rsquare, feed_dict={
            targets: b2, predictions: a2})
        print("          RSQUARE: {}".format(rsquare_val))
'''

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(1000):
        _, step_loss = sess.run([train, loss], feed_dict={
            X: trainX21, Y: trainY21})
        print("[step: {}] loss: {}".format(i, step_loss))

        # Test step
        test_predict21 = sess.run(Y_pred, feed_dict={X: testX21})

        '''
        rsquare_val = sess.run(rsquare, feed_dict={
            targets: b3, predictions: a3})


        print("          RSQUARE: {}".format(rsquare_val))
'''

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(1000):
        _, step_loss = sess.run([train, loss], feed_dict={
            X: trainX31, Y: trainY31})
        print("[step: {}] loss: {}".format(i, step_loss))

        # Test step
        test_predict31 = sess.run(Y_pred, feed_dict={X: testX31})

        '''
        rsquare_val = sess.run(rsquare, feed_dict={
            targets: b4, predictions: a4})


        print("          RSQUARE: {}".format(rsquare_val))

'''
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(2000):
        _, step_loss = sess.run([train, loss], feed_dict={
            X: trainX41, Y: trainY41})
        print("[step: {}] loss: {}".format(i, step_loss))

        # Test step
        test_predict41 = sess.run(Y_pred, feed_dict={X: testX41})



test_predict0 = np.vstack((test_predict10,test_predict20,test_predict30,test_predict40))
test_predict1 = np.vstack((test_predict11,test_predict21,test_predict31,test_predict41))

# In[12]:
test_pre = []
for i in range(0, test_size1):
    test_pre.append(test_predict0[i])
    test_pre.append(test_predict1[i])

testY = []
for i in range(0, test_size1):
    testY.append(testY00[i])
    testY.append(testY01[i])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    rmse_val = sess.run(rmse, feed_dict={
           targets: testY, predictions: test_pre})
    rsquare_val = sess.run(rsquare, feed_dict={
        targets: testY, predictions: test_pre})
    mape_val = sess.run(mape, feed_dict={
           targets: testY, predictions: test_pre})
print("          RMSE: {}".format(rmse_val))
print("          RSQUARE: {}".format(rsquare_val))
print("          MAPE: {}".format(mape_val))

plt.figure(figsize=(15, 5))
testYline = plt.plot(testY, label="observation", color="k")
test_predictline = plt.plot(test_pre, label="prediction", color="r")
plt.xlabel("Time Period")
plt.ylabel("Value")
plt.legend(handles=[testYline[0], test_predictline[0]],
           loc="upper left")
plt.savefig('14.jpg')

# In[ ]:

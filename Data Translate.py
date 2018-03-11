import numpy as np
import pandas as pd
import tensorflow as tf

def AddSeries(input_series,ind_a):

# Add ind_a columns of Input_Series for
# The array
# Return the data frame
    size_in = input_series.size
    t_s = input_series
    t_i =1
    for t_i in range(1, ind_a):
        t_b = pd.Series(np.zeros(t_i)).append(input_series[0:size_in-t_i],ignore_index=True)
        df = [t_s,t_b]
        t_s = pd.concat(df,axis=1)
    return t_s


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#Retrieve the data from Yahoo
#https://finance.yahoo.com/quote/%5EHSI/history?p=%5EHSI
#put the data into DataHandling

Flnb_HSI="D:\\DataHandling\\HSI.csv"
Flnb_VIX="D:\\DataHandling\\VIX.csv"

#Get the data from CSV
data_A = pd.read_csv(Flnb_HSI)
data_B = pd.read_csv(Flnb_VIX)

#Merge two data set  HSI, Volumn, VIX ,
data_M =data_A.merge(data_B, left_on='Date', right_on='Date', how='inner')
data_M.fillna(0)

#Add axis into 4 for projection

data_Close_H= AddSeries(data_M['Close_x'],4)
#data_Close_V= AddSeries(data_M['Close_y'],4)
#Re_D=[data_Close_H,data_Close_V]
#Data_x = pd.concat(Re_D,axis=1)
Data_x= data_Close_H

#split the data with Target and Training data

Data_y = Data_x.iloc[5:100,0:1]
Data_xin = Data_x.iloc[5:100,1:]


xs = tf.placeholder(tf.float32, [None, 3])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 3, 30, activation_function=tf.nn.relu)
prediction = add_layer(l1, 30, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


for i in range(100):
    # training
    sess.run(train_step, feed_dict={xs: Data_xin, ys: Data_y})
    # to see the step improvement
    print(sess.run(loss, feed_dict={xs: Data_xin, ys: Data_y}))
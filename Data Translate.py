import numpy as np
import pandas as pd
import tensorflow as tf

def AddSeries(Input_Series,ind_a)

# Add ind_a columns of Input_Series for
# The array
# Return the data frame

    Size_In = Input_Series.size
    T_S = Input_Series
    t_i =1
    For t_i in range(1, ind_a)
        T_B = pd.Series(np.zeros(t_i)).append(Input_Series[0:Size_In-t_i],ignore_index=True)
        DF = [T_S,T_B]
        T_S = pd.concat(DF,axis=1)

Return T_S

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
data_Close_V= AddSeries(data_M['Close_y'],4)
Re_D=[data_Close_H,data_Close_V]
Data_x = pd.concat(Re_D,axis=1)



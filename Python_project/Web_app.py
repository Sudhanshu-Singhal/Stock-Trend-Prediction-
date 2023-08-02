# -*- coding: utf-8 -*-
"""
Created on Thu May 12 09:37:39 2022

@author: Pikachu
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#import pandas_datareader as data 
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
from datetime import date
from sklearn.metrics import r2_score


date = date.today()

start = '2010-01-01'
end = date

st.title('Stock Price Prediction & Forecasting')

x = {'Symbols': ['AAPL', 'ORCL', 'SBI.NS', 'PNB.NS', 'TSlA', 'TCS.NS', 'AMZN', 'GOOG', 'NFLX', 'WMT'],
     'Company': ['Apple Inc.', 'Oracle Corporation', 'State Bank of India', 'Punjab National Bank', 
     'Tesla Inc.', 'Tata Consultancy Services Limited', 'Amazon.com Inc.', 'Alphabet Inc.', 
     'Netflix Inc.', 'Walmart Inc.']}

df_table = pd.DataFrame(x)

st.table(df_table)

user_input = st.selectbox('Select Stock Ticker: ',('AAPL', 'ORCL', 'SBI.NS', 'PNB.NS', 'TSLA', 
                                                 'TCS.NS', 'AMZN', 'GOOG','NFLX', 'WMT'))

df = yf.download(user_input, start, end)

df.reset_index(inplace = True)

df['Date'] = pd.to_datetime(df['Date']).dt.date


values = st.slider('Select a range of years: ',df.Date[0], end, (df.Date[0], end))
st.write('Values:', values)
#Describing the data
st.subheader('Data (from {} - {})'.format(values[0], values[1]))

#df = df.query("Date BETWEEN {} AND {}".format(values[0], values[1]))

df = df[(df['Date'] >= values[0]) & (df['Date'] <= values[1])]
st.write('Description')
st.write(df.describe()) 


#Data showing
st.subheader('Raw Data (first 10 rows) ')
st.write(df.head(10))

#visualization
st.subheader('Closing Price VS Time ')

fig = plt.figure(figsize = (16,9))
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)


st.subheader('Closing Price VS Time of 100 days moving averge')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (16,9))
plt.plot(ma100,'r')
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)


st.subheader('Closing Price VS Time of 100 & 200 days moving average')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (16,9))
plt.plot(ma200, 'r', label = '100 days moving average')
plt.plot(ma100, 'g', label = '200 days moving average')
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


# Splitting the data into training and testing data 
train_data = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
test_data = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])


#import sklearn.preprocessing

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
train_data_array = scaler.fit_transform(train_data)

x_train = []
y_train = []

for i in range(100, train_data_array.shape[0]):
    x_train.append(train_data_array[i-100: i])
    y_train.append(train_data_array[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)


#load the LSTM model
model = load_model('keras_model.h5')


test_data_array = scaler.fit_transform(test_data)

x_test = []
y_test = []

for i in range(100, test_data_array.shape[0]):
    x_test.append(test_data_array[i-100: i])
    y_test.append(test_data_array[i,0])
    
x_test, y_test = np.array(x_test), np.array(y_test)



# Predictions

train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

train_pred = scaler.inverse_transform(train_pred)  # prec
test_pred = scaler.inverse_transform(test_pred)

# calculating RMSE performance metrices
import math
from sklearn.metrics import mean_squared_error
train_RMSE = math.sqrt(mean_squared_error(y_train, train_pred))
test_RMSE = math.sqrt(mean_squared_error(y_test, test_pred))



y_test = y_test.reshape(-1,1)
y_test = scaler.inverse_transform(y_test)


# Final Graph of prediction 

st.subheader('Prediction VS Original')

fig2 = plt.figure(figsize = (16,9))
plt.plot(y_test[0: (y_test.shape[0]-100)], 'b', label = 'original price')
plt.plot(test_pred, 'r', label = 'predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)



# Accuracy of this model 
accuracy = round((r2_score(y_test, test_pred))*100, 2)
st.subheader('Accuracy of the model: ')
st.write(accuracy, '%')




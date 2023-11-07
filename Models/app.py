import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yfin
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from datetime import date

start = '2010-01-01'
end = date.today()

st.title("Stock Trend Prediction")

user_input = st.text_input("Enter Stock Ticker","AAPL")

ticker = yfin.Ticker(user_input)

df = ticker.history(interval="1d",start=start, end=end)

# Describing Data
st.subheader("Data From 2010 to 2022")
st.write(df.describe())

st.subheader("Closing Price vs. Time Chart")
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs. Time Chart with 100MA")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs. Time Chart with 100MA and 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close)
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7): int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

model = load_model('lstm_model')

past_100_days = data_training.tail(100)
df2 = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data =  scaler.fit_transform(df2)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_pred = model.predict(x_test)

scale_factor = 1/(scaler.scale_[0])
y_pred = y_pred*scale_factor
y_test = y_test*scale_factor

st.subheader("Predicted vs Original")
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label="Original Price")
plt.plot(y_pred,'r',label='Predicted Price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)

from sklearn.metrics import r2_score
mse = r2_score(y_test, y_pred)

st.subheader("R2 Score : ")
st.write(mse)

# st.divider()

# st.subheader('Select the number of days you want prediction of : ')

# days = st.slider("Days between 1 - 31",min_value=1,max_value=31,step=1)

# st.subheader("Predictions : ")

# days_predictions = data_testing.tail(100)
# # data_testing.tail(100)
# for i in range(1,days+1):
#     curr_window = days_predictions.tail(100)
#     df3 = scaler.fit_transform(curr_window)
#     tmp_input = np.array(df3)
#     x = model.predict(tmp_input)
#     y = x*scale_factor
#     days_predictions = pd.concat([days_predictions,y],ignore_index=True)

# fig3 = plt.figure(figsize=(12,6))
# plt.plot(y_test,'b',label="Original Price")
# plt.plot(y_pred,'r',label='Predicted Price')
# plt.plot(days_predictions.tail(days),'g',label="Further Predictions")
# plt.xlabel("Time")
# plt.ylabel("Price")
# plt.legend()
# st.pyplot(fig3)  
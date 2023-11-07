import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yfin

start = '2010-01-01'
end = '2022-12-31'

ticker = yfin.Ticker('AAPL')

df = ticker.history(interval="1d",start=start, end=end)
df.head()
df.tail()

df = df.reset_index()
df.head()

df = df.drop(['Date','Dividends','Stock Splits'],axis=1)
df.head()

plt.plot(df.Close)

ma100 = df.Close.rolling(100).mean()
ma100

plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')

ma200 = df.Close.rolling(200).mean()
ma200

plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')

df.shape

# Split the Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7): int(len(df))])

print(data_training.shape)
print(data_testing.shape)

#Scale down the Data between 0 and 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)
data_training_array.shape

x_train = []
y_train = []

for i in range (100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

x_train.shape

# ML Model

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(50,activation="relu",return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(70,activation="relu",return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(90,activation="relu",return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(120,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=1))

model.summary()

model.compile(optimizer="adam",loss="mean_squared_error")
model.fit(x_train,y_train, epochs = 50)

model.save('lstm_model')

data_testing.head()

past_100_days = data_training.tail(100)
past_100_days.head()

df2 = pd.concat([past_100_days, data_testing], ignore_index=True)
df2.head()
df2.tail()
# df2.shape

# data_testing.tail()

input_data =  scaler.fit_transform(df2)
input_data 

input_data.shape

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

print(x_test.shape)
print(y_test.shape)

# Make Predictions

y_pred = model.predict(x_test)
y_pred
y_pred.shape

scale_factor = 1/(scaler.scale_)
y_pred = y_pred*scale_factor
y_test = y_test*scale_factor

plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label="Original Price")
plt.plot(y_pred,'r',label='Predicted Price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)

from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# load dataset
def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')

#利用二氧化硫(SO2)、二氧化氮(NO2)、臭氧(O3)、細懸浮微粒(PM2.5)


df = read_csv('daliday.csv', header = 0 )
#print(df[:2])




dfday = df['監測日期']
dfSO2 = df['二氧化硫 SO2 (ppb)']
#dfCO = df['一氧化碳 CO (ppm)']
#dfCO2 = df['二氧化碳 CO2 (ppm)']
dfO3 = df['臭氧 O3 (ppb)']
dfO3MAX8 = df['臭氧8小時最大值 O3_Max_8_HR (ppb)']
#dfPM10 = df['懸浮微粒 PM 10  (μg/m 3 )']
dfPM25 = df['細懸浮微粒 PM 2.5  (μg/m 3 )']

#dfNOX = df['氮氧化物 NOx (ppb)']
#dfNO = df['一氧化氮 NO (ppb)']
dfNO2 = df['二氧化氮 NO2 (ppb)']

#dfTHC = df['總碳氫化合物 THC (ppm)']
#dfNMHC = df['非甲烷碳氫化合物 NMHC (ppm)']
#dfCH4 = df['甲烷 CH4 (ppm)']
dfWIND_SPEED = df['風速 WIND_SPEED (m/sec)']
dfWS_HR = df['小時風速值 WS_HR (m/sec)']
dfAMB_TEMP = df['溫度 AMB_TEMP (℃)']

dfAMB_TEMP2 = df['大氣溫度 AMB_TEMP (℃)']
dfRH = df['相對濕度 RH (percent)']
dfPSI = df['空氣污染指標PSI PSI']

#利用二氧化硫(SO2)
#因為前20筆( 0 - 19 )是最近20天的資料，所以第 0 筆到第 19 筆拿來當測試資料
#從第20筆開始到最後一筆是訓練資料
train, test = dfSO2[20:-2], dfSO2[0:20]

#從第20筆開始起拿80筆當訓練資料    (老師上課提過一百筆資料 拿八十筆當訓練資料  二十筆當測試資料)
train, test = dfSO2[20:80], dfSO2[0:20]

# 以此類推  改為800 200
train, test = dfSO2[200:800], dfSO2[0:200]


#二氧化氮(NO2)
train, test = dfNO2[20:-2], dfNO2[0:20]
train, test = dfNO2[20:80], dfNO2[0:20]
train, test = dfNO2[200:800], dfNO2[0:200]

#臭氧(O3)
train, test = dfO3[20:-2], dfO3[0:20]
train, test = dfO3[20:80], dfO3[0:20]
train, test = dfO3[200:800], dfO3[0:200]

# 細懸浮微粒(PM2.5)
train, test = dfPM25[20:-2], dfPM25[0:20]
train, test = dfPM25[20:80], dfPM25[0:20]
train, test = dfPM25[200:800], dfPM25[0:200]

import lstmfunc
# load dataset
series = read_csv('daliday.csv', header = 0 )

series = series[['監測日期', '二氧化硫 SO2 (ppb)']]

# transform data to be stationary
raw_values = series.values

diff_values = lstmfunc.difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values


# split data into train and test-sets
#train, test = dfPM25[20:80], dfPM25[0:20]
train, test = supervised_values[0:-12], supervised_values[-12:]

# transform the scale of the data
scaler, train_scaled, test_scaled = lstmfunc.scale(train, test)

# fit the model
lstm_model = lstmfunc.fit_lstm(train_scaled, 1, 3000, 4)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
	# make one-step forecast
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = lstmfunc.forecast_lstm(lstm_model, 1, X)
	# invert scaling
	yhat = lstmfunc.invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = lstmfunc.inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# store forecast
	predictions.append(yhat)
	expected = lstmfunc.raw_values[len(train) + i + 1]
	print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

from matplotlib import pyplot
# report performance
rmse = lstmfunc.sqrt(lstmfunc.mean_squared_error(raw_values[-12:], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(raw_values[-12:])
pyplot.plot(predictions)
pyplot.show()


#print(dfAMB_TEMP[0])
#print(dfAMB_TEMP[1])
#Day1AMB_TEMP = float(dfAMB_TEMP[0])
#print(Day1AMB_TEMP)
#Day1AMB_TEMPadd05 = Day1AMB_TEMP + 0.5
#Day1AMB_TEMPsub05 = Day1AMB_TEMP - 0.5
#dfTC = df[ ( df['溫度 AMB_TEMP (℃)'] > Day1AMB_TEMPsub05) & (df['溫度 AMB_TEMP (℃)'] < Day1AMB_TEMPadd05) ]
#print(dfTC)

'''
print(dfday[:2])
print('============')
print(dfSO2[:2])
print('============')
print(dfCO[:2])
print('============')
print(dfCO2[:2])
print('============')
print(dfO3[:2])
print('============')
print(dfO3MAX8[:2])
print('============')
print(dfPM10[:2])
print('============')
print(dfPM25[:2])
print('============')
print(dfNOX[:2])
print('============')
print(dfNO[:2])
print('============')
print(dfNO2[:2])
print('============')
print(dfTHC[:2])
print('============')
print(dfNMHC[:2])
print('============')
print(dfCH4[:2])
print('============')
print(dfWIND_SPEED[:2])
print('============')
print(dfWS_HR[:2])
print('============')
print(dfAMB_TEMP[:2])
print('============')
print(dfAMB_TEMP2[:2])
print('============')
print(dfRH[:2])
print('============')
print(dfPSI[:2])
'''

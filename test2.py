from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat






#df = read_csv('daliday.csv', header = 0 )
#print(df[:2])

#dfday = df['監測日期']
#dfSO2 = df['二氧化硫 SO2 (ppb)']
#dfCO = df['一氧化碳 CO (ppm)']
#dfCO2 = df['二氧化碳 CO2 (ppm)']
#dfO3 = df['臭氧 O3 (ppb)']
#dfO3MAX8 = df['臭氧8小時最大值 O3_Max_8_HR (ppb)']
#dfPM10 = df['懸浮微粒 PM 10  (μg/m 3 )']
#dfPM25 = df['細懸浮微粒 PM 2.5  (μg/m 3 )']

#dfNOX = df['氮氧化物 NOx (ppb)']
#dfNO = df['一氧化氮 NO (ppb)']
#dfNO2 = df['二氧化氮 NO2 (ppb)']

#dfTHC = df['總碳氫化合物 THC (ppm)']
#dfNMHC = df['非甲烷碳氫化合物 NMHC (ppm)']
#dfCH4 = df['甲烷 CH4 (ppm)']
#dfWIND_SPEED = df['風速 WIND_SPEED (m/sec)']
#dfWS_HR = df['小時風速值 WS_HR (m/sec)']
#dfAMB_TEMP = df['溫度 AMB_TEMP (℃)']

#dfAMB_TEMP2 = df['大氣溫度 AMB_TEMP (℃)']
#dfRH = df['相對濕度 RH (percent)']
#dfPSI = df['空氣污染指標PSI PSI']

#利用二氧化硫(SO2)
#因為前20筆( 0 - 19 )是最近20天的資料，所以第 0 筆到第 19 筆拿來當測試資料
#從第20筆開始到最後一筆是訓練資料
#train, test = dfSO2[20:-2], dfSO2[0:20]

#從第20筆開始起拿80筆當訓練資料    (老師上課提過一百筆資料 拿八十筆當訓練資料  二十筆當測試資料)
#train, test = dfSO2[20:80], dfSO2[0:20]

# 以此類推  改為800 200
#train, test = dfSO2[200:800], dfSO2[0:200]


#二氧化氮(NO2)
#train, test = dfNO2[20:-2], dfNO2[0:20]
#train, test = dfNO2[20:80], dfNO2[0:20]
#train, test = dfNO2[200:800], dfNO2[0:200]

#臭氧(O3)
#train, test = dfO3[20:-2], dfO3[0:20]
#train, test = dfO3[20:80], dfO3[0:20]
#train, test = dfO3[200:800], dfO3[0:200]

# 細懸浮微粒(PM2.5)
#train, test = dfPM25[20:-2], dfPM25[0:20]
#train, test = dfPM25[20:80], dfPM25[0:20]
#train, test = dfPM25[200:800], dfPM25[0:200]

#利用二氧化硫(SO2)、二氧化氮(NO2)、臭氧(O3)、細懸浮微粒(PM2.5)

import lstmfunc
# load dataset
series = read_csv('daliday.csv', header = 0 )
#拿掉沒值的資料   除去 - 號
series = series[ series['二氧化硫 SO2 (ppb)'] != '-' ]
#只選這兩個欄位    並且重新排序　　最新的日期在最下面
series = series[['監測日期', '二氧化硫 SO2 (ppb)']].sort_values(['監測日期'], ascending=[True])
#-2是為了排除兩筆NAN
series = series[-102:-2]


#print(series )


# transform data to be stationary
#.astype(float)  字串轉數字
raw_values = series['二氧化硫 SO2 (ppb)'].astype(float).values
print(raw_values)
diff_values = lstmfunc.difference(raw_values, 1)

# transform data to be supervised learning
supervised = lstmfunc.timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

#print(supervised_values)

# split data into train and test-sets
#train, test = dfPM25[20:80], dfPM25[0:20]
train, test = supervised_values[0:-20], supervised_values[-20:]


# transform the scale of the data
scaler, train_scaled, test_scaled = lstmfunc.scale(train, test)

# fit the model
lstm_model = lstmfunc.fit_lstm(train_scaled, 1, 1000, 4)
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

    print(len(test_scaled)+1-i)
    yhat = lstmfunc.inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    # store forecast
    predictions.append(yhat)
    expected = raw_values[len(train) + i + 1]
    print(len(train) + i + 1)
    print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

from matplotlib import pyplot
# report performance
rmse = lstmfunc.sqrt(lstmfunc.mean_squared_error(raw_values[-20:], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
#pyplot.plot(raw_values[-12:])

#pyplot.plot(raw_values[20:80])   #建模型的資料
pyplot.plot(raw_values[-20:])    #測試資料
pyplot.plot(predictions)         #預測出來的資料
pyplot.show()


#print(dfAMB_TEMP[0])
#print(dfAMB_TEMP[1])
#Day1AMB_TEMP = float(dfAMB_TEMP[0])
#print(Day1AMB_TEMP)
#Day1AMB_TEMPadd05 = Day1AMB_TEMP + 0.5
#Day1AMB_TEMPsub05 = Day1AMB_TEMP - 0.5
#dfTC = df[ ( df['溫度 AMB_TEMP (℃)'] > Day1AMB_TEMPsub05) & (df['溫度 AMB_TEMP (℃)'] < Day1AMB_TEMPadd05) ]
#print(dfTC)


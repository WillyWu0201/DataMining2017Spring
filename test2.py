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




df = read_csv('daliday.csv', header = 0)
print(df[:5])

dfday = df['監測日期']
dfSO2 = df['二氧化硫 SO2 (ppb)']
dfCO = df['一氧化碳 CO (ppm)']
dfCO2 = df['二氧化碳 CO2 (ppm)']
dfO3 = df['臭氧 O3 (ppb)']
dfO3MAX8 = df['臭氧8小時最大值 O3_Max_8_HR (ppb)']
dfPM10 = df['懸浮微粒 PM 10  (μg/m 3 )']
dfPM25 = df['細懸浮微粒 PM 2.5  (μg/m 3 )']

print(dfday[:5])
print('============')
print(dfSO2[:5])
print('============')
print(dfCO[:5])
print('============')
print(dfCO2[:5])
print('============')
print(dfO3[:5])
print('============')
print(dfO3MAX8[:5])
print('============')
print(dfPM10[:5])
print('============')
print(dfPM25[:5])

#	氮氧化物 NOx (ppb)	一氧化氮 NO (ppb)	二氧化氮 NO2 (ppb)	總碳氫化合物 THC (ppmC)	總碳氫化合物 THC (ppm)	非甲烷碳氫化合物 NMHC (ppm)	甲烷 CH4 (ppm)	風速 WIND_SPEED (m/sec)	小時風速值 WS_HR (m/sec)	溫度 AMB_TEMP (℃)	大氣溫度 AMB_TEMP (℃)	相對濕度 RH (percent)	空氣污染指標PSI PSI

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

# series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
series = read_csv('201001.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# transform to supervised learning
X = series.values
supervised = timeseries_to_supervised(X, 1)
# print(supervised.head())

#取出201001~201704大里的資訊
for year in range(2010, 2017):
    if year == 2017:
        for month in range(1,4):
              filePath = 'AQX_reName/%d%02d.csv' % (year, month)
              # print(filePath)
              df = read_csv(filePath, header=None)
              dfzzz = df[(df[1] == 'Taichung') & (df[3] == 'Big mile')]
              print(dfzzz)
    else:
        for month in range(1, 12):
            filePath = 'AQX_reName/%d%02d.csv' % (year, month)
            # print(filePath)
            df = read_csv(filePath, header=None)
            dfzzz = df[(df[1] == 'Taichung') & (df[3] == 'Big mile')]
            print(dfzzz)


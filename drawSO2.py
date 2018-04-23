# load and plot dataset
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot


# load dataset
def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')


series = read_csv('so2.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)  # , date_parser=parser)
# summarize first few rows

series1 = series[:20]
series2 = series[20:100]

print(series.head())

# line plot
series1.plot()
series2.plot()
pyplot.show()
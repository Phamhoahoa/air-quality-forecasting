import pandas as pd
import numpy as np 
import glob 
from matplotlib import pyplot as plt
from datetime import datetime

def vizualize_data(station):
    df = pd.read_csv(f'./data/data-train/input/{station}.csv')
    df['timestampStr'] = df['timestamp'].apply(lambda x:datetime.strptime(x, "%d/%m/%Y %H:%M"))
    list_fields = ['PM2.5','temperature', 'humidity']
    for field in list_fields:
        print('vizualize',field)
        temperatureDF = df[['timestampStr', field]]
        temperatureDF = temperatureDF.set_index('timestampStr')
        t0 = temperatureDF.index
        t1 = pd.date_range(pd.to_datetime('01/8/2020',dayfirst=True),pd.to_datetime('1/6/2021',dayfirst=True),freq='H')
        t2 = pd.date_range(pd.to_datetime('01/8/2020',dayfirst=True),pd.to_datetime('01/10/2020' ,dayfirst=True),freq='H')
        t3 = pd.date_range(pd.to_datetime('01/8/2020',dayfirst=True),pd.to_datetime('8/8/2020',dayfirst=True),freq='H')
        t = [t0, t1, t2, t3]

        fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(15,14))
        for i, t in enumerate(t):
            ax[i].plot(t,temperatureDF.loc[t,field])
        for i in range(len(ax)): ax[i].set_ylabel(f'{field}', fontsize=11)
        ax[3].set_xlabel('time', fontsize=14)
        plt.show()


pd.set_option('display.max_rows', None)
list_file = glob.glob('./data/data-train/input/*')
train = pd.DataFrame()
c = 0
for file_name in list_file:
    c += 1
    place = file_name.split('/')[4][:-4]
    print(place)
    vizualize_data(place)
    break
    

list_file = glob.glob('./data/data-train/input/*')
train = pd.DataFrame()
for file_name in list_file:
    place = file_name.split('/')[4][:-4]
    sub_train = pd.read_csv(file_name)
    sub_train['place'] = place
    train = pd.concat([train, sub_train])
train['timestampStr'] = train['timestamp'].apply(lambda x:datetime.strptime(x, "%d/%m/%Y %H:%M"))
display(train.sample(n = 5))

from pandas.plotting import autocorrelation_plot, lag_plot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
placeToShow = 'S0000264-FDS - Ton That Thuyet'
selectedLagPoints = [1,3,6,9,12,24,36,48,60]
maxLagDays = 7

originalSignal = temperatureDF[placeToShow]

plt.figure(figsize=(12,6))
gs = gridspec.GridSpec(2, len(selectedLagPoints))
axTopRow = plt.subplot(gs[0, :])
axBottomRow = []
for i in range(len(selectedLagPoints)):
    axBottomRow.append(plt.subplot(gs[1, i]))

# ## plot autocorr
allTimeLags = np.arange(1,maxLagDays*24)
autoCorr = [originalSignal.autocorr(lag=dt) for dt in allTimeLags]
axTopRow.plot(allTimeLags,autoCorr); 
axTopRow.set_title('Autocorrelation Plot of PM2.5', fontsize=18);
axTopRow.set_xlabel('time lag [hours]'); axTopRow.set_ylabel('correlation coefficient')
selectedAutoCorr = [originalSignal.autocorr(lag=dt) for dt in selectedLagPoints]
axTopRow.scatter(x=selectedLagPoints, y=selectedAutoCorr, s=50, c='r')

# ## plot scatter plot of selected points
for i in range(len(selectedLagPoints)):
    lag_plot(originalSignal, lag=selectedLagPoints[i], s=0.5, alpha=0.7, ax=axBottomRow[i])    
    if i >= 1:
        axBottomRow[i].set_yticks([],[])
plt.tight_layout()

# ### Các biểu đồ phân tán phía dưới tương ứng với các điểm màu đỏ đánh dấu ở biểu đồ auto-corelation 
# ### Biểu đồ đầu tiên, cho thấy mối tương quan giữa PM2.5 tại thời điểm t so với Pm2.5 tại giờ t+1 
# ### Chúng ta thấy răng PM2.5 không thay đổi nhiều như vậy trong một giờ và do đó chúng ta thấy mối tương quan cực kỳ cao giữa các gía trị PM2.5 đó
# ### Mối tương quan này giảm dần ở mức chênh lệch 12h sau đó, tương ứng với mức chuyển từ ngày sang đêm, và tiếp tục dao động với xu hướng đó khi các ngày trôi qua  

#%% zoom in and out on the autocorr plot
fig, ax = plt.subplots(nrows=4,ncols=1, figsize=(14,14))

timeLags = np.arange(1,25*24*30)
autoCorr = [originalSignal.autocorr(lag=dt) for dt in timeLags]
ax[0].plot(1.0/(24*30)*timeLags, autoCorr); ax[0].set_title('Autocorrelation Plot', fontsize=20);
ax[0].set_xlabel('time lag [months]'); ax[0].set_ylabel('correlation coeff', fontsize=12);

timeLags = np.arange(1,20*24*7)
autoCorr = [originalSignal.autocorr(lag=dt) for dt in timeLags]
ax[1].plot(1.0/(24*7)*timeLags, autoCorr);
ax[1].set_xlabel('time lag [weeks]'); ax[1].set_ylabel('correlation coeff', fontsize=12);

timeLags = np.arange(1,20*24)
autoCorr = [originalSignal.autocorr(lag=dt) for dt in timeLags]
ax[2].plot(1.0/24*timeLags, autoCorr);
ax[2].set_xlabel('time lag [days]'); ax[2].set_ylabel('correlation coeff', fontsize=12);

timeLags = np.arange(1,3*24)
autoCorr = [originalSignal.autocorr(lag=dt) for dt in timeLags]
ax[3].plot(timeLags, autoCorr);
ax[3].set_xlabel('time lag [hours]'); ax[3].set_ylabel('correlation coeff', fontsize=12);
windowSize = 5*24
# ### Rolling 
lowPassFilteredSignal = originalSignal.rolling(windowSize, center=True).mean()

t0 = temperatureDF.index
t1 = pd.date_range(pd.to_datetime('01/8/2020',dayfirst=True),pd.to_datetime('1/6/2021',dayfirst=True),freq='H')
t2 = pd.date_range(pd.to_datetime('01/8/2020',dayfirst=True),pd.to_datetime('01/10/2020' ,dayfirst=True),freq='H')
t3 = pd.date_range(pd.to_datetime('01/8/2020',dayfirst=True),pd.to_datetime('21/8/2020',dayfirst=True),freq='H')

fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(14,12))
ax[0].plot(t0,originalSignal,c='y')
ax[0].plot(t0,lowPassFilteredSignal,c='r')

ax[1].plot(t1,originalSignal[t1],c='y')
ax[1].plot(t1,lowPassFilteredSignal[t1],c='r')

ax[2].plot(t2,originalSignal[t2],c='y')
ax[2].plot(t2,lowPassFilteredSignal[t2],c='r')

ax[3].plot(t3,originalSignal[t3],c='y')
ax[3].plot(t3,lowPassFilteredSignal[t3],c='r')

ax[0].legend(['original signal','low pass filtered'], fontsize=18,
              loc='upper left',bbox_to_anchor=(0.02,1.4), ncol=len(placeToShow))
for i in range(len(ax)): ax[i].set_ylabel('PM2.5', fontsize=11)
ax[3].set_xlabel('time', fontsize=14);
!pip3 install seaborn
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('seaborn')
data = originalSignal
data = data.reset_index().rename(columns={'S0000264-FDS - Ton That Thuyet': 'PM2.5'})
data = data.fillna(method = 'bfill', axis=0).dropna()
print(data['PM2.5'].describe())
ax = sns.distplot(data['PM2.5'])
data['hour'] = [d.strftime('%H') for d in data.timestampStr]
data['hour'] = data['hour'].apply(lambda x:int(x))
display(data.head())
sample = data[:168]
ax = sample['hour'].plot()

# Here we see exactly what we would expect from hourly data for a week:
# a cycle between 0 and 23 that repeats 7 times
# Vizulize ncoding Cyclical Features
data['hour_sin'] = np.sin(2 * np.pi * data['hour']/23.0)
data['hour_cos'] = np.cos(2 * np.pi * data['hour']/23.0)
sample = data[0:168]
ax = sample['hour_sin'].plot()

"""
Created on Sat Oct 14 17:07:23 2017

Have a look for plots and poisson: http://docs.pymc.io/notebooks/GLM-poisson-regression.html
@author: Ajinkya
"""

import pandas as pd
import numpy as np


train=pd.read_csv("Data/train.csv")
test=pd.read_csv("Data/test.csv")

train["Date"] = pd.to_datetime(train["ID"], format='%Y%m%d%H')
test["Date"] = pd.to_datetime(test["ID"], format='%Y%m%d%H')


# Function for splitting Date into Day, Month and Year and adding other features

def process_date(data):
    data['Day'] = data['Date'].dt.dayofweek    
    data['Month'] = data['Date'].dt.month
    data['Summer']=0
    data.loc[(4<=data.Month) & (data.Month <=7),'Summer']=1
    data['Year'] = data['Date'].dt.year
    data['Hour'] = data['Date'].dt.hour
    data['Quarter'] = data['Date'].dt.quarter
    data['month_start'] = data['Date'].dt.is_month_start
    data['month_end'] = data['Date'].dt.is_month_end
    data["WEEKEND"] = data["Day"] > 4
    data.drop(['Date','ID'], axis=1, inplace=True)
    return data

train=process_date(train)
train.describe()

sub=test.copy()
sub.drop('Date', axis=1, inplace=True)

test=process_date(test)
test.drop('Count', axis=1, inplace=True)

# Plotting graphs

train.plot(x='Hour', y='Count',kind='scatter')
train.plot(x='Month', y='Count',kind='scatter')
train.plot(x='Year', y='Count',kind='scatter')

# Outliers spotted

outliers=train.loc[(train.Count>400) & (train.Year ==2011),]

outliers_index=train.loc[(train.Count>400) & (train.Year ==2011),].index.values.tolist()
train=train.drop(train.index[outliers_index])
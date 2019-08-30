# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:05:25 2019

@author: USER
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

#Read the data

df = pd.read_csv("FuelConsumptionCo2.csv")
#take a look at dataset
df.head()

#get some features that you want to use
cdf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

#plot

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel('Engine Size')
plt.ylabel('Emission')
plt.show()

#get training and test dataset
msk = np.random.rand(len(df))<0.8
train =cdf[msk]
test=cdf[~msk]

#train the data distribution

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='red')
plt.xlabel('Engine Size')
plt.ylabel('co2 emission')
plt.show()

#multiple linear regression

from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE' , 'CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x,y)

# coefficients

print('coefficients:',regr.coef_)

#prediction

y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x= np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y=np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares : %.2f" %np.mean((y_hat -y) **2))
#explained variance score : 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y)) 
 


























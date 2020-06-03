#  STEP0 import libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#  STEP1 import csv file
bikes=pd.read_csv('hour.csv')

# STEP2 prelim analysis and feature selection
bikes_prep=bikes.copy()
bikes_prep=bikes_prep.drop(['index','date','casual','registered'],axis=1)
 
# check the null values
nullvalues=bikes_prep.isnull().sum()

# simple visualization of the data using  pandas

bikes_prep.hist(rwidth=0.9)
plt.tight_layout()
 
# STEP3 visualizing the data
# visualizating the contineous feature with demand

plt.subplot(2,2,1)
plt.title("Temperature VS Demand")
plt.scatter(bikes_prep['temp'],bikes_prep['demand'],s=1,c='g')


plt.subplot(2,2,2)
plt.title("ATemp VS Demand")
plt.scatter(bikes_prep['atemp'],bikes_prep['demand'],s=1,c='m')

plt.subplot(2,2,3)
plt.title("Humidity VS Demand")
plt.scatter(bikes_prep['humidity'],bikes_prep['demand'],s=1,c='b')


plt.subplot(2,2,4)
plt.title("Windspeed VS Demand")
plt.scatter(bikes_prep['windspeed'],bikes_prep['demand'],s=1,c='y')
plt.tight_layout()

# visualize the categorical feature

colors=['g','b','y','m']
plt.subplot(3,3,1)
plt.title("Average demand per sesion")
cat_list=bikes_prep['season'].unique()
cat_average=bikes_prep.groupby('season').mean()['demand']
plt.bar(cat_list,cat_average,color=colors)


plt.subplot(3,3,2)
plt.title("Average demand per year")
cat_list=bikes_prep['year'].unique()
cat_average=bikes_prep.groupby('year').mean()['demand']
plt.bar(cat_list,cat_average,color=colors)


plt.subplot(3,3,3)
plt.title("Average demand per month")
cat_list=bikes_prep['month'].unique()
cat_average=bikes_prep.groupby('month').mean()['demand']
plt.bar(cat_list,cat_average,color=colors)


plt.subplot(3,3,4)
plt.title("Average demand per hour")
cat_list=bikes_prep['hour'].unique()
cat_average=bikes_prep.groupby('hour').mean()['demand']
plt.bar(cat_list,cat_average,color=colors)


plt.subplot(3,3,5)
plt.title("Average demand per holiday")
cat_list=bikes_prep['holiday'].unique()
cat_average=bikes_prep.groupby('holiday').mean()['demand']
plt.bar(cat_list,cat_average,color=colors)


plt.subplot(3,3,6)
plt.title("Average demand per weekday")
cat_list=bikes_prep['weekday'].unique()
cat_average=bikes_prep.groupby('weekday').mean()['demand']
plt.bar(cat_list,cat_average,color=colors)
 

plt.subplot(3,3,7)
plt.title("Average demand per working day")
cat_list=bikes_prep['workingday'].unique()
cat_average=bikes_prep.groupby('workingday').mean()['demand']
plt.bar(cat_list,cat_average,color=colors)

plt.subplot(3,3,8)
plt.title("Average demand per weather")
cat_list=bikes_prep['weather'].unique()
cat_average=bikes_prep.groupby('weather').mean()['demand']
plt.bar(cat_list,cat_average,color=colors)
plt.tight_layout()

# check the outliers

describevlues=bikes_prep['demand'].describe()
some_quntile=bikes_prep['demand'].quantile([0.05,0.1,0.15,0.9,0.95,0.99])


# STEP 4 check multiple coolinear regression

# linearity using correlation coefficient matrix using corr
correlation = bikes_prep[['temp','atemp','humidity','windspeed','demand']].corr()

# drop
bikes_prep=bikes_prep.drop(['weekday','workingday','year','atemp','windspeed'],axis=1)
 
# check the autocorrelation in demand using the aplot
df1=pd.to_numeric(bikes_prep['demand'],downcast='float')
plt.acorr(df1,maxlags=12)

# log normalise the feature demand
df1=bikes_prep['demand']
df2=np.log(df1)

plt.figure()
df1.hist(rwidth=0.9,bins=20)


plt.figure()
df2.hist(rwidth=0.9,bins=20)


# modify the actual data
bikes_prep['demand']=np.log(bikes_prep['demand'])

#autocorrelation in the demand column
t_1=bikes_prep['demand'].shift(+1).to_frame()
t_1.columns=['t-1']



t_2=bikes_prep['demand'].shift(+2).to_frame()
t_2.columns=['t-2']



t_3=bikes_prep['demand'].shift(+3).to_frame()
t_3.columns=['t-3']

#new dataframe
bikes_prep_lag=pd.concat([bikes_prep,t_1,t_2,t_3],  axis=1)
bikes_prep_lag=bikes_prep_lag.dropna()




# using get_dummies
data_type=bikes_prep_lag.dtypes

# converting into categorical variable

bikes_prep_lag['season']=bikes_prep_lag['season'].astype('category')
bikes_prep_lag['holiday']=bikes_prep_lag['holiday'].astype('category')
bikes_prep_lag['weather']=bikes_prep_lag['weather'].astype('category')
bikes_prep_lag['month']=bikes_prep_lag['month'].astype('category')
bikes_prep_lag['hour']=bikes_prep_lag['hour'].astype('category')

data_types=bikes_prep_lag.dtypes

bikes_prep_lag=pd.get_dummies(bikes_prep_lag,drop_first=True)

# split the data set
from sklearn.model_selection import train_test_split
Y=bikes_prep_lag[['demand']]
X=bikes_prep_lag.drop(['demand'],axis=1)

#create trainig set as 70%
tr_size=0.7 * len(X)

tr_size=int(tr_size)
x_train=X.values[0:tr_size]
x_test=X.values[tr_size:len(X)]

y_train=Y.values[0:tr_size]
y_test=Y.values[tr_size:len(Y)]



# Fit and score the model
from sklearn.linear_model import LinearRegression
std_reg=LinearRegression()
std_reg.fit(x_train,y_train)


r2_train=std_reg.score(x_train,y_train)
r2_test=std_reg.score(x_test,y_test)


# Create Y prediction

y_predict=std_reg.predict(x_test)
from sklearn.metrics import mean_squared_error
rmse=math.sqrt(mean_squared_error(y_test,y_predict))



# Calculate the RMSLE

y_test_e=[]
y_predict_e=[]
for i in range(0,len(y_test)):
    y_test_e.append(math.exp(y_test[i]))
    y_predict_e.append(math.exp(y_predict[i]))
    
    
#calculate the sum
log_sq_sum=0.0
for i in range(0,len(y_test_e)):
    log_a = math.log(y_test_e[i] + 1)
    log_p = math.log(y_predict_e[i] + 1)
    log_diff = (log_p-log_a)**2
    log_sq_sum = log_sq_sum + log_diff
    
    
    
rmsle=math.sqrt(log_sq_sum/len(y_test))
print(rmsle)
    
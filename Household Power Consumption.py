#Import Libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import median_absolute_error
from sklearn.decomposition import PCA
#read Data
data=pd.read_csv('Household_power_consumption.csv',index_col='datetime')
#print frist 5 data
print(data.head())
#print data columns
x=1
for col in data.columns:
    print(f'col {x} is :',col)
    x+=1 
#data corr  
sns.heatmap(data.corr())  
#data describe
print(data.describe())
#data information
print(data.info())
#show sum of null data
print(data.isnull().sum())
#show sum of duplicated
print(data.duplicated().sum())
#calculation of output power
data['power']=np.sqrt(np.square(data.iloc[:,0]+np.square(data.iloc[:,1])))
#show input X and output y
col=data.shape[1]
X=data.iloc[:,0:col-1]
y=data.iloc[:,col-1:col]
print(col)
print(X.shape)
print(y.shape)
#cleaning input X
ImputedModule = SimpleImputer(missing_values = np.nan, strategy ='mean')
ImputedX = ImputedModule.fit(X)
X = ImputedX.transform(X)
print('X Data is \n' , X[:10])
print('y Data is \n' , y[:10])  
#scaling input X  
scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)
print('X \n' , X[:10])
#data spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)
#Applying Linear Regression Model
LinearRegressionModel = LinearRegression(fit_intercept=True, normalize=True,copy_X=True,n_jobs=-1)
LinearRegressionModel.fit(X_train, y_train)
#Calculating Details
print('Linear Regression Train Score is : ' , LinearRegressionModel.score(X_train, y_train))
print('Linear Regression Test Score is : ' , LinearRegressionModel.score(X_test, y_test))
#Calculating Prediction
y_pred = LinearRegressionModel.predict(X_test)
print('Predicted Value for Linear Regression is : ' , y_pred[:10])    
#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
print('Mean Absolute Error Value is : ', MAEValue)
#Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') 
print('Mean Squared Error Value is : ', MSEValue)
#Calculating Median Absolute Error
MdSEValue = median_absolute_error(y_test, y_pred)
print('Median Absolute Error Value is : ', MdSEValue )
#display y_train ,y_test_train
plt.figure()
plt.plot(y_train[:200],label='y_train')
plt.plot(LinearRegressionModel.predict(X_train[:200]),label='y_train_pred')
plt.title('y_pred and y_actual')
plt.legend()
#display y_test ,y_test_pred
plt.figure()
plt.plot(y_test[:200],label='y_test')
plt.plot(LinearRegressionModel.predict(X_test[:200]),label='y_test_pred')
plt.title('y_pred and y_actual')
plt.legend()

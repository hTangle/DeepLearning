import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

print("hello python")
data = pd.read_csv('D:\\program\\python\\DeepLearning\\CCPP\\Folds5x2_pp.csv')
X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(y.head())
print(data.head())
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
print(linreg.intercept_)
print(linreg.coef_)
#模型拟合测试集
y_pred = linreg.predict(X_test)
from sklearn import metrics
# 用scikit-learn计算MSE
print("MSE:",metrics.mean_squared_error(y_test, y_pred))
# 用scikit-learn计算RMSE
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
fig, ax = plt.subplots()
predicted = cross_val_predict(linreg, X, y, cv=10)

ax.scatter(y, predicted)

ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


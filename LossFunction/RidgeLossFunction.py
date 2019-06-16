import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV

data = pd.read_csv('D:\\program\\python\\DeepLearning\\CCPP\\Folds5x2_pp.csv')
X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
ridgecv = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100])
ridgecv.fit(X_train, y_train)
print(ridgecv.alpha_)
print(ridgecv.intercept_)
print(ridgecv.coef_)

y_pred = ridgecv.predict(X_test)
from sklearn import metrics
# 用scikit-learn计算MSE
print("MSE:",metrics.mean_squared_error(y_test, y_pred))
# 用scikit-learn计算RMSE
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
fig, ax = plt.subplots()
predicted = cross_val_predict(ridgecv, X, y, cv=10)

ax.scatter(y, predicted)

ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
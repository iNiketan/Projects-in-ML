import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/media/niketan/cartush/practice/storedata')

# importing dataset and basic things
StoreData = pd.read_csv("StoreData.csv")
StoreData.describe()
print(StoreData.info())
print(StoreData.head(2))

X = StoreData.iloc[:, 5:].values
y1 = StoreData.iloc[:, 3:4].values
y2 = StoreData.iloc[:, 4:5].values

# data priprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lablencoder_X = LabelEncoder()
X[:, 4] = lablencoder_X.fit_transform(X[:, 4]) # converting categorical to numbers
onehotencoder = OneHotEncoder(categorical_features = [4])
X = onehotencoder.fit_transform(X).toarray()

# escaping the dummy variable trap
X = X[:, 1:]

# finding the corelation between sales and promotion of data
from sklearn.model_selection import train_test_split
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=.2, random_state=0)
X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size=.2, random_state=0)


# fitting linear regressior model
from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor2 = LinearRegression()

regressor1.fit(X_train, y1_train)
regressor2.fit(X_train, y2_train)

# predicting dataset from fitted
y_pred1 = regressor1.predict(X_test)
y_pred2 = regressor2.predict(X_test)

# Backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones([2080, 1]).astype(int), values= X, axis=1) # y = b + x1b1 + x2b2..
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
regressor_ols = sm.OLS(endog= y1, exog= X_opt).fit()
regressor_ols.summary()
#2
X_opt = X[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]]
regressor_ols = sm.OLS(endog= y1, exog= X_opt).fit()
regressor_ols.summary()
#3
X_opt = X[:, [0, 1, 3, 4, 5, 6, 7, 8, 9]]
regressor_ols = sm.OLS(endog= y1, exog= X_opt).fit()
regressor_ols.summary()
#4
X_opt = X[:, [0, 1, 3, 4, 5, 7, 8, 9]]
regressor_ols = sm.OLS(endog= y1, exog= X_opt).fit()
regressor_ols.summary()
#5
X_opt = X[:, [0, 1, 3, 4, 5, 7, 8, 9]]
regressor_ols = sm.OLS(endog= y1, exog= X_opt).fit()
regressor_ols.summary()
#6
X_opt = X[:, [0, 3, 4, 5, 7, 8, 9]]
regressor_ols = sm.OLS(endog= y1, exog= X_opt).fit()
regressor_ols.summary()
#7
X_opt = X[:, [0, 3, 4, 7, 8, 9]]
regressor_ols = sm.OLS(endog= y1, exog= X_opt).fit()
regressor_ols.summary()
#8
X_opt = X[:, [0, 3, 7, 8, 9]]
regressor_ols = sm.OLS(endog= y1, exog= X_opt).fit()
regressor_ols.summary()
#9
X_opt = X[:, [0, 7, 8, 9]]
regressor_ols = sm.OLS(endog= y1, exog= X_opt).fit()
regressor_ols.summary()

X_opt_train, X_opt_test, y1_train, y1_test = train_test_split(X, y1, test_size=.2, random_state=0)
regressor1.fit(X_opt_train, y1_train)
y_pred12 = regressor1.predict(X_opt_test)

# for y2
X_opt2 = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
regressor_ols = sm.OLS(endog= y2, exog= X_opt2).fit()
regressor_ols.summary()
X_opt2 = X[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]]
regressor_ols = sm.OLS(endog= y2, exog= X_opt2).fit()
regressor_ols.summary()
X_opt2 = X[:, [0, 1, 2, 4, 6, 7, 8, 9, 10]]
regressor_ols = sm.OLS(endog= y2, exog= X_opt2).fit()
regressor_ols.summary()
X_opt2 = X[:, [0, 1, 4, 6, 7, 8, 9, 10]]
regressor_ols = sm.OLS(endog= y2, exog= X_opt2).fit()
regressor_ols.summary()
X_opt2 = X[:, [0, 1, 6, 7, 8, 9, 10]]
regressor_ols = sm.OLS(endog= y2, exog= X_opt2).fit()
regressor_ols.summary()
X_opt2 = X[:, [0, 1, 7, 8, 9, 10]]
regressor_ols = sm.OLS(endog= y2, exog= X_opt2).fit()
regressor_ols.summary()

X_opt2 = X[:, [0, 1, 7, 8, 10]]
regressor_ols = sm.OLS(endog= y2, exog= X_opt2).fit()
regressor_ols.summary()

#############################################################################
# exploring data with visualisation
y1 = y1.reshape(-1)
y2 = y2.reshape(-1)
box_plot_arr1 =[y1, y2]
plt.boxplot(box_plot_arr1, patch_artist=True, labels=['sales store1', 'sales store2'])
plt.show()

plt.scatter(X_train[:, 6:7], y1_train, color= "red")
plt.title('P1 Price vs Sales (Training set)')
plt.xlabel('P1 Price')
plt.ylabel('Sales')
plt.show()

plt.scatter(X_test[:, 6:7], y1_test, color= "red")
plt.title('P1 Price vs Sales (Testing set)')
plt.xlabel('P1 Price')
plt.ylabel('Sales')
plt.show()

plt.scatter(X_train[:, 7:8], y2_train, color= "green")
plt.title('P2 Price vs Sales (Training set)')
plt.xlabel('P2 Price')
plt.ylabel('Sales')
plt.show()

plt.scatter(X_test[:, 7:8], y2_test, color= "green")
plt.title('P2 Price vs Sales (Testing set)')
plt.xlabel('P2 Price')
plt.ylabel('Sales')
plt.show()

plt.hist2d(StoreData['p1prom'], StoreData['p1sales'])
plt.show()

plt.scatter(StoreData['p1prom'], StoreData['p1sales'], color= 'red')
plt.title('Promotion vs Sales')
plt.xlabel('Product1 Promotion')
plt.ylabel('Sales Product1')
plt.show()

plt.scatter(StoreData['p2prom'], StoreData['p2sales'], color= 'green')
plt.title('Promotion vs Sales')
plt.xlabel('Product2 Promotion')
plt.ylabel('Sales Product2')
plt.show()

plt.scatter(y1_test, y_pred1, color= 'red')
plt.title('Sales vs Sales_predictected')
plt.xlabel('Sales')
plt.ylabel('Sales Predicted')
plt.show()

plt.scatter(y2_test, y_pred2, color= 'green')
plt.title('Sales vs Sales_predictected(product 2)')
plt.xlabel('Sales')
plt.ylabel('Sales Predicted')
plt.show()

#############################################

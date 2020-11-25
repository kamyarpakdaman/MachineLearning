# In this program, we investigate the decline of the population of honeybees and how the trends of the past predict the future for the honeybees.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("honey_production.csv")
# print(df.head())

# Computing the mean total production of honey per year

prod_per_year = df.groupby('year')['totalprod'].mean().reset_index()

X = prod_per_year['year'].values.reshape(-1, 1)
y = prod_per_year['totalprod']

# Drawing a scatter showing the trend of the mean total production over years. We want to initially look for potential trends.

plt.figure()

plt.scatter(X, y, color = 'darkslateblue')
plt.title('Mean Total Production in the Past Years')
plt.grid(axis = 'y', alpha = 0.5)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks(list(range(3600000, 5500000, 300000)))
ax.set_yticklabels(['{:,}'.format(int(number)) for number in list(range(3600000, 5500000, 300000))])

# plt.show()

# Now we create a linear regression model to make some predictions.

linreg = LinearRegression()
linreg.fit(X, y)
print('Slope is: ', linreg.coef_[0], '\nIntercept is: ', linreg.intercept_)

y_predict = linreg.predict(X)

plt.plot(X, y_predict, color = 'springgreen')

plt.show()

# Now we predict years to come, the data for which are unknown, from 2013 to 2050.

X_future = np.array(range(2013, 2051))
X_future = X_future.reshape(-1, 1)
future_predict = linreg.predict(X_future)


plt.close('all')
plt.figure()

plt.plot(X_future, future_predict, color = 'darkviolet')
plt.title('Predicted Mean Total Production in Upcoming Years')
plt.grid(axis = 'y', alpha = 0.5)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks(list(range(100000, 3500000, 500000)))
ax.set_yticklabels(['{:,}'.format(int(number)) for number in list(range(100000, 3500000, 500000))])

plt.show()

print('\nThanks for reviewing')

# Thanks for reviewing

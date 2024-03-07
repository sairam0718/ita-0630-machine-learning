from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=100, n_features=1, noise=0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
print('Predictions:', predictions)

import matplotlib.pyplot as plt

plt.scatter(X_train, y_train, color='blue', label='Training Data')

plt.plot(X_test, predictions, color='red', linewidth=2, label='Regression Line')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()

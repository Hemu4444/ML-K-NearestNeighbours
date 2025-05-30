import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate random dataset
np.random.seed(0)
feature = np.random.rand(100, 1)  # More descriptive name for x
target = 2 + 3 * feature + np.random.rand(100, 1)  # More descriptive name for y

# Model initialization
regression_model = LinearRegression()

# Fit the data (train the model)
regression_model.fit(feature, target)

# Predict
target_predicted = regression_model.predict(feature)  # More descriptive name

# Model evaluation
rmse = np.sqrt(mean_squared_error(target, target_predicted))
r2 = r2_score(target, target_predicted)

# Printing values
print('Slope:', regression_model.coef_[0][0])
print('Intercept:', regression_model.intercept_[0])
print('Root mean squared error:', rmse)
print('R2 score:', r2)

# Plotting values
# Data points
plt.scatter(feature, target, s=10, label='Data Points')  # Added label
plt.xlabel('Feature (x)')  # More descriptive x-axis label
plt.ylabel('Target (y)')  # More descriptive y-axis label

# Predicted values
plt.plot(feature, target_predicted, color='r', label='Linear Regression')  # Added label
plt.legend()  # Added a legend
plt.title("Linear Regression Example")  # Added a title
plt.show()

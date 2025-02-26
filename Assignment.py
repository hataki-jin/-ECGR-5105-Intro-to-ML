import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1.Load the dataset D3(x1, x2, x3, y)
data = pd.read_csv('D3.csv')
x1 = data.iloc[:, 0].values
x2 = data.iloc[:, 1].values
x3 = data.iloc[:, 2].values
y = data.iloc[:, 3].values


# 2.Normalize the data
def normalize(data):
    return (data - np.mean(data)) / np.std(data)


x1 = normalize(x1)
x2 = normalize(x2)
x3 = normalize(x3)
y = normalize(y)

# 3.Set learning rate and iterations
learning_rate = 0.01
iterations = 1000


# Problem 1
# Gradient Descent Function
# def gradient_descent(x, y, learning_rate, iterations):
#     theta = 0
#     m = len(x)
#     cost_history = []
#
#     for _ in range(iterations):
#         prediction = theta * x
#         error = prediction - y
#         cost = (1 / (2 * m)) * np.sum(error ** 2)
#         cost_history.append(cost)
#         gradient = (1 / m) * np.sum(error * x)
#         theta = theta - learning_rate * gradient
#
#     return theta, cost_history
#
#
# # 4.Train for each explanatory variable
# theta1, cost_history1 = gradient_descent(x1, y, learning_rate, iterations)
# theta2, cost_history2 = gradient_descent(x2, y, learning_rate, iterations)
# theta3, cost_history3 = gradient_descent(x3, y, learning_rate, iterations)
#
# # 5.Report the linear models
# print(f"Linear model for x1: y = {theta1:.4f} * x1")
# print(f"Linear model for x2: y = {theta2:.4f} * x2")
# print(f"Linear model for x3: y = {theta3:.4f} * x3")
#
#
# # 6.Plot the final regression model and loss over iterations
# def plot_figure(x, x_name, y, theta, cost_history):
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.scatter(x, y, color='blue', label='Data points')
#     plt.plot(x, theta * x, color='red', label='Regression line')
#     plt.title('Regression Line for ' + x_name)
#     plt.xlabel(x_name)
#     plt.ylabel('y')
#     plt.legend()
#
#     # Loss over iterations
#     plt.subplot(1, 2, 2)
#     plt.plot(range(iterations), cost_history, color='green')
#     plt.title('Loss over Iterations for ' + x_name)
#     plt.xlabel('Iterations')
#     plt.ylabel('Loss')
#
#     plt.tight_layout()
#     plt.show()
#
#
# plot_figure(x1, 'x1', y, theta1, cost_history1)
# plot_figure(x2, 'x2', y, theta2, cost_history2)
# plot_figure(x3, 'x3', y, theta3, cost_history3)


# Problem 2
# Gradient Descent for All three Variables
def gradient_descent_multi(X, y, learning_rate, iterations):
    theta = np.zeros(X.shape[1])
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        prediction = X.dot(theta)
        error = prediction - y
        cost = (1 / (2 * m)) * np.sum(error ** 2)
        cost_history.append(cost)
        gradient = (1 / m) * X.T.dot(error)
        theta = theta - learning_rate * gradient

    return theta, cost_history


# Prepare data
X = np.column_stack((x1, x2, x3))
X = np.c_[np.ones(len(X)), X]  # Add bias term

# Train for all three variables
theta, cost_history = gradient_descent_multi(X, y, learning_rate, iterations)

# Report the final linear model
print(f"Final linear model: y = {theta[0]:.4f} + {theta[1]:.4f} * x1 + {theta[2]:.4f} * x2 + {theta[3]:.4f} * x3")

# Plot loss over iterations
plt.plot(range(iterations), cost_history, color='blue')
plt.title('Loss over Iterations (All Three Variables)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()


# Predict new values (get value of y)
def predict(X, theta):
    return X.dot(theta)


new_values = np.array([[1, 1, 1], [2, 0, 4], [3, 2, 1]])
new_values = np.c_[np.ones(new_values.shape[0]), new_values]  # Add bias term
predictions = predict(new_values, theta)

print("Predictions for new values:")
for i, pred in enumerate(predictions):
    print(f"({new_values[i, 1]}, {new_values[i, 2]}, {new_values[i, 3]}): {pred:.4f}")

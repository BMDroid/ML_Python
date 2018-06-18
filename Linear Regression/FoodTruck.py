import numpy as np
import matplotlib.pyplot as plt

# Store the training set in the vectors
data = np.loadtxt('food_truck_data.txt', delimiter=',')
x = data[:, 0]
y = data[:, 1]
m = len(y) # the size of the training set

# Plot the trainning set
fig, ax = plt.subplots()
ax.scatter(x, y, marker="x", c="red")
plt.title("Food Truck Dataset", fontsize=16)
plt.xlabel("City Population in 10,000s", fontsize=14)
plt.ylabel("Food Truck Profit in 10,000s", fontsize=14)
plt.axis([4, 25, -5, 25])

# Create the hypothesis function
theta = np.zeros(2)
grad = np.zeros(2)
X = np.ones(shape=(len(x), 2)) # add x0
X[:, 1] = x

# Cosr Function J
def cost(theta, X, y):
  predictions = X@theta
  errors = predictions - y
  squared_errors = np.square(errors)
  return np.sum(squared_errors) / (2 * m)

# The initial cost of the hypothesis when theta = [0; 0]
initial_cost = cost(theta, X, y)
# print(f"The initial cost is: {initial_cost}.")

# Gradient
def gradient(theta, X, y):
  predictions = X@theta
  errors = predictions - y
  return X.transpose()@errors / len(y)

# Gradient Descent
def grad_descent(theta, X, y, alpha, num_iters):
  for i in range(1, num_iters):
    theta -= alpha * gradient(theta, X, y)
  return theta

theta = grad_descent(theta, X, y, 0.02, 600)
predictions = X@theta
ax.plot(X[:, 1], predictions, linewidth=2)
plt.show()

# To check the gradient decent function with seeing the cost function's value of each iterations
def grad_descent_cost(X, y, alpha, num_iters):
  cost_history = np.zeros(num_iters)
  num_features = X.shape[1] # column numbers
  theta = np.zeros(num_features) #initialize the theta
  for i in range(num_iters): # i start from 0
    theta -= alpha * gradient(theta, X, y)
    cost_history[i] = cost(theta, X, y)
  return theta, cost_history # the function will return in tuple

plt.figure()
num_iters = 1200
learning_rates = [0.01, 0.015, 0.02]

# Plot the cost functions value of different learning rates
for l in learning_rates:
  _, cost_history = grad_descent_cost(X, y, l, num_iters)
  plt.plot(cost_history, linewidth=2)

plt.title("Gradient descent with different learning rates", fontsize=16)
plt.xlabel("number of iterations", fontsize=14)
plt.ylabel("cost", fontsize=14)
plt.legend(list(map(str, learning_rates)))
plt.axis([0, num_iters, 4, 6])
plt.grid()
plt.show()

# Prediction
theta, _ = grad_descent_cost(X, y, 0.02, 600)
test_example = np.array([1, 7]) # pick a city with 70,000 population as a test example
prediction = test_example@theta
print(f"For the city with a population of 70,000, we predict a profit of $ {prediction * 10000}.")
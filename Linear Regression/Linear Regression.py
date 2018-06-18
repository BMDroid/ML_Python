import numpy as np

def cost_grad(theta, X, y):
  prediction = X@theta
  error = prediction - y
  squared_error = np.square(error)
  J = np.sum(squared_error) / (2 * len(y))
  grad = X.transpose()@error / len(y)
  return J, grad

def cost_grad_regularized(theta, X, y, lambdda):
  prediction = X@theta
  error = prediction - y
  squared_error = np.square(error)
  J = np.sum(squared_error) / (2 * len(y)) + np.sum(np.square(theta[1, :])) * lambdda / (2 * len(y))
  theta_temp = theta
  theta_temp[0, :] = 0
  grad = X.transpose()@error / len(y) + theta_temp * lambdda / len(y)
  return J, grad

def grad_descent(X, y, lambdda, alpha, num_iters):
  num_features = X.shape[1]
  theta = np.zeros(num_features)
  for i in range(num_iters):
    _, grad = cost_grad_regularized(theta, X, y,lambdda)
    theta -= alpha * grad
  return theta

def grad_descent_cost(X, y, lambdda, alpha, num_iters):
  cost_history = np.zeros(num_iters)
  num_features = X.shape[1]
  theta = np.zeros(num_features)
  for i in range(num_iters):
    _, grad = cost_grad_regularized(theta, X, y, lambdda)
    theta -= alpha * grad
    cost_history[i], _ = cost_grad_regularized(theta, X, y, lambdda)
  return theta, cost_history

def prediction(theta, X):
  return X@theta
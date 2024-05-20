import numpy as np

def quadratic_example_1(x):
    Q = np.array([[2, 0], [0, 2]])
    b = np.array([0, 0])
    f = 0.5 * x.T @ Q @ x + b.T @ x
    grad = Q @ x + b
    hess = Q
    return f, grad, hess

def quadratic_example_2(x):
    Q = np.array([[3, 0], [0, 1]])
    b = np.array([0, 0])
    f = 0.5 * x.T @ Q @ x + b.T @ x
    grad = Q @ x + b
    hess = Q
    return f, grad, hess

def quadratic_example_3(x):
    Q = np.array([[1, 0], [0, 4]])
    b = np.array([0, 0])
    f = 0.5 * x.T @ Q @ x + b.T @ x
    grad = Q @ x + b
    hess = Q
    return f, grad, hess

def rosenbrock(x):
    f = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    grad = np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2), 200 * (x[1] - x[0]**2)])
    hess = np.array([[2 - 400 * x[1] + 1200 * x[0]**2, -400 * x[0]], [-400 * x[0], 200]])
    return f, grad, hess

def linear_function(x):
    A = np.array([1, 2])
    b = 1
    f = A @ x - b
    grad = A
    hess = np.zeros((2, 2))
    return f, grad, hess

def boyds_function(x):
    f = np.exp(x[0] + 3 * x[1]) + np.exp(x[0] - 3 * x[1]) + np.exp(-x[0])
    grad = np.array([np.exp(x[0] + 3 * x[1]) + np.exp(x[0] - 3 * x[1]) - np.exp(-x[0]), 3 * np.exp(x[0] + 3 * x[1]) - 3 * np.exp(x[0] - 3 * x[1])])
    hess = np.array([[np.exp(x[0] + 3 * x[1]) + np.exp(x[0] - 3 * x[1]) + np.exp(-x[0]), 3 * np.exp(x[0] + 3 * x[1]) - 3 * np.exp(x[0] - 3 * x[1])],
                     [3 * np.exp(x[0] + 3 * x[1]) - 3 * np.exp(x[0] - 3 * x[1]), 9 * np.exp(x[0] + 3 * x[1]) + 9 * np.exp(x[0] - 3 * x[1])]])
    return f, grad, hess

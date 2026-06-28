import numpy as np
import matplotlib.pyplot as plt

# Gaussian kernel
def gaussian_kernel(x, xi, tau):
    return np.exp(-np.sum((x - xi) ** 2) / (2 * tau ** 2))

# Locally Weighted Regression
def locally_weighted_regression(x, X, y, tau):
    W = np.diag([gaussian_kernel(x, xi, tau) for xi in X])
    theta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
    return x @ theta

# Generate sample dataset
np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 100)
y = np.sin(X) + 0.1 * np.random.randn(100)

# Add bias term
X_bias = np.c_[np.ones(len(X)), X]

# Test points
x_test = np.linspace(0, 2 * np.pi, 200)
x_test_bias = np.c_[np.ones(len(x_test)), x_test]

# Bandwidth parameter
tau = 0.5

# Predict values
y_pred = np.array([locally_weighted_regression(x, X_bias, y, tau) for x in x_test_bias])

# Plot the results
plt.scatter(X, y, label="Training Data")
plt.plot(x_test, y_pred, label="LWR Fit")
plt.plot(X, np.sin(X), "--", label="True Function")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Locally Weighted Regression")
plt.legend()
plt.show()

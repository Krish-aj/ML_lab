import numpy as np
import matplotlib.pyplot as plt

# Gaussian kernel function
def gaussian_kernel(x, xi, tau):
    return np.exp(-np.sum((x - xi) ** 2) / (2 * tau ** 2))

# Locally Weighted Regression function
def locally_weighted_regression(x, X, y, tau):
    m = X.shape[0]
    weights = np.array([gaussian_kernel(x, X[i], tau) for i in range(m)])
    W = np.diag(weights)
    
    X_transpose_W = X.T @ W
    theta = np.linalg.inv(X_transpose_W @ X) @ X_transpose_W @ y
    
    return x @ theta

# Generate sample dataset
np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 100)
y = np.sin(X) + 0.1 * np.random.randn(100)

# Add bias term
X_bias = np.c_[np.ones(X.shape), X]

# Test points
x_test = np.linspace(0, 2 * np.pi, 200)
x_test_bias = np.c_[np.ones(x_test.shape), x_test]

# Bandwidth parameter
tau = 0.5

# Predict values using LWR
y_pred = np.array([locally_weighted_regression(x, X_bias, y, tau) for x in x_test_bias])

# Plot the results
plt.scatter(X, y, color='blue', label='Training Data')
plt.plot(x_test, y_pred, color='red', label='LWR Fit')
plt.plot(X, np.sin(X), color='green', linestyle='dashed', label='True Function')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Locally Weighted Regression')
plt.legend()
plt.show()

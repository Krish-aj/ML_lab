# Naive Bayes Classification Demonstration
# BCSL606 Machine Learning Lab

import numpy as np
import matplotlib.pyplot as plt
import random

# Generate training data
np.random.seed(0)

class1 = np.random.normal(0.3, 0.1, 50)
class2 = np.random.normal(0.8, 0.1, 50)

# Compute mean and variance
mean1 = np.mean(class1)
var1 = np.var(class1)

mean2 = np.mean(class2)
var2 = np.var(class2)

print("Mean Class1:", mean1)
print("Variance Class1:", var1)
print("Mean Class2:", mean2)
print("Variance Class2:", var2)


# Gaussian probability function
def gaussian(x, mean, var):
    var = var + 1e-6
    exponent = np.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / np.sqrt(2 * np.pi * var)) * exponent


# Prediction function
def predict(x):
    p1 = gaussian(x, mean1, var1)
    p2 = gaussian(x, mean2, var2)

    if p1 > p2:
        return "Class1"
    else:
        return "Class2"


# Generate test points
test_points = np.random.rand(100)

print("\nBCSL606 Machine Learning Lab\n")

results = []

for i, val in enumerate(test_points, start=1):
    label = predict(val)
    results.append(label)

    print(f"Point x{i} (value: {round(val,4)}) is classified as {label}")

print("\nClassification complete.")

# Plot results
plt.figure(figsize=(8,5))

for i, val in enumerate(test_points):
    if results[i] == "Class1":
        plt.scatter(val, 1, marker='o')
    else:
        plt.scatter(val, 2, marker='x')

plt.yticks([1,2], ["Class1","Class2"])
plt.xlabel("Value")
plt.ylabel("Class")
plt.title("Naive Bayes Classification Result")
plt.grid(True)

plt.show()

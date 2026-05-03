Program 8: 
AIM: Develop a program to demonstrate the working of the decision tree algorithm. Use Breast Cancer 
Data set for building the decision tree and apply this knowledge to classify a new sample. 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# Step 1: Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data        # Features
y = data.target      # Target labels

# Step 2: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train the Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 4: Make predictions on test data
y_pred = clf.predict(X_test)

# Step 5: Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 6: Predict class for a new sample
new_sample = X_test[0].reshape(1, -1)  # Reshape for single input
prediction = clf.predict(new_sample)

# Step 7: Convert prediction to class label
prediction_class = "Benign" if prediction == 1 else "Malignant"
print(f"Predicted Class for the new sample: {prediction_class}")

# Step 8: Visualize the Decision Tree
plt.figure(figsize=(12, 8))
tree.plot_tree(
    clf,
    filled=True,
    feature_names=data.feature_names.tolist(),
    class_names=data.target_names.tolist()
)
plt.title("Decision Tree - Breast Cancer Dataset")
plt.show()

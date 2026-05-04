Program 8: 
AIM: Develop a program to demonstrate the working of the decision tree algorithm. Use Breast Cancer 
Data set for building the decision tree and apply this knowledge to classify a new sample. 

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test)*100:.2f}%")

pred = model.predict([X_test[0]])
print("Prediction:", "Benign" if pred==1 else "Malignant")

plt.figure(figsize=(10,6))
plot_tree(model, filled=True)
plt.show()st Cancer Dataset")
plt.show()

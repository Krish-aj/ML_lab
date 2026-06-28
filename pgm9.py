import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
X, y = fetch_olivetti_faces( return_X_y=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train & Predict
model = GaussianNB().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%')
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Cross-validation
print(f'\nCross-validation accuracy: {cross_val_score(model, X, y).mean()*100:.2f}%')

# Visualization
plt.figure(figsize=(12, 8))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(X_test[i].reshape(64, 64), cmap="gray")
plt.show()

 PROGRAM 10
AIM

To develop a Python program to implement K-Means Clustering using the Wisconsin Breast Cancer dataset and to visualize the clustering results using PCA.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

# Step 1: Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Step 4: Evaluate clustering (comparison with true labels)
print("Confusion Matrix:")
print(confusion_matrix(y, y_kmeans))

print("\nClassification Report:")
print(classification_report(y, y_kmeans))

# Step 5: Reduce dimensions using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 6: Create DataFrame for visualization
df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df['Cluster'] = y_kmeans
df['True Label'] = y

# Step 7: Plot clustered data
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, x='PC1', y='PC2',
    hue='Cluster', palette='Set1',
    s=100, edgecolor='black', alpha=0.7
)
plt.title('K-Means Clustering of Breast Cancer Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title="Cluster")
plt.show()

# Step 8: Plot true labels
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, x='PC1', y='PC2',
    hue='True Label', palette='coolwarm',
    s=100, edgecolor='black', alpha=0.7
)
plt.title('True Labels of Breast Cancer Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title="True Label")
plt.show()

# Step 9: Plot clusters with centroids
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, x='PC1', y='PC2',
    hue='Cluster', palette='Set1',
    s=100, edgecolor='black', alpha=0.7
)

# Transform centroids to PCA space
centers = pca.transform(kmeans.cluster_centers_)

plt.scatter(centers[:, 0], centers[:, 1],
            s=200, c='red', marker='X', label='Centroids')

plt.title('K-Means Clustering with Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title="Cluster")
plt.show()

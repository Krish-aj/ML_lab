 PROGRAM 10
AIM

To develop a Python program to implement K-Means Clustering using the Wisconsin Breast Cancer dataset and to visualize the clustering results using PCA.

import matplotlib.pyplot as plt, seaborn as sns, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)

k = KMeans(2, random_state=42).fit(X)
y_k = k.labels_

print(confusion_matrix(y, y_k))
print(classification_report(y, y_k))

p = PCA(2)
df = pd.DataFrame(p.fit_transform(X), columns=['PC1','PC2'])
df['C'], df['T'] = y_k, y

sns.scatterplot(data=df, x='PC1', y='PC2', hue='C'); plt.show()
sns.scatterplot(data=df, x='PC1', y='PC2', hue='T'); plt.show()

c = p.transform(k.cluster_centers_)
sns.scatterplot(data=df, x='PC1', y='PC2', hue='C')
plt.scatter(c[:,0], c[:,1], c='red', marker='X'); plt.show()
plt.show()

import matplotlib.pyplot as plt, seaborn as sns, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)

k = KMeans(2, random_state=42)
y_k = k.fit_predict(X)

print(confusion_matrix(y, y_k))
print(classification_report(y, y_k))

p = PCA(2)
df = pd.DataFrame(p.fit_transform(X), columns=["PC1", "PC2"])
df["Cluster"] = y_k
df["Target"] = y

sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cluster")
plt.title("K-Means")
plt.show()
sns.scatterplot(data=df, x="PC1", y="PC2", hue="Target")
plt.title("Cluster")
plt.show()

c = p.transform(k.cluster_centers_)
sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cluster")
plt.scatter(c[:,0], c[:,1], c="red", marker="X")
plt.title("Centroids")
plt.show()

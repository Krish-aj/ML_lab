import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

file_path = "iris.csv"
df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns)

X = df.iloc[:, 0:4].values    
y = df.iloc[:, 4].values      

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    data=X_pca,
    columns=['Principal Component 1', 'Principal Component 2']
)
pca_df['Species'] = y

plt.figure(figsize=(8, 6))

for species in pca_df['Species'].unique():
    plt.scatter(
        pca_df[pca_df['Species'] == species]['Principal Component 1'],
        pca_df[pca_df['Species'] == species]['Principal Component 2'],
        label=species
    )

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset (4 Features → 2 Components)')
plt.legend()
plt.grid(True)
plt.show()

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Explained:", sum(pca.explained_variance_ratio_))

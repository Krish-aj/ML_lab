import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
file_path = "housing.csv"  
df = pd.read_csv(file_path)
print("Dataset Shape:", df.shape)
print("\nColumn Names:\n", df.columns)
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
print("\nCorrelation Matrix:\n", correlation_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Matrix Heatmap - California Housing Dataset")
plt.show()
sns.pairplot(df, diag_kind="kde")
plt.suptitle("Pair Plot of California Housing Features", y=1.02)
plt.show()

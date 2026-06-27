import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.datasets import fetch_california_housing

# Load California Housing dataset
df = fetch_california_housing(as_frame=True).frame

# Select numerical columns
num_cols = df.select_dtypes("number").columns

# Histograms (matrix layout)
df[num_cols].hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()

# Box plots (matrix layout)
cols = 3
rows = math.ceil(len(num_cols) / cols)

plt.figure(figsize=(12, 4 * rows))

for i, col in enumerate(num_cols):
    plt.subplot(rows, cols, i + 1)
    sns.boxplot(x=df[col])
    plt.title(col)

plt.tight_layout()
plt.show()

# Outlier count using IQR
for col in num_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1

    outliers = (
        (df[col] < Q1 - 1.5 * IQR)
        | (df[col] > Q3 + 1.5 * IQR)
    ).sum()

    print(f"{col}: {outliers} outliers")

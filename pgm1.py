# Step 1: Import required libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load dataset from current directory
file_path = "housing.csv"
df = pd.read_csv(file_path)

# Display basic dataset information
print("Dataset Shape:", df.shape)
print("\nColumn Names:\n", df.columns)

# Step 3: Select numerical features
numerical_features = df.select_dtypes(include=[np.number]).columns

# =====================================
# Step 4: OUTLIER DETECTION USING IQR
# =====================================

print("\n================ OUTLIER SUMMARY (IQR Method) ================\n")

outlier_summary = []

for feature in numerical_features:
    
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    
    outlier_count = outliers.shape[0]
    total_count = df.shape[0]
    percentage = (outlier_count / total_count) * 100
    
    outlier_summary.append([
        feature, 
        round(Q1,3), 
        round(Q3,3), 
        round(lower_bound,3), 
        round(upper_bound,3), 
        outlier_count, 
        round(percentage,2)
    ])

# Convert to DataFrame
outlier_df = pd.DataFrame(outlier_summary, 
                          columns=["Feature", "Q1", "Q3", 
                                   "Lower Bound", "Upper Bound", 
                                   "Outlier Count", "Outlier %"])

print(outlier_df)

# =====================================
# Step 5: Plot Histograms
# =====================================
plt.figure(figsize=(15, 10))

for i, feature in enumerate(numerical_features):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df[feature], bins=30, kde=True)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# =====================================
# Step 6: Plot Box Plots
# =====================================
plt.figure(figsize=(15, 10))

for i, feature in enumerate(numerical_features):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x=df[feature])
    plt.title(f"Box Plot of {feature}")
    plt.xlabel(feature)

plt.tight_layout()
plt.show()

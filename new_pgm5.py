PROGRAM 5 new
AIM

To develop a Python program to implement the k-Nearest Neighbour (KNN) algorithm for classifying randomly generated data points in the range [0,1], and to analyze the effect of different values of k.
import numpy as np, matplotlib.pyplot as plt
from collections import Counter

data = np.random.rand(100)
labels = ["Class1" if x<=0.5 else "Class2" for x in data[:50]]

def euclidean_distance(x1,x2): return abs(x1-x2)

def knn_classifier(train_data,train_labels,test_point,k):
    d = sorted([(euclidean_distance(test_point,train_data[i]),train_labels[i])
                for i in range(len(train_data))])
    return Counter([l for _,l in d[:k]]).most_common(1)[0][0]

train_data, train_labels = data[:50], labels
test_data = data[50:]

k_values = [1,2,3,4,5,20,30]
results = {}

print("--- k-NN Classification ---\n")

for k in k_values:
    print(f"Results for k = {k}:")
    
    classified_labels = [knn_classifier(train_data,train_labels,x,k) for x in test_data]
    results[k] = classified_labels
    
    for i,label in enumerate(classified_labels,51):
        print(f"Point x{i} (value: {test_data[i-51]:.4f}) -> {label}")
    print()

print("Classification complete.\n")

for k in k_values:
    classified_labels = results[k]
    
    class1_points = [test_data[i] for i in range(len(test_data)) if classified_labels[i]=="Class1"]
    class2_points = [test_data[i] for i in range(len(test_data)) if classified_labels[i]=="Class2"]
    
    plt.figure(figsize=(10,6))
    plt.scatter(train_data,[0]*len(train_data),
        c=["blue" if l=="Class1" else "red" for l in train_labels], label="Training Data")
    
    plt.scatter(class1_points,[1]*len(class1_points),c="blue",label="Class1 (Test)")
    plt.scatter(class2_points,[1]*len(class2_points),c="red",label="Class2 (Test)")
    
    plt.title(f"k-NN (k={k})"); plt.legend(); plt.grid(); plt.show()

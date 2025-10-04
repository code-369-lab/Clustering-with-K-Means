# K-Means Clustering on Real Estate Dataset
# -----------------------------------------
# Objective: Perform unsupervised learning using K-Means

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Step 1: Load dataset
data = pd.read_csv("Real Estate DataSet.csv")
print("Dataset Loaded Successfully!\n")
print(data.head())

# Step 2: Handle missing values (if any)
data = data.dropna()

# Step 3: Select numeric features for clustering
X = data.select_dtypes(include=['float64', 'int64'])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Elbow Method to find optimal K
inertia = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# Step 5: Fit KMeans with chosen K (say 3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(X_scaled)
data['Cluster'] = labels

# Step 6: PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
plt.title(f"K-Means Clustering Visualization (K={optimal_k})")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label="Cluster")
plt.show()

# Step 7: Evaluate using Silhouette Score
score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {score:.3f}")

# Step 8: Save clustered dataset
data.to_csv("Real_Estate_Clustered.csv", index=False)
print("\nClustered data saved as 'Real_Estate_Clustered.csv'")

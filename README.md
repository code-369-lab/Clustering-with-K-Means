# 🏠 Real Estate Clustering using K-Means

## 📌 Objective
This project applies **K-Means Clustering**, an unsupervised machine learning algorithm, to group real estate data into meaningful clusters based on similar property attributes.

---

## ⚙️ Tools & Libraries
- **Python 3**
- **Pandas** – Data manipulation  
- **Scikit-learn** – Machine learning (KMeans, PCA, Silhouette Score)  
- **Matplotlib** – Data visualization  

---

## 📂 Dataset
File: `Real Estate DataSet.csv`

The dataset includes real estate attributes such as transaction date, house age, distance to MRT station, number of convenience stores, latitude, and longitude.

---

## 🧩 Steps Performed

1. **Load & Clean Data**  
   Handle missing values and select numeric columns for clustering.

2. **Feature Scaling**  
   Standardize numerical features using `StandardScaler`.

3. **Find Optimal K**  
   Use the **Elbow Method** to visualize inertia and determine the best number of clusters.

4. **K-Means Clustering**  
   Fit the K-Means model to the scaled data and assign cluster labels.

5. **Dimensionality Reduction (PCA)**  
   Reduce high-dimensional data to 2D for visualization.

6. **Visualization**  
   Plot clusters in 2D color-coded scatter plots.

7. **Evaluation**  
   Compute the **Silhouette Score** to measure cluster quality.

8. **Export Results**  
   Save the dataset with cluster labels as `Real_Estate_Clustered.csv`.

---

## 📊 Output
- **Elbow Plot** – Helps identify the best number of clusters.  
- **Cluster Visualization (PCA 2D Plot)** – Shows how data points are grouped.  
- **Silhouette Score** – Quantitative measure of cluster quality.

---

## 🧠 Example Results

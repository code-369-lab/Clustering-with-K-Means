# Mall Customer Segmentation with K-Means

This project demonstrates **unsupervised learning** using **K-Means clustering** on the Mall Customer Segmentation dataset.

---

## 📂 Dataset
The dataset is loaded directly from an online source (no need to download manually):
- **URL:** [Mall Customers Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mall_customers.csv)

---

## ⚙️ Steps Performed

1. **Load Dataset**  
   - Used `pandas.read_csv()` to read directly from the online link.

2. **Feature Selection**  
   - Selected features: **Age**, **Annual Income (k$)**, and **Spending Score (1-100)**.

3. **Elbow Method**  
   - Used to determine the optimal number of clusters by plotting inertia vs. k.

4. **K-Means Clustering**  
   - Applied K-Means with the optimal k (commonly `k=5` for this dataset).  
   - Assigned cluster labels to customers.

5. **Evaluation**  
   - Used **Silhouette Score** to evaluate clustering quality.

6. **Visualization**  
   - Reduced features to **2D with PCA** and plotted clusters with different colors.  
   - Displayed centroids as black `X`.

---

## 📊 Output

- **Elbow Method Graph** → Helps choose k.  
- **Cluster Visualization (PCA)** → Customers grouped in different colors.  
- **Silhouette Score** → Numerical measure of clustering quality.  
- **clustered_customers.csv** → Dataset with assigned cluster labels.

---

## 🚀 Tools & Libraries

- Python  
- Pandas  
- Matplotlib  
- Scikit-learn  

---

## ▶️ How to Run

1. Install dependencies:
   ```bash
   pip install pandas matplotlib scikit-learn

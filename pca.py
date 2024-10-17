import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import joblib

# Step 1: Prepare your dataset
# For example, assuming you have a dataset loaded into a DataFrame
# Replace this with your actual dataset
data = pd.read_csv(r'C:\Users\91989\PycharmProjects\pythonProject3\.venv\LUNG PROJECT\lung_tumor_features_with_labels.csv')

# Step 2: Create and fit the PCA model
pca = PCA(n_components=25)  # Specify the number of components you want
pca.fit(data)  # Fit the PCA model on your dataset

# Step 3: Save the PCA model for future use
joblib.dump(pca, 'new_trained_pca_model.pkl')

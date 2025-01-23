import pandas as pd
import numpy as np

# Load data with missing values marked as '?'
train = pd.read_csv('training_companydata.csv', na_values='?')
test = pd.read_csv('test_unlabeled.csv', na_values='?')

# Check dimensions
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Display first 5 rows and missing values
print(train.head())
print("Missing values in training data:\n", train.isnull().sum())
print("Missing values in test data:\n", test.isnull().sum())
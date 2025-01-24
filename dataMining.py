import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Load data with missing values marked as '?'
train = pd.read_csv('training_companydata.csv', na_values='?')
test = pd.read_csv('test_unlabeled.csv', na_values='?')

# Check dimensions
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Display first 5 rows and missing values
print(train.head())
print("Missing values in training data:\n", train.isnull().sum())
print("Missing values in test data:\n", test.isnull().sum())

from sklearn.impute import SimpleImputer

# Identify columns with >50% missing values
missing_cols = train.columns[train.isnull().mean() > 0.5]
train.drop(columns=missing_cols, inplace=True)
test.drop(columns=missing_cols, inplace=True)

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Log-transform skewed features (e.g., X25)
# train['X25_log'] = np.log1p(train['X25'])
# test['X25_log'] = np.log1p(test['X25'])

# Median imputation for remaining missing values
imputer = SimpleImputer(strategy='median', missing_values=np.nan)
train_filled = pd.DataFrame(imputer.fit_transform(train), columns=train.columns)
test_filled = pd.DataFrame(imputer.transform(test), columns=test.columns)

# --- NEW CODE: Log Transform Here ---
# 1. Define skewed features (customize this list)
# Calculate skewness for all numeric features
numeric_cols = train_filled.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('X65')  # Exclude target

skewness = train_filled[numeric_cols].skew().sort_values(ascending=False)
print("Top 5 skewed features:")
print(skewness.head(5))

# Visualize before transformation (example: X25)
import seaborn as sns

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(train_filled['X25'], kde=True)
plt.title('Original X25 Distribution')

# For features with NO negative values
skewed_features = skewness[abs(skewness) > 1].index.tolist()  # Threshold = ±1
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')
for col in skewed_features:
    if (train_filled[col] < 0).all() or (test_filled[col] < 0).all:  # Avoid negative values
        train_filled[f'{col}_transformed'] = pt.fit_transform(train_filled[[col]])
        test_filled[f'{col}_transformed'] = pt.transform(test_filled[[col]])
        train_filled.drop(col, axis=1, inplace=True)
        test_filled.drop(col, axis=1, inplace=True)
        continue

    train_filled[f'{col}_log'] = np.log1p(train_filled[col])
    test_filled[f'{col}_log'] = np.log1p(test_filled[col])
    train_filled.drop(col, axis=1, inplace=True)  # Drop original column
    test_filled.drop(col, axis=1, inplace=True)

print("Missing values in training data:\n", train_filled.isnull().sum())
print("Missing values in test data:\n", test_filled.isnull().sum())

print(f"Train shape: {train_filled.shape}, Test shape: {test_filled.shape}")

# Optional: Add missingness flags
# for col in train.columns[train.isnull().any()]:
#   train_filled[f'{col}_missing'] = train[col].isnull().astype(int)
#  test_filled[f'{col}_missing'] = test[col].isnull().astype(int)

from sklearn.feature_selection import VarianceThreshold

# Remove constant/quasi-constant features
# Preserve column names after VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
train_clean = pd.DataFrame(
    selector.fit_transform(train_filled),
    columns=train_filled.columns[selector.get_support()]  # Keep feature names
)

test_clean = pd.DataFrame(
    selector.transform(test_filled),
    columns=train_filled.columns[selector.get_support()]  # Same columns as train
)

print(f"Train shape: {train_clean.shape}, Test shape: {test_clean.shape}")


# After handling missing values and before feature scaling
def winsorize(df, lower=0.05, upper=0.95):
    """Apply Winsorization to all numeric columns in a DataFrame."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'X65':  # Skip target variable
            lower_bound = df[col].quantile(lower)
            upper_bound = df[col].quantile(upper)
            df[col] = df[col].clip(lower_bound, upper_bound)
    return df


# Apply to training and test data
train_clean = winsorize(train_clean)
test_clean = winsorize(test_clean)

print(f"Train shape: {train_clean.shape}, Test shape: {test_clean.shape}")

print(type(train_clean))  # Should output: <class 'pandas.core.frame.DataFrame'>
print(train_clean.columns.tolist())  # Should show retained feature names

class_dist = train_clean['X65'].value_counts(normalize=True)
print(f"Class Distribution:\n{class_dist}")

plt.figure(figsize=(8, 5))
plt.bar(['Non-Bankrupt (0)', 'Bankrupt (1)'], class_dist.values)
plt.title("Class Distribution")
plt.ylabel("Proportion")
plt.show()

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV

X = train_clean.drop(columns=['X65'])
y = train_clean['X65'].astype(int)

# 1. Split data with stratification
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

# 2. Apply SMOTE to training data only
smote = SMOTE(sampling_strategy=0.8)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 3. Train cost-sensitive XGBoost
# Define parameter grid


scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

param_grid = {
    # 'subsample':[i/100.0 for i in range(65,75)],
    # 'colsample_bytree':[i/100.0 for i in range(85,95)]
    # "learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
}

xgb = XGBClassifier(max_depth=6,
                    min_child_weight=0.01,
                    learning_rate=0.05,
                    gamma=0,
                    n_estimators=5000,
                    reg_alpha=0,
                    reg_lambda=0.5,
                    scale_pos_weight=scale_pos_weight,
                    colsample_bytree=0.85,
                    subsample=0.7,
                    eval_metric='aucpr'  # Optimize for AUC-PR
                    )
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, scoring='average_precision')
grid_search.fit(X_train_res, y_train_res)

# Print best parameters
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

import shap
import matplotlib.pyplot as plt


def shap_feature_selection(xgb, X_train, y_train, X_test, top_n=20):
    """Perform SHAP-based feature selection and return filtered datasets"""
    xgb.fit(X_train, y_train)

    # 1. Calculate SHAP values
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_train)

    # 2. Get feature importance
    shap_df = pd.DataFrame({
        'features': X_train.columns,
        'shap_importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)

    # 3. Select top features
    selected_features = shap_df.head(top_n)['features'].tolist()

    # 4. Plot feature importance
    shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=top_n)

    # selected_features = ['X27', 'X24', 'X58', 'X46', 'X34', 'X60', 'X6', 'X61', 'X33', 'X64', 'X9', 'X44', 'X23', 'X21', 'X20', 'X5', 'X32', 'X59', 'X29', 'X39']

    # 5. Filter datasets
    return X_train[selected_features], X_test[selected_features], selected_features


X_train_res_selected, X_val_selected, selected_features = shap_feature_selection(
    xgb, X_train_res, y_train_res, X_val
)

# Update your model training
xgb.fit(X_train_res_selected, y_train_res)

# Filter test data
test_clean_selected = test_clean[selected_features]

# Verify shapes
print(f"Selected training shape: {X_train_res_selected.shape}")
print(f"Selected test shape: {test_clean_selected.shape}")
print(f"\nTop {len(selected_features)} Features Selected:")
print(selected_features)

from xgboost import plot_importance

plot_importance(xgb, max_num_features=20)
plt.show()

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, \
    ConfusionMatrixDisplay

# Generate predictions and probabilities
y_pred = xgb.predict(X_val_selected)
y_proba = xgb.predict_proba(X_val_selected)[:, 1]

# Calculate metrics
report = classification_report(y_val, y_pred, output_dict=True)
roc_auc = roc_auc_score(y_val, y_proba)
pr_auc = average_precision_score(y_val, y_proba)

# Create metrics dataframe
metrics_df = pd.DataFrame({
    'Metric': ['Precision (Class 0)', 'Recall (Class 0)', 'F1 (Class 0)',
               'Precision (Class 1)', 'Recall (Class 1)', 'F1 (Class 1)',
               'ROC AUC', 'PR AUC', 'Accuracy'],
    'Value': [
        report['0']['precision'], report['0']['recall'], report['0']['f1-score'],
        report['1']['precision'], report['1']['recall'], report['1']['f1-score'],
        roc_auc, pr_auc, report['accuracy']
    ]
})

# Format the table
metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.3f}")
print("Performance Report:")
print(metrics_df.to_markdown(index=False))

# Confusion matrix visualization
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Non-Bankrupt', 'Bankrupt'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Ensure test data has the same features as training data
# (Already handled during preprocessing, but double-check)
# test_clean_selected = test_clean_selected.drop(columns=['X65'])

assert set(test_clean_selected.columns) == set(X_train_res_selected.columns), "Feature mismatch!"

# Generate predictions and probabilities for test data
test_pred = xgb.predict(test_clean_selected)
test_proba = xgb.predict_proba(test_clean_selected)[:, 1]  # Probability of bankruptcy (Class 1)

# Create submission files
# 1. For all test samples (class predictions)
pd.Series(test_pred).to_csv('final_predictions.csv', index=False, header=False)

# 2. Top 50 risky companies (assuming rowid starts at 1)
top_50 = pd.DataFrame({
    'rowid': np.arange(1, len(test_proba) + 1)[np.argsort(-test_proba)[:50]]
})

top_50.to_csv('top_50_risky.csv', index=False, header=False)

print("Predictions saved:")
print(f"- Class predictions for all samples: final_predictions.csv")
print(f"- Top 50 risky companies: top_50_risky.csv")

plt.close('all')

import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# -----------------------
# Load and clean dataset
# -----------------------
df = pd.read_csv("PCOS_infertility.csv")

# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Quick look at first 5 rows
print(df.head())

# Column info and missing values
print(df.info())
print(df.isnull().sum())

# Summary statistics
print(df.describe())

# Drop unnecessary columns
df = df.drop(columns=['Sl. No', 'Patient File No.'], errors='ignore')

# Convert AMH to numeric and fill missing
df['AMH(ng/mL)'] = pd.to_numeric(df['AMH(ng/mL)'], errors='coerce')
df['AMH(ng/mL)'] = df['AMH(ng/mL)'].fillna(df['AMH(ng/mL)'].median())

# Check cleaned data
print(df.info())
print(df.head())

# -----------------------
# Features and target
# -----------------------
X = df.drop('PCOS (Y/N)', axis=1)
y = df['PCOS (Y/N)']

print("Features (X):")
print(X.head())
print("\nTarget distribution:")
print(y.value_counts())

# -----------------------
# Train-test split and scaling
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------
# Logistic Regression
# -----------------------
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("Logistic Regression Results")
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))

# -----------------------
# Random Forest
# -----------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Results")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# Feature importance
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importance:")
print(importance)

# -----------------------
# Exploratory Data Analysis (histograms)
# -----------------------

hist_features = ['PCOS (Y/N)', 'I   beta-HCG(mIU/mL)', 'II    beta-HCG(mIU/mL)', 'AMH(ng/mL)']

for col in hist_features:
    if col in df.columns:
        plt.figure()
        plt.hist(df[col], bins=20)
        plt.xlabel(col)
        plt.ylabel("Number of patients")
        plt.title(f"Distribution of {col}")
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', col)
        plt.savefig(f"{safe_name}_hist.png")
        plt.close()
    else:
        print(f"Column {col} not found in DataFrame, skipping plot.")

import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# Load the dataset
df = pd.read_csv("PCOS_infertility.csv")

# Quick look at the first 5 rows
print(df.head())

# Check column types and missing values
print(df.info())
print(df.isnull().sum())

# Summary statistics for numeric columns
print(df.describe())


# Drop unnecessary columns
df = df.drop(columns=['Sl. No', 'Patient File No.'], errors='ignore')

# Convert AMH(ng/mL) to numeric (it was a string)
df['AMH(ng/mL)'] = pd.to_numeric(df['AMH(ng/mL)'], errors='coerce')

# Fill any missing AMH values with median
df['AMH(ng/mL)'] = df['AMH(ng/mL)'].fillna(df['AMH(ng/mL)'].median())

# Check the cleaned data
print(df.info())
print(df.head())


# Features = all columns except the target
X = df.drop('PCOS (Y/N)', axis=1)

# Target = PCOS (Y/N)
y = df['PCOS (Y/N)']

# Check features and target
print("Features (X):")
print(X.head())
print("\nTarget distribution:")
print(y.value_counts())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split dataset into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Train Logistic Regression
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Results
print("Logistic Regression Results")
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))



from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Results
print("Random Forest Results")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))



# Show which features are most important
import pandas as pd

importance = pd.Series(rf.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("Feature Importance:")
print(importance)


# -----------------------
# Exploratory Data Analysis
# -----------------------

import re
import matplotlib.pyplot as plt

features = ["Age", "BMI", "Insulin", "Testosterone"]

for col in features:
    plt.figure()
    plt.hist(df[col], bins=20)
    plt.xlabel(col)
    plt.ylabel("Number of patients")
    plt.title(f"Distribution of {col}")

    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', col)  # clean filename

    plt.savefig(f"{safe_name}_hist.png")
    plt.close()




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib
import os

# Define the features to use for training. This list must be consistent
# between the training script and the Streamlit app.
features = ['koi_period', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_srad', 'koi_slogg']

try:
    # NOTE: This code assumes the CSV file is named 'cumulative.csv'
    # If your file has a different name, please change it here.
    df = pd.read_csv('cumulative.csv', comment='#', on_bad_lines='skip', engine='python')
    print("Dataset loaded successfully.")

except FileNotFoundError:
    print("Error: 'cumulative.csv' not found. Please download the dataset and place it in the same directory as this script.")
    exit()

# Drop rows that are not 'CONFIRMED' or 'FALSE POSITIVE'
df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]

# Map the target variable to numerical values
y = df['koi_disposition'].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})

# Prepare the feature matrix X
X = df[features]

# Handle any missing values in the selected features
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and scalers
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

print("\nModel training complete. Model, imputer, scaler, and feature names saved to disk.")
print("You can now run 'streamlit run app.py' to launch the application.")

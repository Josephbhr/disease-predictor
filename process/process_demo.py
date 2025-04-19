import pandas as pd
import numpy as np
import json
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
raw_data = pd.read_csv('../dataset/heart.csv')

def encode(data):
    return pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)

def normalize(data):
    scaler = StandardScaler()
    numeric = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    data[numeric] = scaler.fit_transform(data[numeric])
    return data

def random_from_column(column):
    unique_vals = column.unique()
    return int(random.choice(unique_vals))

def simulate_user(data):
    numeric = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    user = {}
    for col in data.columns:
        if col in numeric:
            user[col] = np.random.normal(data[col].mean(), data[col].std())
        else:
            user[col] = random_from_column(data[col])
    return pd.Series(user)

# Train models
encoded_data = encode(normalize(raw_data.copy()))
X = encoded_data.drop("target", axis=1)
y = encoded_data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

lr = LogisticRegression(C=0.1, max_iter=1000)
lr.fit(X_train, y_train)

# Simulate and preprocess new user
sample_raw = simulate_user(raw_data.drop(columns="target"))
sample_raw_df = pd.concat([raw_data.drop("target", axis=1), sample_raw.to_frame().T], ignore_index=True)
sample_raw_df = sample_raw_df.astype({'sex': 'int', 'cp': 'int', 'fbs': 'int', 'restecg': 'int', 'exang': 'int', 'slope': 'int', 'ca': 'int', 'thal': 'int'})
sample_encoded = encode(sample_raw_df)
sample_user_encoded = sample_encoded.iloc[[-1]]

# Predict probabilities
rf_prob = rf.predict_proba(sample_user_encoded)[0][1]
lr_prob = lr.predict_proba(sample_user_encoded)[0][1]

# Simple rule-based risk factor explanation
thresholds = {
    "chol": 240,
    "trestbps": 130,
    "thalach": 100,
    "oldpeak": 2.0
}
risk_flags = []
for feature, limit in thresholds.items():
    raw_val = sample_raw[feature]
    if feature != "thalach" and raw_val > limit:
        risk_flags.append(f"{feature} high ({raw_val:.1f})")
    elif feature == "thalach" and raw_val < limit:
        risk_flags.append(f"{feature} low ({raw_val:.1f})")

# Prepare JSON output
output = {
    "user_profile": sample_raw.to_dict(),
    "RandomForest_RiskScore": f"{rf_prob:.2%}",
    "LogisticRegression_RiskScore": f"{lr_prob:.2%}",
    "Predicted_Risk": "High Risk" if rf_prob > 0.5 or lr_prob > 0.5 else "Low Risk",
    "Contributing_Risk_Factors": risk_flags if risk_flags else ["None"]
}

# Save to JSON
with open("../output/simulated_user_long_term.json", "w") as f:
    json.dump(output, f, indent=4)

print("Simulated user prediction saved to simulated_user_long_term.json")

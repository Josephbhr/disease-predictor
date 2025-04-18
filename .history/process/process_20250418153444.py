import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import random

# Load the dataset
raw_data = pd.read_csv('../dataset/heart.csv')


def encode(data):
    # One-hot encode categorical variables
    data_encoded = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)
    return data_encoded

def normalize(data_encoded):
    # Scale numeric features
    scaler = StandardScaler()
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    data_encoded[numerical_cols] = scaler.fit_transform(data_encoded[numerical_cols])
    return data_encoded



def random_from_column(column):
    val = []
    count = []
    for entry in column:
        if not entry in val:
            val.append(entry)
            count.append(1)
        else:
            count[val.index(entry)] += 1
    rd = random.randint(1, len(column))
    index = 0
    while True:
        rd -= count[index]
        if rd > 0:
            index += 1
        else:
            return int(val[index])


def simulate_sample_user(raw_data):
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    sample_user = {}
    for column in raw_data.columns:
        if column in numerical_cols:
            mean = raw_data[column].mean()
            std_dev = raw_data[column].std()
            sample_user[column] = np.random.normal(loc=mean, scale=std_dev)
        else:
            sample_user[column] = random_from_column(raw_data[column])
    
    return pd.Series(sample_user)


data = raw_data.copy()
data = encode(normalize(data))
print(data)

print(data.isnull().sum())

X = data.drop("target", axis=1)
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


sample_dataset = raw_data.copy()
sample_dataset = sample_dataset.drop("target", axis=1)
sample_user = simulate_sample_user(normalize(sample_dataset))
print(sample_user)
sample_dataset = pd.concat([sample_dataset, sample_user.to_frame().T], ignore_index=True)
sample_dataset = sample_dataset.astype({'sex': 'int', 'cp': 'int', 'fbs': 'int', 'restecg': 'int', 'exang': 'int', 'slope': 'int', 'ca': 'int', 'thal': 'int'})
print(sample_dataset)
sample_dataset = encode(sample_dataset)
print(sample_dataset)
sample_user = sample_dataset.iloc[-1]
sample_user = sample_user.to_frame().T

print(sample_user)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

prediction = model.predict(sample_user)
risk = "High Risk" if prediction[0] == 1 else "Low Risk"

print("Long-Term Prediction for this User:", risk)


model2 = LogisticRegression(C=0.1)
model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred2))
print(classification_report(y_test, y_pred2))

prediction = model2.predict(sample_user)
risk = "High Risk" if prediction[0] == 1 else "Low Risk"

print("Long-Term Prediction for this User:", risk)

# Predict on the full dataset X
rf_predictions = model.predict(X)
lr_predictions = model2.predict(X)

# Prepare DataFrame with both models' predictions
results_df = pd.DataFrame({
    'RandomForest_Prediction': ['High Risk' if p == 1 else 'Low Risk' for p in rf_predictions],
    'LogisticRegression_Prediction': ['High Risk' if p == 1 else 'Low Risk' for p in lr_predictions]
})

# Save to CSV
results_df.to_csv('../output/long_term_prediction_results.csv', index=False)
print("All long-term prediction results saved to long_term_prediction_results.csv")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Create 'models' folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("R:\student-performance-ml\Student-Performance-Prediction-System\data\student_data.csv")

# Features and target
X = df[['study_hours', 'attendance', 'previous_score', 'extra_activities', 'parental_support', 'health_index']]
y = df['pass']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = LogisticRegression()
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, 'models/logistic_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("✅ Model and scaler saved to /models/")
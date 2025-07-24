import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# Load your data
df = pd.read_csv(r"R:\student-performance-ml\Student-Performance-Prediction-System\data\StudentsPerformance.csv")

# Convert categorical columns
label_encoders = {}
categorical_cols = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and label
X = df.drop(columns=["math score", "reading score", "writing score"])  # inputs
y = ((df["math score"] + df["reading score"] + df["writing score"]) / 3 >= 60).astype(int)  # pass = 1, fail = 0

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test)))

# Save model and label encoders
joblib.dump(model, "models/logistic_model.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

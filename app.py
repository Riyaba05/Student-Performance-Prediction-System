from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load dataset
df = pd.read_csv(r"R:\student-performance-ml\Student-Performance-Prediction-System\data\StudentsPerformance.csv")
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
df['pass'] = df['average_score'].apply(lambda x: 1 if x >= 40 else 0)

# Encode categorical columns
label_encoders = {}
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and Target
features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch',
            'test preparation course', 'math score', 'reading score', 'writing score']
X = df[features]
y = df['pass']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

@app.route("/", methods=["GET"])
def form():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    gender = request.form['gender']
    race = request.form['race']
    parent_edu = request.form['parent_education']
    lunch = request.form['lunch']
    test_prep = request.form['test_prep']
    math = int(request.form['math'])
    reading = int(request.form['reading'])
    writing = int(request.form['writing'])

    # Encode input values using the label encoders
    input_data = [
        label_encoders['gender'].transform([gender])[0],
        label_encoders['race/ethnicity'].transform([race])[0],
        label_encoders['parental level of education'].transform([parent_edu])[0],
        label_encoders['lunch'].transform([lunch])[0],
        label_encoders['test preparation course'].transform([test_prep])[0],
        math, reading, writing
    ]

    pred = model.predict([input_data])[0]
    prediction = "Pass" if pred == 1 else "Fail"

    # Generate feedback
    avg_math = df["math score"].mean()
    avg_reading = df["reading score"].mean()
    avg_writing = df["writing score"].mean()

    weak_areas = []
    if math < avg_math: weak_areas.append("Math")
    if reading < avg_reading: weak_areas.append("Reading")
    if writing < avg_writing: weak_areas.append("Writing")
    feedback = "Improve in: " + ", ".join(weak_areas) if weak_areas else "You’re doing great!"

    # Generate comparison chart
    labels = ['Math', 'Reading', 'Writing']
    student_scores = [math, reading, writing]
    avg_scores = [avg_math, avg_reading, avg_writing]

    x = range(len(labels))
    plt.figure(figsize=(6, 4))
    plt.bar(x, student_scores, width=0.4, label="You", align='center')
    plt.bar([i + 0.4 for i in x], avg_scores, width=0.4, label="Class Avg", align='center')
    plt.xticks([i + 0.2 for i in x], labels)
    plt.ylabel("Scores")
    plt.title("Your Scores vs Class Average")
    plt.legend()
    plt.tight_layout()

    chart_path = os.path.join("static", "comparison.png")
    plt.savefig(chart_path)
    plt.close()

    return render_template("result.html", prediction=prediction, feedback=feedback, show_chart=True)

if __name__ == "__main__":
    app.run(debug=True)
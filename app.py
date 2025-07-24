from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [
            float(request.form['study_hours']),
            float(request.form['attendance']),
            float(request.form['previous_score']),
            float(request.form['extra_activities']),
            float(request.form['parental_support']),
            float(request.form['health_index'])
        ]
        data_scaled = scaler.transform([data])
        prediction = model.predict(data_scaled)[0]
        return render_template("index.html", prediction=f"Prediction: {'Pass' if prediction == 1 else 'Fail'}")
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)


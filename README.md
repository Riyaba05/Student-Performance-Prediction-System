🎓 Student Performance Prediction System

A machine learning-powered web application built using Flask that predicts student performance based on their demographic and academic attributes. The system also allows users to register, log in, and view their prediction history.

Features:
Predicts student performance using ML models (Random Forest, XGBoost, Logistic Regression)
User Registration and Login (using Flask-Login)
Input form for student details and exam scores
Displays prediction results with feedback and graph
Tracks prediction history per user (optional database integration)

ML Models Used:
Random Forest Classifier
XGBoost Classifier
Logistic Regression
All models are trained using datasets with attributes like gender, parental education level, lunch type, test preparation course, and scores in math, reading, and writing.

How to Run the Project:

1.Clone the repository:
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction

2.Create a virtual environment:
python -m venv venv
source venv/bin/activate (For Windows: venv\Scripts\activate)

3.Install dependencies:
pip install -r requirements.txt

4.Run the application:
python app.py

Open your browser and go to:
http://127.0.0.1:5000

Future Improvements:
Add PostgreSQL integration for storing prediction history
Deploy on Render or Heroku
Add model accuracy metrics and comparisons
Enable CSV export of prediction history

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from model_pipeline import predict_churn

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input data from the form
        input_data = {
            "SeniorCitizen": int(request.form["SeniorCitizen"]),
            "tenure": float(request.form["tenure"]),
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "gender": request.form["gender"],
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "MultipleLines": request.form["MultipleLines"],
            "InternetService": request.form["InternetService"],
            "OnlineSecurity": request.form["OnlineSecurity"],
            "OnlineBackup": request.form["OnlineBackup"],
            "DeviceProtection": request.form["DeviceProtection"],
            "TechSupport": request.form["TechSupport"],
            "StreamingTV": request.form["StreamingTV"],
            "StreamingMovies": request.form["StreamingMovies"]
        }

        # Predict churn
        prediction = predict_churn(input_data)

        # Show result on the webpage
        return render_template("result.html", prediction="Churn" if prediction == 1 else "No Churn")

    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

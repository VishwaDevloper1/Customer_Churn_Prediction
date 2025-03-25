import pickle
import numpy as np
import pandas as pd

# Load the trained model
model_path = "artifects/xgboost_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Load the preprocessor (data transformation pipeline)
preprocessor_path = "artifects/preprocessor.pkl"
with open(preprocessor_path, "rb") as file:
    preprocessor = pickle.load(file)

# Define expected feature names & order
FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "gender", "Partner", "Dependents",
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

# Function to preprocess input and make prediction
def predict_churn(input_data):
    """
    Transforms input data and makes a churn prediction.
    :param input_data: Dictionary of input features.
    :return: Predicted churn class (0 or 1)
    """
    try:
        # Convert input dictionary to DataFrame (single row)
        input_df = pd.DataFrame([input_data])

        # Apply transformations
        transformed_features = preprocessor.transform(input_df)

        # Predict using trained model
        prediction = model.predict(transformed_features)[0]

        return int(prediction)

    except Exception as e:
        return str(e)  # Return error message if transformation fails

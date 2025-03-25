import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    Preprocessor_file_path: str = os.path.join("artifects", "preprocessor.pkl")  # Fixed typo in "artifacts"

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_transformer_obj(self, df):
        all_columns = df.columns.tolist()

        num_col = [col for col in ["SeniorCitizen", "tenure", "MonthlyCharges"] if col in all_columns]
        cat_col = [col for col in ["gender", "Partner", "Dependents", 'MultipleLines', 'InternetService',
                                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                   'StreamingTV', 'StreamingMovies', 'Churn'] if col in all_columns]  # Keep Churn in cat_col

        # Categorical Pipeline (including Churn)
        cat_pipeline = Pipeline([
            ("label_encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])

        # Define column transformer
        preprocessor_pipeline = ColumnTransformer([
            ("num_pipeline", "passthrough", num_col),  # Keep numerical columns as-is
            ("cat_pipeline", cat_pipeline, cat_col),   # Apply OneHotEncoding to categorical columns (including Churn)
        ])

        return preprocessor_pipeline

    def initiate_data_transformation(self, train_path, test_path):
        """Loads data, applies transformations first, then drops unnecessary columns."""
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        preprocessor_obj = self.get_transformer_obj(train_df)

        # Apply transformations first
        input_feature_train_arr = preprocessor_obj.fit_transform(train_df)
        input_feature_test_arr = preprocessor_obj.transform(test_df)

        # feature_names = preprocessor_obj.get_feature_names_out()
        # print("Transformed Column Names:", feature_names)

        # Convert transformed arrays back to DataFrame
        transformed_train_df = pd.DataFrame(input_feature_train_arr)
        transformed_test_df = pd.DataFrame(input_feature_test_arr)

        # Define columns to drop AFTER transformation
        drop_cols = ["customerID", "PaperlessBilling", "PaymentMethod", "TotalCharges"]

        # Drop only existing columns
        transformed_train_df.drop(columns=[col for col in drop_cols if col in transformed_train_df.columns], errors="ignore", inplace=True)
        transformed_test_df.drop(columns=[col for col in drop_cols if col in transformed_test_df.columns], errors="ignore", inplace=True)

        transformed_train_df = np.array(transformed_train_df)
        transformed_test_df = np.array(transformed_test_df)
        # Save preprocessor pipeline
        save_object(
            file_path=self.transformation_config.Preprocessor_file_path,
            obj=preprocessor_obj,
        )

        return transformed_train_df, transformed_test_df, self.transformation_config.Preprocessor_file_path

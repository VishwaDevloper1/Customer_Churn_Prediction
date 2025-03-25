import os
import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import accuracy_score, classification_report
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    model_file_path: str = os.path.join("artifects", "xgboost_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, x_train, y_train, x_test, y_test):
        clf = xgb.XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=200, tree_method="hist",
                                early_stopping_rounds=10)

        xgmodel = clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)

        # Predictions
        y_pred = xgmodel.predict(x_test)

        # Model Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Save model
        os.makedirs(os.path.dirname(self.config.model_file_path), exist_ok=True)
        with open(self.config.model_file_path, "wb") as f:
            pickle.dump(xgmodel, f)

        return {
            "model_path": self.config.model_file_path,
            "accuracy": accuracy,
            "classification_report": report
        }

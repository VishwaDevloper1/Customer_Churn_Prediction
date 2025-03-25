import os
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from src.Components.Data_tranformation import DataTransformation
from src.Components.Model_trainer import ModelTrainer
@dataclass
class DataIngestionconfig:
    train_data_path: str = os.path.join('artifects', 'train.csv')
    test_data_path: str = os.path.join('artifects', 'test.csv')
    raw_data_path: str = os.path.join('artifects', 'data.csv')


class DataIngestion:
 def __init__(self):
   self.ingestion_config = DataIngestionconfig()

 def initiate_data_ingestion(self) -> object:
  df = pd.read_csv(r'D:\python\Customer_Churn_Modelling\raw_data\Telco-Customer-Churn.csv')

  os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
  df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

  train_set,test_set = train_test_split(df, test_size=0.2, random_state=42)

  train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
  test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)

  return (  self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path )

if __name__ == '__main__':
  obj = DataIngestion()
  train_data,test_data = obj.initiate_data_ingestion()

  data_transformation = DataTransformation()
  train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)


  X_train = train_arr[:, :-1]
  y_train = train_arr[:, -1].astype(int)

  X_test = test_arr[:, :-1]
  y_test = test_arr[:, -1].astype(int)

  # Call Model Trainer
  model_trainer = ModelTrainer()
  print(model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test))

import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from dataclasses import dataclass

# from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        logging.info(f"DataIngestionConfig initialized with: {self.ingestion_config}")

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process")
        try:
            df = pd.read_csv('notebook/data/spam.csv', encoding="ISO-8859-1")
            logging.info("Dataset read successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved to %s", self.ingestion_config.raw_data_path)

            return self.ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    raw_data =obj.initiate_data_ingestion()
    logging.info("Data ingestion completed successfully")

    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor_obj, tfidf = data_transformation.initiate_data_transformation(raw_data)
    logging.info("Data transformation completed successfully")

    # Step 3: (Optional) Pass to ModelTrainer
    # model_trainer = ModelTrainer()
    # model_trainer.initiate_model_training(X_train, X_test, y_train, y_test)
import sys
import os
import pickle
from dataclasses import dataclass

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    vectorizer_path: str = os.path.join("artifacts", "vectorizer.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, X_test, y_train, y_test, tfidf):
        """
        Trains a MultinomialNB model, evaluates it, and saves the model + TF-IDF vectorizer.
        """
        try:
            logging.info("Model training initiated")

            # Train the model
            model = MultinomialNB()
            model.fit(X_train, y_train)
            logging.info("Model training completed")

            # Predictions
            y_pred = model.predict(X_test)

            # Evaluation metrics
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            precision = precision_score(y_test, y_pred)

            logging.info(f"Accuracy: {acc}")
            logging.info(f"Confusion Matrix: \n{cm}")
            logging.info(f"Precision: {precision}")

            # Save trained model
            os.makedirs(os.path.dirname(self.model_trainer_config.model_path), exist_ok=True)
            pickle.dump(model, open(self.model_trainer_config.model_path, "wb"))

            # Save TF-IDF vectorizer
            pickle.dump(tfidf, open(self.model_trainer_config.vectorizer_path, "wb"))

            logging.info(f"Model saved at {self.model_trainer_config.model_path}")
            logging.info(f"Vectorizer saved at {self.model_trainer_config.vectorizer_path}")

            return {
                "accuracy": acc,
                "confusion_matrix": cm,
                "precision": precision,
                "model_path": self.model_trainer_config.model_path,
                "vectorizer_path": self.model_trainer_config.vectorizer_path
            }

        except Exception as e:
            raise CustomException(e, sys)

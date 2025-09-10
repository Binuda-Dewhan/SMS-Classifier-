import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import os
import nltk


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text_column(df):
        df = df.copy()

        def transform_text(text):
            text = text.lower()
            text = nltk.word_tokenize(text)

            # remove special chars
            y = [i for i in text if i.isalnum()]

            # remove stopwords & punctuation
            y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

            # stemming
            y = [ps.stem(i) for i in y]

            return " ".join(y)

        df['transformed_text'] = df['text'].apply(transform_text)
        return df

def prepare_features_and_split(df, max_features=3000, test_size=0.2, random_state=2):
    """
    Vectorize text with TF-IDF and split into train/test sets.
    Saves df_transformed.csv automatically.
    """
    # Vectorize text
    tfidf = TfidfVectorizer(max_features=max_features)
    X = tfidf.fit_transform(df['transformed_text']).toarray()

    # Target variable
    y = df['target'].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Save transformed dataframe
    df.to_csv("artifacts/df_transformed.csv", index=False)

    return X_train, X_test, y_train, y_test, tfidf



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')  
    transformed_data_path: str = os.path.join('artifacts', 'transformed_data.csv')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        try:
            logging.info("Data Transformation initiated")

            # ✅ Custom function for feature engineering
            def add_text_features(df):
                df = df.copy()

                # Label encoding target
                le = LabelEncoder()
                df['target'] = le.fit_transform(df['target'])

                # Feature engineering
                df['num_characters'] = df['text'].apply(len)
                df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
                df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

                return df

            # Wrap inside sklearn FunctionTransformer so it works in a pipeline
            feature_engineering = FunctionTransformer(add_text_features, validate=False)

            # Step 2: Apply transform_text to text column
            text_cleaning = FunctionTransformer(transform_text_column, validate=False)

            # Build pipeline
            preprocessor = Pipeline(
                steps=[
                    ('feature_engineering', feature_engineering),
                    ('text_cleaning', text_cleaning)
                ]
            )

            logging.info("Data Transformation pipeline created successfully")
            return preprocessor

        except Exception as e: 
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, raw_path):
        try:
            # Read the raw data
            df = pd.read_csv(raw_path, encoding="ISO-8859-1")
            logging.info("Read raw data successfully")

            #changing columns 
            drop_col = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]

            # drop unnecessary columns
            df.drop(columns=drop_col, inplace=True)
            logging.info(f"Dropped columns: {drop_col}")

            # Rename column 'v1' to 'target' and 'v2' to 'text'
            df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)
            logging.info("Renamed columns 'v1' to 'target' and 'v2' to 'text'")

            # Drop duplicate rows across all columns
            df.drop_duplicates(inplace=True)
            logging.info("Dropped duplicate rows")

            preprocessor_obj = self.get_data_transformer_object()

            # Apply pipeline
            df_transformed = preprocessor_obj.fit_transform(df)
            logging.info("Applied data transformation pipeline")

            # ✅ Save to CSV
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_data_path), exist_ok=True)
            df_transformed.to_csv(self.data_transformation_config.transformed_data_path, index=False)

            logging.info(f"Transformed data saved to {self.data_transformation_config.transformed_data_path}")

            # ✅ Apply TF-IDF + train/test split
            X_train, X_test, y_train, y_test, tfidf = prepare_features_and_split(df_transformed)

            return X_train, X_test, y_train, y_test, preprocessor_obj, tfidf

        except Exception as e:
            raise CustomException(e, sys)

            
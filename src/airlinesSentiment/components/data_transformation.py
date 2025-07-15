from airlinesSentiment.entity.artifacts_entity import DataTransformationArtifact
from airlinesSentiment.entity.config_entity import DataTransformationConfig
from airlinesSentiment.components.data_preprocessing import DataPreprocessor
from pathlib import Path
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

        self.text_preprocessor = DataPreprocessor(config.data_preprocessing_config)

    def transform(self, data:pd.DataFrame) ->DataTransformationArtifact:
        data['cleaned_text'] = self.data_preprocessor.preprocess_batch(data[self.config.text_column].tolist())

        #create output directories if they don't exist

        self.config.vectorize_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.cleaned_data_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.features_path.parent.mkdir(parents=True, exist_ok=True)


        #save cleaned data
        data.to_csv(self.config.cleaned_data_path, index=False)


        #vectorization
        vectorizer = TfidfVectorizer(max_features=5000)
        features = vectorizer.fit_transform(data['cleaned_text'])

        #save vectorizer and features
        joblib.dump(vectorizer, self.config.vectorize_path)
        joblib.dump(features, self.config.features_path)

        return DataTransformationArtifact(
            vectorizer_path=self.config.vectorize_path,
            cleaned_data_path=self.config.cleaned_data_path,
            features_path=self.config.features_path
        )







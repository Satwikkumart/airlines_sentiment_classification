from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataPreprocessingConfig:
    spacy_model: str
    remove_punctuation: bool
    lowercase: bool
    lemmatize: bool
    remove_stopwords: bool
    custom_stopwords: List[str]

@dataclass(frozen=True)
class DataTransformationConfig:
    data_preprocessing_config: DataPreprocessingConfig
    text_column: str
    vectorize_path: Path
    cleaned_data_path: Path
    features_path: Path
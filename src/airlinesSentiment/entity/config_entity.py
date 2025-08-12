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
    root_dir : Path
    training_data_path : Path
    training_data_file : Path
    training_cleansed_data: Path
    datasets_dir: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    data_preprocessing_config: DataPreprocessingConfig
    text_column: str
    vectorize_path: Path
    cleaned_data_path: Path
    features_path: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    params_num_train_epochs: int
    params_per_device_train_batch_size: int
    params_per_device_eval_batch_size: int
    params_warmup_steps: int
    params_weight_decay: float
    params_max_steps: int
    params_save_steps: int
    params_logging_steps: int


@dataclass(frozen=True)
class TrainingConfig:
    datasets_dir: Path
    output_dir: Path
    base_model_path: Path
    model_save_path: Path
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    warmup_steps: int
    weight_decay: float
    max_steps: int
    save_steps: int
    logging_steps: int

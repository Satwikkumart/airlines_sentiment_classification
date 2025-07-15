from airlinesSentiment.constants import *
from airlinesSentiment.utils.common import read_yaml, create_directories
from airlinesSentiment.entity.config_entity import (DataIngestionConfig, DataPreprocessingConfig, DataTransformationConfig)


class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH):

            self.config = read_yaml(config_filepath)
            self.params = read_yaml(params_filepath)

            create_directories([self.config.artifacts_root])
        

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config


    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config["data_preprocessing"]
        return DataPreprocessingConfig(**config)
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
         config = self.config["data_tranformation"]


         data_preprocessing_config = DataPreprocessingConfig(
              spacy_model=config["data_preprocessing"]["spacy_model"],
              remove_punctuation=config["data_preprocessing"]["remove_punctuation"],
              lowercase=config["data_preprocessing"]["lowercase"],
              lemmatize=config["data_preprocessing"]["lemmatize"],
              remove_stopwords=config["data_preprocessing"]["remove_stopwords"],
              custom_stopwords=config["data_preprocessing"]["custom_stopwords"]
         )

         return DataTransformationConfig(
              data_preprocessing_config= data_preprocessing_config,
              text_column=config["text_column"],
              vectorize_path=Path(config["vectorize_path"]),
              cleaned_data_path=Path(config["cleaned_data_path"]),
              features_path=Path(config["features_path"])
             
        )
         
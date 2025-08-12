from airlinesSentiment import logger
from airlinesSentiment.components.feature_engineering import DataPreprocessing
from airlinesSentiment.config.configuration import ConfigurationManager
# from airlinesSentiment.entity.artifacts_entity import DataTransformationArtifact


STAGE_NAME = "Feature Engineering pipeline"

class FeatureEngineeringPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        get_data_pre_config = config.get_data_preprocessing_config()
        data_pre_process = DataPreprocessing(config=get_data_pre_config)

        # Preprocess the data
        data_pre_process.text_process()
        data_pre_process.mapping_labels_func()
        data_pre_process.tokenize_text()

        # Save the preprocessed data
        data_pre_process.save_data()   #No need to pass the path

    # Split the dataset into train, validation, and test sets
        splits = data_pre_process.train_val_test_split()

    # Convert splits into pytorch datasets
        tokenized_datasets = data_pre_process.convert_to_tokenized_datasets(splits)

    #Access the pytorch datasets
        train_dataset = tokenized_datasets['train']
        val_dataset = tokenized_datasets['val']
        test_dataset = tokenized_datasets['test']
        logger.info("PyTorch datasets created successfully")
        datasets = data_pre_process.save_datasets(train_dataset, val_dataset, test_dataset)


    # def run_pipeline(self, data: pd.DataFrame) -> DataTransformationArtifact:

    #     transformation_config = self.config_manager.get_data_transformation_config()


    #     #initialize and run transformation config
    #     data_transformation = DataTransformation(transformation_config)
    #     return data_transformation.transform(data)
    

    if __name__ == '__main__':
        try:
            logger.info(f">>> stage {STAGE_NAME} started <<<<<")
            obj = FeatureEngineeringPipeline()
            obj.main()
            logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n x==============x")
        
        except Exception as e:
             logger.exception(e)
             raise e



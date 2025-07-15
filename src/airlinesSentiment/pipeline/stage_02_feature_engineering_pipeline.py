import pandas as pd
from airlinesSentiment import logger
from airlinesSentiment.pipeline.stage_02_feature_engineering_pipeline import FeaturePipeline
from airlinesSentiment.config.configuration import ConfigurationManager
from airlinesSentiment.components.data_transformation import DataTransformation
from airlinesSentiment.entity.artifacts_entity import DataTransformationArtifact


STAGE_NAME = "Feature Engineering pipeline"

class FeaturePipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()


    def run_pipeline(self, data: pd.DataFrame) -> DataTransformationArtifact:

        transformation_config = self.config_manager.get_data_transformation_config()


        #initialize and run transformation config
        data_transformation = DataTransformation(transformation_config)
        return data_transformation.transform(data)
    

    if __name__ == '__main__':
        try:
            logger.info(f">>> stage {STAGE_NAME} started <<<<<")
            obj = FeaturePipeline()
            obj.main()
            logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n x==============x")
        
        except Exception as e:
             logger.exception(e)
             raise e



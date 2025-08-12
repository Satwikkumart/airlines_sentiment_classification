from airlinesSentiment import logger
from airlinesSentiment.pipeline.stage_02_feature_engineering_pipeline import FeatureEngineeringPipeline
from airlinesSentiment.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from airlinesSentiment.pipeline.stage_03_base_model import BaseModelPipeline
from airlinesSentiment.pipeline.stage_04_model_training import ModelTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<< ")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed \n\n x================x")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Feature Engineering pipeline"

try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<< ")
    obj = FeatureEngineeringPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed \n \n x==============x")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training Pipeline"

try:
    logger.info(f" >>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<< \n \n x===============x")

except Exception as e:
    logger.exception(e)
    raise e


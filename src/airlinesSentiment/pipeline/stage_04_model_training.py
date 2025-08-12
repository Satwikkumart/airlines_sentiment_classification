from airlinesSentiment.config.configuration import ConfigurationManager
from airlinesSentiment.components.model_training import ModelTraining
from airlinesSentiment import logger

STAGE_NAME = "Model Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        trainingconfig = config.get_training_config()
        model_training = ModelTraining

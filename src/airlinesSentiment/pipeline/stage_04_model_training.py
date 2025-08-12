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
        model_training = ModelTraining(config=trainingconfig)
        model_training.train()
        model_training.evaluate()


if __name__ == '__main__':
    try:
        logger.info(F">>>>>>>> stage name {STAGE_NAME} started <<<<<<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f" >>>>>>> stage {STAGE_NAME} completed <<<<<<<<<< \n \n x============x")
    
    except Exception as e:
        logger.exception(e)
        raise e
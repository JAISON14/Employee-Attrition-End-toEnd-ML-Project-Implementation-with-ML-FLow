from Employee_Attition_End_to_end_ML_project_with_MLflow.config.configuration import ConfigurationManager
from Employee_Attition_End_to_end_ML_project_with_MLflow.components.data_transformation import DataTransformation
from Employee_Attition_End_to_end_ML_project_with_MLflow import logger
from Employee_Attition_End_to_end_ML_project_with_MLflow.entity import DataTransformationConfig
from pathlib import Path





STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.data_preprocessing()
                data_transformation.train_test_spliting()

            else:
                raise Exception("You data schema is not valid")

        except Exception as e:
            print(e)





if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e







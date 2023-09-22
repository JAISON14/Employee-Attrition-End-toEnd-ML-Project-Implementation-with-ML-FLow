from Employee_Attition_End_to_end_ML_project_with_MLflow import logger
from Employee_Attition_End_to_end_ML_project_with_MLflow.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

# from Employee_Attition_End_to_end_ML_project_with_MLflow import DataValidationTrainingPipeline
# from Employee_Attition_End_to_end_ML_project_with_MLflow import DataTransformationTrainingPipeline
# from Employee_Attition_End_to_end_ML_project_with_MLflow import ModelTrainerTrainingPipeline
# from Employee_Attition_End_to_end_ML_project_with_MLflow import ModelEvaluationTrainingPipeline


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e




# STAGE_NAME = "Data Validation stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = DataValidationTrainingPipeline()
#    data_ingestion.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e



# STAGE_NAME = "Data Transformation stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = DataTransformationTrainingPipeline()
#    data_ingestion.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e




# STAGE_NAME = "Model Trainer stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = ModelTrainerTrainingPipeline()
#    data_ingestion.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e



# STAGE_NAME = "Model evaluation stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = ModelEvaluationTrainingPipeline()
#    data_ingestion.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e







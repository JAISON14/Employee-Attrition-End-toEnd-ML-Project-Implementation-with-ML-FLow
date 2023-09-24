from Employee_Attition_End_to_end_ML_project_with_MLflow.config.configuration import ConfigurationManager
from Employee_Attition_End_to_end_ML_project_with_MLflow.components.model_trainer import ModelTrainer
from Employee_Attition_End_to_end_ML_project_with_MLflow import logger

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.ensemble import RUSBoostClassifier, EasyEnsembleClassifier

STAGE_NAME = "Model Trainer stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            models = [
                 (LogisticRegression(random_state=42), "Logistic Regression"),
                 (SVC(random_state=42), "Support Vector Machines"),
                (KNeighborsClassifier(), "K-Nearest Neighbors"),
                (DecisionTreeClassifier(random_state=42), "Decision Trees"),
                (RandomForestClassifier(random_state=42), "Random Forest"),
                (GradientBoostingClassifier(random_state=42), "Gradient Boosting"),
                (MLPClassifier(random_state=42), "Neural Networks"),
                (GaussianNB(), "Naive Bayes"),
                (AdaBoostClassifier(random_state=42), "AdaBoost"),
                (XGBClassifier(random_state=42), "XGBoost"),
                (LGBMClassifier(random_state=42), "LightGBM"),
                (CatBoostClassifier(random_state=42, verbose=0), "CatBoost"),
                (RUSBoostClassifier(random_state=42),"RUSBoost"),
                (EasyEnsembleClassifier(random_state=42),"EasyEnsemble")
            ]


            for model, model_type in models:
                model_trainer_config = config_manager.get_model_trainer_config(model_type=model_type)
                trainer = ModelTrainer(config=model_trainer_config, model=model)
                trainer.train()

        except Exception as e:
            raise e




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

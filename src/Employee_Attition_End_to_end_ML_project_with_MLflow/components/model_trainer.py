import pandas as pd
import os
from Employee_Attition_End_to_end_ML_project_with_MLflow import logger
from sklearn.linear_model import ElasticNet
import joblib
from Employee_Attition_End_to_end_ML_project_with_MLflow.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
       # train_data = pd.read_csv(self.config.train_data_path)
        #test_data = pd.read_csv(self.config.test_data_path)

        # joblib.load(os.path.join(self.config.root_dir, "y.pkl"))
        train_x = joblib.load(self.config.train_data_path)
        test_x = joblib.load(self.config.test_data_path)
        train_y = joblib.load(self.config.train_target_path)
        test_y = joblib.load(self.config.test_target_path)


        lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))

import pandas as pd
import os
from Employee_Attition_End_to_end_ML_project_with_MLflow import logger
from sklearn.linear_model import ElasticNet
import joblib
import mlflow
import mlflow.sklearn 
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Employee_Attition_End_to_end_ML_project_with_MLflow.entity.config_entity import ModelTrainerConfig
from urllib.parse import urlparse



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig,model):
        self.config = config
        self.model = model

    
    def train(self):
            # Set MLflow server URI
            mlflow.set_tracking_uri('https://dagshub.com/JAISON14/End_to_End_Implementation.mlflow')
            

            mlflow.set_experiment(f"Experiment_{self.config.model_type}")
            
            with mlflow.start_run(run_name=f"Run_{self.config.model_type}") as run:
                train_x = joblib.load(self.config.train_data_path)
                test_x = joblib.load(self.config.test_data_path)
                train_y = joblib.load(self.config.train_target_path)
                test_y = joblib.load(self.config.test_target_path)
                
                model = clone(self.model)

                # Dynamic Parameter Setting
                valid_params = self.model.get_params().keys()
                
                # Assuming self.config.hyperparameters is already model-specific
                hyperparams = {k: v for k, v in self.config.hyperparameters.items() if k in valid_params}
                
                self.model.set_params(**hyperparams)
                model.fit(train_x, train_y)
                
                # Predict on train and test sets
                train_pred = model.predict(train_x)
                test_pred = model.predict(test_x)
                
                # Calculate metrics for the train set
                train_accuracy = accuracy_score(train_y, train_pred)
                train_precision = precision_score(train_y, train_pred, average='weighted')
                train_recall = recall_score(train_y, train_pred, average='weighted')
                train_f1 = f1_score(train_y, train_pred, average='weighted')
                
                # Log metrics to MLflow
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("train_precision", train_precision)
                mlflow.log_metric("train_recall", train_recall)
                mlflow.log_metric("train_f1", train_f1)
                
                # Calculate metrics for the test set
                test_accuracy = accuracy_score(test_y, test_pred)
                test_precision = precision_score(test_y, test_pred, average='weighted')
                test_recall = recall_score(test_y, test_pred, average='weighted')
                test_f1 = f1_score(test_y, test_pred, average='weighted')
                
                # Log metrics to MLflow
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.log_metric("test_precision", test_precision)
                mlflow.log_metric("test_recall", test_recall)
                mlflow.log_metric("test_f1", test_f1)
                
                # Log hyperparameters
                mlflow.log_params(self.config.hyperparameters)
                
                # Log model
                mlflow.sklearn.log_model(model, f"{self.config.model_name}_{self.config.model_type}")
                joblib.dump(model, os.path.join(self.config.root_dir, f"{self.config.model_type}.pkl"))





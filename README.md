# Employee Attrition End toEnd ML Project Implementation with ML FLow
![image](https://github.com/JAISON14/Employee-Attrition-End-toEnd-ML-Project-Implementation-with-ML-FLow/assets/24632348/eb0f0a2b-220d-4510-9803-5f185d07d6eb)

Dataset: https://www.kaggle.com/datasets/patelprashant/employee-attrition 
Jupiter notebook used for EDA and preliminary analysis: https://github.com/JAISON14/Employee_Attrition 

## About this project

-> Conducted exploratory data analysis (EDA) on an employee attrition dataset and identified significant features impacting attrition, such as OverTime, JobRole, and BusinessTravel.

-> Performed data preprocessing, including handling missing values and encoding categorical variables.

-> Applied feature selection techniques, including univariate statistical tests, feature importance scores, and recursive feature elimination, to select the most relevant features.

-> Trained and evaluated multiple binary classification algorithms, utilizing stratified K-Fold cross-validation to account for class imbalance in the dataset.

-> Selected the F1-score as the primary evaluation metric to balance precision and recall.

-> Using MLFLow and Dagshub, I conducted 78 experiments with 14 classification algorithms. The neural network achieved an F1-score of 0.83, while Naive Bayes achieved 0.829 on the test set.

-> Explored techniques to handle imbalanced datasets, such as oversampling with SMOTE and ensemble methods like RUSBoost and EasyEnsemble.

-> Analyzed feature importance scores and partial dependence plots to interpret the models' decision-making processes, identifying OverTime, JobRole, and BusinessTravel as key features.

-> Developed predictive models to identify employees at risk of leaving the company, enabling timely interventions and retention strategies to improve employee satisfaction and reduce attrition rates.

-> Deployed in AWS using Docker and Github Actions.

## Screenshots
# Web App
![image](https://github.com/JAISON14/Employee-Attrition-End-toEnd-ML-Project-Implementation-with-ML-FLow/assets/24632348/592defc0-6eee-44b6-ab70-561f4b02c391)
![image](https://github.com/JAISON14/Employee-Attrition-End-toEnd-ML-Project-Implementation-with-ML-FLow/assets/24632348/ad04cc69-f32c-4e2e-98f3-c899c75c1de3)
![image](https://github.com/JAISON14/Employee-Attrition-End-toEnd-ML-Project-Implementation-with-ML-FLow/assets/24632348/1cebb6f4-53f1-431a-95b6-a8a99a08f5ea)
![image](https://github.com/JAISON14/Employee-Attrition-End-toEnd-ML-Project-Implementation-with-ML-FLow/assets/24632348/3737fbe0-1684-4cbb-b0a4-bc47657fe336)
# MLFLow Console 
![image](https://github.com/JAISON14/Employee-Attrition-End-toEnd-ML-Project-Implementation-with-ML-FLow/assets/24632348/caf09252-521f-4b21-80ae-8cdc9d4def55)
# AWS
![image](https://github.com/JAISON14/Employee-Attrition-End-toEnd-ML-Project-Implementation-with-ML-FLow/assets/24632348/4c57aa5c-3901-4049-92ab-e3c3747fa8c1)
![image](https://github.com/JAISON14/Employee-Attrition-End-toEnd-ML-Project-Implementation-with-ML-FLow/assets/24632348/ce7a233a-6c60-40cf-ac1b-4a5bb56fbf84)

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py



# How to run?
### STEPS:

Clone the repository

```bash
[https://github.com/entbappy/End-to-end-Machine-Learning-Project-with-MLflow](https://github.com/JAISON14/Employee-Attrition-End-toEnd-ML-Project-Implementation-with-ML-FLow)
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mlproject python=3.8 -y
```

```bash
conda activate mlproject
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```



## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)


Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI="Enter Tracking URL"

export MLFLOW_TRACKING_USERNAME="Enter USername"

export MLFLOW_TRACKING_PASSWORD="Enter Tracking Password"

```


# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 351534541670.dkr.ecr.us-east-2.amazonaws.com/mlproject

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  351534541670.dkr.ecr.us-east-2.amazonaws.com

    ECR_REPOSITORY_NAME = mlproject




## About MLflow 
MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & tagging your model# End_to_End_Implementation

import joblib 
import numpy as np
import pandas as pd
from pathlib import Path



class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('./artifacts/model_trainer/Naive Bayes.pkl'))


    def predict(self, data):
        col_names =['Age', 'BusinessTravel', 'DailyRate', 'Department',
                'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
                'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
                'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
                'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
                'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
                'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
                'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
                'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                'YearsWithCurrManager']
        df = pd.DataFrame([data], columns=col_names)

        df['TWY_JobLevel'] = df['TotalWorkingYears'] * df['JobLevel']
        # Aggregation Features
        df['Income_to_Age'] = df['MonthlyIncome'] / df['Age']
        # Polynomial Features (as an example, we'll square the YearsAtCompany)
        df['YearsAtCompany_sq'] = df['YearsAtCompany']**2
        #df[['TWY_JobLevel', 'Income_to_Age', 'YearsAtCompany_sq']].head()

        # Drop features that provide no information
        df.drop(columns=['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], inplace=True)

        # Define numerical and categorical features
        numerical_features = df.select_dtypes(include=['int64', 'float64','int']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        preprocessor = joblib.load(Path('./artifacts/data_transformation/preprocessor.pkl'))
        selected_features = joblib.load(Path('./artifacts/data_transformation/selected_features.pkl'))
        print("preprocessor & selected_features loaded")
        print("List of Selected Features")

        # Apply the preprocessor to the input data
        X_processed = preprocessor.transform(df)

        # Get the one-hot encoded feature names
        encoded_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)

        # Combine the numerical and one-hot encoded feature names
        all_features = numerical_features + list(encoded_features)

        # Convert the processed data to a DataFrame
        X_df = pd.DataFrame(X_processed, columns=all_features)

        # Now, filter using the selected features
        X_selected = X_df[selected_features]


        prediction = self.model.predict(X_selected)
        if prediction == 0:
            result ="Company is at the risk of losing this employee"
        else:
            result ="Company is not at the risk of losing this employee"
        print(result)
        return result
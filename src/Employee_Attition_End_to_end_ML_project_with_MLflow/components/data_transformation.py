import os
from Employee_Attition_End_to_end_ML_project_with_MLflow import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from Employee_Attition_End_to_end_ML_project_with_MLflow.entity.config_entity import DataTransformationConfig
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import joblib


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def data_preprocessing(self):
        try:        
            df = pd.read_csv(self.config.data_path)
            logger.info(f"df loaded")
            # Interaction Features
            df['TWY_JobLevel'] = df['TotalWorkingYears'] * df['JobLevel']
            # Aggregation Features
            df['Income_to_Age'] = df['MonthlyIncome'] / df['Age']
            # Polynomial Features (as an example, we'll square the YearsAtCompany)
            df['YearsAtCompany_sq'] = df['YearsAtCompany']**2
            df[['TWY_JobLevel', 'Income_to_Age', 'YearsAtCompany_sq']].head()
            # Drop features that provide no information
            df.drop(columns=['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], inplace=True)
            # Define numerical and categorical features
            numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = df.select_dtypes(include=['object']).columns.tolist()
            # Remove target variable from categorical features
            categorical_features.remove('Attrition')

            # Define the preprocessing pipeline for numerical features
            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            # Define the preprocessing pipeline for categorical features
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(drop='first'))
            ])

            # Combine the preprocessing pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            # Apply the preprocessing pipeline to the data
            X = preprocessor.fit_transform(df)
            y = df['Attrition'].map({'Yes': 1, 'No': 0}).values
            logger.info(f"Pre-processing completed Successfully")
            # FEATURE SELECTION
            
            # Apply ANOVA F-statistic for numerical features
            anova_selector = SelectKBest(score_func=f_classif, k='all')
            X_numerical = preprocessor.transformers_[0][1].fit_transform(df[numerical_features])
            anova_selector.fit(X_numerical, y)

            # Apply chi-squared test for categorical features
            chi2_selector = SelectKBest(score_func=chi2, k='all')
            X_categorical = preprocessor.transformers_[1][1].fit_transform(df[categorical_features])
            chi2_selector.fit(X_categorical, y)

            # Calculate feature importance scores using Random Forest
            rf = RandomForestClassifier(random_state=42)
            rf.fit(X, y)
            feature_importances = rf.feature_importances_

            # Perform Recursive Feature Elimination (RFE)
            rfe = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=15)
            rfe.fit(X, y)

            # Get the feature names after one-hot encoding
            encoded_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)

            # Combine the numerical and encoded categorical feature names
            all_features = numerical_features + encoded_features.tolist()

            # Update the feature scores DataFrame
            feature_scores = pd.DataFrame({
                'Features': all_features,
                'ANOVA F-statistic': list(anova_selector.scores_) + ['N/A'] * len(encoded_features),
                'Chi-Squared Test': ['N/A'] * len(numerical_features) + list(chi2_selector.scores_),
                'Feature Importances': feature_importances,
                'RFE Support': rfe.support_,
                'RFE Ranking': rfe.ranking_
            })
            # ANOVA F-statistic for numerical features
            anova_scores = pd.DataFrame({
                'Numerical Features': numerical_features,
                'ANOVA F-statistic': anova_selector.scores_
            })
            anova_scores = anova_scores.sort_values(by='ANOVA F-statistic', ascending=False)

            # Chi-Squared Test for categorical features
            chi2_scores = pd.DataFrame({
                'Categorical Features': encoded_features,
                'Chi-Squared Test': chi2_selector.scores_
            })
            chi2_scores = chi2_scores.sort_values(by='Chi-Squared Test', ascending=False)

            # Feature Importance Scores from Random Forest
            feature_importance_scores = pd.DataFrame({
                'Features': all_features,
                'Feature Importances': feature_importances
            })
            feature_importance_scores = feature_importance_scores.sort_values(by='Feature Importances', ascending=False)

            # Recursive Feature Elimination (RFE)
            rfe_scores = pd.DataFrame({
                'Features': all_features,
                'RFE Support': rfe.support_,
                'RFE Ranking': rfe.ranking_
            })
            rfe_scores = rfe_scores.sort_values(by='RFE Ranking')

            # Selection Criteria
            # To select the features to be used for the machine learning model, we'll consider the following criteria:

            # 1. Features with high ANOVA F-statistic scores (for numerical features) or high chi-squared test scores (for categorical features).
            # 2. Features with high feature importance scores (from Random Forest).
            # 3. Features selected by the Recursive Feature Elimination (RFE) method.
            # 4. We'll select the features that satisfy at least two of the above criteria. After selecting the features, we'll also remove features that are highly correlated with each other to avoid multicollinearity.

            # Let's start by selecting the features based on the above criteria.
            # Initialize a dictionary to keep track of how many criteria each feature satisfies
            feature_criteria_counts = {}

            # Define a function to add features that meet a criterion to the dictionary
            def add_features(features, criterion_name):
                for feature in features:
                    if feature not in feature_criteria_counts:
                        feature_criteria_counts[feature] = 1
                    else:
                        feature_criteria_counts[feature] += 1

            # Select features with high ANOVA F-statistic scores
            anova_selected = set(anova_scores[anova_scores['ANOVA F-statistic'] > 10]['Numerical Features'])
            add_features(anova_selected, 'ANOVA')

            # Select features with high chi-squared test scores
            chi2_selected = set(chi2_scores[chi2_scores['Chi-Squared Test'] > 10]['Categorical Features'])
            add_features(chi2_selected, 'Chi-Squared')

            # Select features with high feature importance scores
            importance_selected = set(feature_importance_scores[feature_importance_scores['Feature Importances'] > 0.03]['Features'])
            add_features(importance_selected, 'Feature Importance')

            # Select features selected by RFE
            rfe_selected = set(rfe_scores[rfe_scores['RFE Support']]['Features'])
            add_features(rfe_selected, 'RFE')

            # Identify features that satisfy at least two criteria
            selected_features = [feature for feature, count in feature_criteria_counts.items() if count >= 2]

            # Create a DataFrame with only the selected features
            selected_df = pd.DataFrame(X, columns=all_features)[selected_features]

            # Remove features that are highly correlated with each other
            correlation_matrix = selected_df.corr()
            upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape, dtype=bool), k=1))
            highly_correlated_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]

            # Remove the highly correlated features
            for feature in highly_correlated_features:
                if feature in selected_features:
                    selected_features.remove(feature)
            #X_selected=df[selected_features]
            # Create a DataFrame with only the selected features
            X_selected = pd.DataFrame(X, columns=all_features)[selected_features]
            X_selected.to_csv(os.path.join(self.config.root_dir, "X_selected.csv"),index = False)
            
            joblib.dump(preprocessor,(os.path.join(self.config.root_dir, "preprocessor.pkl")))
            joblib.dump(selected_features,(os.path.join(self.config.root_dir, "selected_features.pkl")))
            joblib.dump(y,(os.path.join(self.config.root_dir, "y.pkl")))
            logger.info("Feature Selection Completed Successfully")
        except Exception as e:
            raise e
    def train_test_spliting(self):
        try:
            data = pd.read_csv(self.config.data_path)
            logger.info("Train Test Splitting Started")
            X_selected = pd.read_csv(os.path.join(self.config.root_dir, "X_selected.csv"))
            y = joblib.load(os.path.join(self.config.root_dir, "y.pkl"))

            # Split the data into training and test sets. (0.75, 0.25) split.
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

            joblib.dump(X_train,(os.path.join(self.config.root_dir, "X_train.pkl")))
            joblib.dump(X_test,(os.path.join(self.config.root_dir, "X_test.pkl")))
            joblib.dump(y_train,(os.path.join(self.config.root_dir, "y_train.pkl")))
            joblib.dump(y_test,(os.path.join(self.config.root_dir, "y_test.pkl")))

            logger.info("Splited data into training and test sets")
            logger.info(X_train.shape)
            logger.info(y_train.shape)

            logger.info(X_test.shape)
            logger.info(y_test.shape)
        except Exception as e:
            raise e

        
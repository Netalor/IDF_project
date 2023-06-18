from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from confs.conf import *
import pandas as pd
import json

from src.models.evaluation import Evaluation
from src.visualization.visualize import shap_plot

import numpy as np


class CorrelationSelectionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr = np.absolute(np.corrcoef(X, rowvar=False))
        logger.info(f'corr: {corr}')
        upper = corr * np.triu(np.ones(corr.shape), k=1).astype(np.bool)
        to_drop = [column for column in range(upper.shape[1]) if any(upper[:, column] >= self.threshold)]
        self.columns_to_drop_ = to_drop
        logger.info(f'columns_to_drop_: {to_drop}')
        logger.info(f'columns_to_drop_: {len(to_drop)}')
        return self

    def transform(self, X):
        return np.delete(X, self.columns_to_drop_, axis=1)


def create_pipeline(X, y, cumulative_threshold=0.8):
    # Define the preprocessing steps for numerical and categorical features
    numerical_transformer = "passthrough"
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore'))
    ])

    # Identify categorical features with few observations
    infrequent_categories = []
    for feature in categorical_features:
        category_counts = X[feature].value_counts()
        category_cumulative_perc = category_counts.cumsum() / category_counts.sum()
        filtered_categories = category_cumulative_perc[
            category_cumulative_perc <= cumulative_threshold].index.tolist()
        infrequent_categories.append(filtered_categories)
        logger.info(f'{feature}: {filtered_categories}')

    # Create the column transformer for numerical and categorical features
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)

    ])

    if algo_name == 'xgboost':
        classifier = XGBClassifier(
            n_jobs=-1,
            eval_metric=["auc", "logloss"],
            verbose=1,
            use_label_encoder=False)
    else:
        classifier = LogisticRegression()

        # Create the pipeline with preprocessing and logistic regression modeling
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('correlation_selection', CorrelationSelectionTransformer()),

        ('classifier', classifier)

    ])
    return pipeline


def evaluation(y, y_pred, y_pred_proba):
    # Evaluate the model
    evaluation_results = classification_report(y, y_pred, output_dict=True)
    evaluation_results['roc_auc_score'] = roc_auc_score(y, y_pred_proba)
    # logger.info(evaluation_results)
    logger.info('\n'.join([f'{metric} : {score}' for metric, score in evaluation_results.items()]))

    f"output/{folder_name}/analytics/na_columns.csv"
    with open(f"output/{folder_name}/model/evaluation.json", "w") as fp:
        json.dump(evaluation_results, fp)

    evaluation = Evaluation(y=y, proba=y_pred_proba, output_path=f"output/{folder_name}/model/")
    eval_results = evaluation.eval_classification()
    with open(f"output/{folder_name}/model/evaluation_2.json", "w") as fp:
        json.dump(eval_results, fp)


def get_model_components(pipeline):
    # Access the logistic regression model from the pipeline
    model = pipeline.named_steps['classifier']

    # Access the preprocessor from the pipeline
    preprocessor = pipeline.named_steps['preprocessor']
    correlation_selection = pipeline.named_steps['correlation_selection']

    # Get the feature names after preprocessing
    numerical_feature_names = numerical_features  # Assuming numerical_features is a list of numerical feature names
    logger.info(f'numerical_feature_names len: {len(numerical_feature_names)}')

    categorical_transformer = preprocessor.named_transformers_['cat']
    categorical_feature_names = categorical_transformer.named_steps['encoder'].get_feature_names_out(
        categorical_features)
    logger.info(f'categorical_feature_names len: {len(categorical_feature_names)}')

    all_feature_names = []
    # Concatenate numerical and categorical feature names
    all_feature_names.extend(numerical_feature_names)
    logger.info(len(all_feature_names))

    all_feature_names.extend(categorical_feature_names)
    logger.info(f'all_feature_names len: {len(all_feature_names)}')
    logger.info(f'all_feature_names: {all_feature_names}')
    features = [i for j, i in enumerate(all_feature_names) if j not in correlation_selection.columns_to_drop_]
    logger.info(f'features: {features}')
    return model, features


def get_model_coef(pipeline):
    model, feature_names = get_model_components(pipeline)
    if algo_name == 'xgboost':
        # get feature importances
        importances = model.feature_importances_
        # sort the scores in descending order
        sorted_idx = importances.argsort()[::-1]
        final_feature_names = []
        for idx in sorted_idx:
            logger.info('{}: {}'.format(feature_names[idx], importances[idx]))
        logger.info(f'feature_names: {feature_names}')
        logger.info(f'len feature_names: {len(feature_names)}')
        logger.info(f'importances: {importances}')
        logger.info(f'len importances: {len(importances)}')
        # Create a DataFrame with feature names and coefficients
        coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': importances})
        coefficients_df['abs_Coefficient'] = coefficients_df['Coefficient'].abs()

    else:
        # Get the coefficients of the logistic regression model
        coefficients = model.coef_[0]
        # Create a DataFrame with feature names and coefficients
        coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
        # logger.info the coefficients
        coefficients_df['abs_Coefficient'] = coefficients_df['Coefficient'].abs()

    coefficients_df.sort_values(by='abs_Coefficient', ascending=False).round(2).\
        to_csv(f"output/{folder_name}/model/coefficients.csv")
    return coefficients_df

# to focus in conjoin data, make assumptions about the prbtability of the market
# to calculate market share like the ppt,
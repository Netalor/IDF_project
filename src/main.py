from sklearn.metrics import make_scorer

from confs.conf import *
import pandas as pd
from src.data.make_dataset import load_data, filter_data, load_manila_data
from src.models.matching import create_matched_df
from src.models.train_model import create_pipeline, evaluation, get_model_coef, get_model_components
from src.visualization.summary import numerical_summary, categorical_summary
from src.visualization.visualize import plot_hist, plot_smd, get_favorite_job, shap_plot
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import warnings

warnings.filterwarnings('ignore')


def run_classification(X_train, X_test, y_train, y_test):
    # create pipeline
    pipeline = create_pipeline(X_train, y_train)
    model = pipeline.fit(X_train, y_train)
    search_space = {
            "classifier__subsample": [0.75, 1],
            "classifier__colsample_bytree": [0.75, 1],
            "classifier__max_depth": [2, 4, 6],
            "classifier__lambda": [0, 0.1, 1, 3],
            "classifier__alpha": [0, 0.1, 1, 3],
            # "classifier__min_child_weight": [1, 6],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__n_estimators": [50, 100, 200]}  # , 10, 50]}

    kfold = KFold(n_splits=3)
    from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score
    scoring = {'AUC': 'roc_auc',
               'Accuracy': make_scorer(accuracy_score),
               'Precision': make_scorer(precision_score),
               'Recall': make_scorer(recall_score)
               }
    clf = GridSearchCV(
        pipeline,
        param_grid=search_space,
        cv=kfold,
        scoring=scoring,
        refit='AUC',
        verbose=1,
        n_jobs=-1
    )

    model = clf.fit(X_train, y_train)

    # # print(f'model: {model}')
    # clf = GridSearchCV(
    #     model,
    #     parameters.get('XGB'),
    #     # random_state=RANDOM_STATE,
    #     scoring="roc_auc",
    #     #         n_iter=4,
    #     cv=3
    # )
    # search = clf.fit(X_train, y_train)
    print(model.cv_results_)
    pd.DataFrame(model.cv_results_).to_csv(f"output/{folder_name}/model/cv_results.csv")

    #
    # cv_results_log = pd.DataFrame(parameters.get("Logistic").cv_results_).sort_values("rank_test_score", ascending=True)

    # best model
    best_clf = clf.best_estimator_
    pipeline = best_clf

    get_model_coef(pipeline)
    # prediction
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    evaluation(y_test, y_pred, y_pred_proba[:, 1])

    model, feature_names = get_model_components(pipeline)
    if algo_name == 'xgboost':
        # Access the preprocessor from the pipeline
        preprocessor = pipeline.named_steps['preprocessor']

        # for dataset in [X_train, X_test]:
        transformed_x = preprocessor.transform(X_train)
        correlation_selection = pipeline.named_steps['correlation_selection']
        transformed_x = correlation_selection.transform(transformed_x)
        transformed_x = pd.DataFrame(transformed_x, columns=feature_names)
        shap_plot(model, transformed_x, name='all')


    return pipeline


def run(path):
    manila_data, _, hebrew_cols = load_manila_data(path)
    logger.info(manila_data.shape)

    data = load_data(path)
    logger.info(f'loaded data shape: {data.shape}')

    clean_data = filter_data(data)
    logger.info(f'clean_data.shape: {clean_data.shape}')
    # logger.info(f'clean_data columns: {clean_data.columns.to_list()}')

    manila_preferences_data = manila_data[hebrew_cols]
    favorite_job = get_favorite_job(manila_preferences_data.loc[clean_data.index], desc='_all')

    clean_data[categorical_features] = clean_data[categorical_features].astype(str)

    cols = [*categorical_features, *numerical_features, target_feature]
    logger.info(f"cols for model len: {len(cols)}")
    logger.info(f"cols for model: {cols}")

    df_for_model = clean_data[cols].copy()
    logger.info(f'df_for_model.shape: {df_for_model.shape}')

    # drop rows with na
    logger.info(df_for_model.isna().sum().head(10))
    df_for_model.dropna(inplace=True)
    logger.info(f'df.shape: {df_for_model.shape}')

    #
    y = df_for_model[target_feature].copy()
    X = df_for_model.loc[:, df_for_model.columns != target_feature].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    pipeline = run_classification(X_train, X_test, y_train, y_test)
    # propensity_score
    propensity_score = pipeline.predict_proba(X_train)[:, 1]
    X_train['propensity_score'] = propensity_score
    df_for_matching = df_for_model.loc[X_train.index]
    df_for_matching['propensity_score'] = propensity_score
    plot_hist(df_for_matching[df_for_matching[target_feature] == 0]['propensity_score'],
              df_for_matching[df_for_matching[target_feature] == 1]['propensity_score'], 'propensity_score overall')
    # matching
    df_balanced = create_matched_df(df_for_matching)
    print(f'df_balanced.columns: df_balanced.columns:')
    # create_model_plots(df_balanced, ['propensity_score'], desc='')
    # y = df_balanced[target_feature].copy()
    # X = df_balanced[df_balanced.columns.difference(['target', 'propensity_score', 'propensity_score_round'])]
    # # X = df_balanced.loc[:, df_for_model.columns != target_feature].copy()
    # print(f'x.columns: x.columns:')
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # preprocessor = pipeline.named_steps['preprocessor']
    # transformed_x = preprocessor.transform(X)
    # correlation_selection = pipeline.named_steps['correlation_selection']
    # transformed_x = correlation_selection.transform(transformed_x)
    # model, feature_names = get_model_components(pipeline)
    # transformed_x = pd.DataFrame(transformed_x, columns=feature_names)
    # shap_plot(model, transformed_x, name='cond')

    # run_classification(X_train, X_test, y_train, y_test)
    with_treatment_matched = df_balanced[df_balanced[target_feature] == 1]
    without_treatment_matched = df_balanced[df_balanced[target_feature] == 0]

    # summary
    summary_num_df = numerical_summary(numerical_features, with_treatment_matched, without_treatment_matched,
                                       df=df_for_matching)
    summary_num_df.to_csv(f'output/{folder_name}/matched_summary/summary_num_df.csv')
    # logger.info(summary_num_df)
    if len(categorical_features)>0:
        summary_cat_df = categorical_summary(categorical_features, with_treatment_matched, without_treatment_matched)
        summary_cat_df.to_csv(f'output/{folder_name}/matched_summary/summary_cat_df.csv')
    # logger.info(summary_cat_df)
    for col in with_treatment_matched.columns:
        # logger.info(f'plot column: {col}')
        plot_hist(without_treatment_matched[col], with_treatment_matched[col], f'matched_{col}')

    plot_smd(smd_scores=summary_num_df.set_index('attribute'), cols=["smd_global", "smd"],
             diff_name='smd')
    plot_smd(smd_scores=summary_num_df.set_index('attribute'),
             cols=["mean_difference_global", "mean_difference"], diff_name='difference')

    # logger.info(f'manila_data high all: {manila_data.loc[df_for_model[df_for_model[target_feature] == 1].index]}')
    favorite_job = get_favorite_job(manila_preferences_data.loc[with_treatment_matched.index], desc='_total_high')

    # logger.info(f'manila_data low all: {manila_data.loc[df_for_model[df_for_model[target_feature] == 0].index]}')
    favorite_job = get_favorite_job(manila_preferences_data.loc[with_treatment_matched.index], desc='_total_low')

    # logger.info(f'manila_data.loc[with_treatment_matched.index]: {manila_data.loc[with_treatment_matched.index]}')
    favorite_job = get_favorite_job(manila_preferences_data.loc[with_treatment_matched.index], desc='_matched_high')

    # logger.info(f'manila_data.loc[with_treatment_matched.index]: {manila_data.loc[with_treatment_matched.index]}')
    favorite_job = get_favorite_job(manila_preferences_data.loc[without_treatment_matched.index], desc='_matched_low')


if __name__ == '__main__':
    project_path = '/Users/netalorberbom/Library/CloudStorage/OneDrive-Payoneer/personal/msc/analytics_project'
    run(project_path)

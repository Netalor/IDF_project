from confs.conf import *
import pandas as pd
from src.data.make_dataset import load_data, filter_data, load_manila_data
from src.models.matching import create_matched_df
from src.models.train_model import create_pipeline, evaluation, get_model_coef, get_model_components
from src.visualization.summary import numerical_summary, categorical_summary
from src.visualization.visualize import create_model_plots, plot_hist, plot_smd, get_favorite_job, shap_plot
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def run_classification(X_train, X_test, y_train, y_test):
    # create pipeline
    pipeline = create_pipeline(X_train, y_train)
    pipeline.fit(X_train, y_train)
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


        transformed_x = preprocessor.transform(X_test[(y_test == 1) & (y_pred_proba[:, 1] < 0.5) ])
        correlation_selection = pipeline.named_steps['correlation_selection']
        transformed_x = correlation_selection.transform(transformed_x)
        transformed_x = pd.DataFrame(transformed_x, columns=feature_names)
        shap_plot(model, transformed_x, name='cond')
    return pipeline


def run(path):
    manila_data, _, hebrew_cols = load_manila_data(path)
    print(manila_data.shape)

    data = load_data(path)
    print(f'loaded data shape: {data.shape}')

    clean_data = filter_data(data)
    print(f'clean_data.shape: {clean_data.shape}')
    # print(f'clean_data columns: {clean_data.columns.to_list()}')

    manila_preferences_data = manila_data[hebrew_cols]
    favorite_job = get_favorite_job(manila_preferences_data.loc[clean_data.index], desc='_all')
    # print(favorite_job.head(100))

    clean_data[categorical_features] = clean_data[categorical_features].astype(str)

    # bagrut_features = []
    # likui_features = []
    # if include_bagrut:
    #     bagrut_features = clean_data.filter(regex='mpr_code').columns.to_list()
    # if include_likui:
    #     likui_features = clean_data.filter(regex='seif_likuy_group').columns.to_list()
    # print(f'bagrut_features len: {len(bagrut_features)}')
    # print(f'likui_features len: {len(likui_features)}')

    cols = categorical_features + numerical_features + [target_feature] #+ bagrut_features + likui_features
    print(f"cols for model len: {len(cols)}")
    print(f"cols for model: {cols}")

    df_for_model = clean_data[cols].copy()
    print(f'df_for_model.shape: {df_for_model.shape}')

    # drop rows with na
    print(df_for_model.isna().sum().head(10))
    df_for_model.dropna(inplace=True)
    print(f'df.shape: {df_for_model.shape}')

    #
    y = df_for_model[target_feature].copy()
    X = df_for_model.loc[:, df_for_model.columns != target_feature].copy()

    # print(f'y.shape: {y.shape}')
    #
    # # create pipeline
    # pipeline = create_pipeline(X, y)
    # pipeline.fit(X, y)
    # y_pred = pipeline.predict(X)
    # evaluation(y, y_pred)
    # get_model_coef(pipeline)
    #
    # model, feature_names = get_model_components(pipeline)
    # if algo_name == 'xgboost':
    #     # Access the preprocessor from the pipeline
    #     preprocessor = pipeline.named_steps['preprocessor']
    #     transformed_x = preprocessor.transform(X)
    #     transformed_x = pd.DataFrame(transformed_x, columns=feature_names)
    #     shap_plot(model, transformed_x)
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
    # create_model_plots(df_balanced, ['propensity_score'], desc='')

    with_treatment_matched = df_balanced[df_balanced[target_feature] == 1]
    without_treatment_matched = df_balanced[df_balanced[target_feature] == 0]

    # summary
    summary_num_df = numerical_summary(numerical_features, with_treatment_matched, without_treatment_matched,
                                       df=df_for_matching)
    summary_num_df.to_csv(f'output/{folder_name}/matched_summary/summary_num_df.csv')
    # print(summary_num_df)
    summary_cat_df = categorical_summary(categorical_features, with_treatment_matched, without_treatment_matched)
    summary_cat_df.to_csv(f'output/{folder_name}/matched_summary/summary_cat_df.csv')
    # print(summary_cat_df)
    for col in with_treatment_matched.columns:
            # print(f'plot column: {col}')
            plot_hist(without_treatment_matched[col], with_treatment_matched[col], f'matched_{col}')

    plot_smd(smd_scores=summary_num_df.set_index('attribute'), cols=["smd_global", "smd"],
             diff_name='smd')
    plot_smd(smd_scores=summary_num_df.set_index('attribute'),
             cols=["mean_difference_global", "mean_difference"], diff_name='difference')

    # print(f'manila_data high all: {manila_data.loc[df_for_model[df_for_model[target_feature] == 1].index]}')
    favorite_job = get_favorite_job(manila_preferences_data.loc[with_treatment_matched.index], desc='_total_high')
    # print(favorite_job.head(100))
    # print(f'manila_data low all: {manila_data.loc[df_for_model[df_for_model[target_feature] == 0].index]}')
    favorite_job = get_favorite_job(manila_preferences_data.loc[with_treatment_matched.index], desc='_total_low')
    # print(favorite_job.head(100))

    # print(f'manila_data.loc[with_treatment_matched.index]: {manila_data.loc[with_treatment_matched.index]}')
    favorite_job = get_favorite_job(manila_preferences_data.loc[with_treatment_matched.index], desc='_matched_high')
    # print(favorite_job.head(100))
    # print(f'manila_data.loc[with_treatment_matched.index]: {manila_data.loc[with_treatment_matched.index]}')
    favorite_job = get_favorite_job(manila_preferences_data.loc[without_treatment_matched.index], desc='_matched_low')
    # print(favorite_job.head(100))


if __name__ == '__main__':
    project_path = '/Users/netalorberbom/Library/CloudStorage/OneDrive-Payoneer/personal/msc/analytics_project'
    run(project_path)


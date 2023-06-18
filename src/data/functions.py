import pandas as pd

from confs.conf import logger


def show_nulls(df):
    null_columns = df.columns#[df.isnull().any()]
    null_count = df[null_columns].isnull().sum()
    total_count = len(df)

    null_ratio = null_count / total_count

    null_result = pd.concat([null_count, null_ratio], axis=1)
    null_result.columns = ['Null Count', 'Null Ratio']
    null_result = null_result.sort_values(by='Null Count', ascending=False)

    logger.info("Columns with null values and their corresponding ratios:")
    return null_result


def get_non_nan_cols(df, threshold=0.5):
    return df[df['Null Ratio'] < threshold].index
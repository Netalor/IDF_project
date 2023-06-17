from math import sqrt
import pandas as pd
import scipy.stats as stats
import numpy as np


def calc_standardization(treated_metric, untreated_metric) -> float:
    return sqrt(
        (
                (treated_metric.count() - 1) * treated_metric.std() ** 2 +
                (untreated_metric.count() - 1) * untreated_metric.std() ** 2) / (
                treated_metric.count() + untreated_metric.count() - 2)
    )


def numerical_summary(numeric_attributes: object, with_treatment_matched: object, without_treatment_matched: object, df: object) -> object:

    attributes_scores = pd.DataFrame(columns=[
        'attribute', 'count_high', 'mean_high', 'std_high', 'min_high', '25%_high', '50%_high', '75%_high', 'max_high',
        'count_low', 'mean_low', 'std_low', 'min_low', '25%_low', '50%_low', '75%_low', 'max_low', 'ks_statistic', 'p_value'
    ])

    for attribute in numeric_attributes:
        print(f"Describe: {attribute}")

        ks_statistic, p_value = stats.ks_2samp(with_treatment_matched[attribute], without_treatment_matched[attribute])
        treated_metric = with_treatment_matched[attribute]
        untreated_metric = without_treatment_matched[attribute]
        mean_difference: float = treated_metric.mean() - untreated_metric.mean()
        denominator: float = calc_standardization(treated_metric, untreated_metric)

        treated_global_metric = df[df['target'] == 1][attribute]
        untreated_global_metric = df[df['target'] == 0][attribute]
        mean_difference_global: float = treated_global_metric.mean() - untreated_global_metric.mean()
        denominator_global: float = calc_standardization(treated_global_metric, untreated_global_metric)
        attributes_scores = pd.concat([
            attributes_scores,
            pd.DataFrame([{
                'attribute': attribute,
                **{f'{key}_high': value for key, value in
                   with_treatment_matched[attribute].describe().to_dict().items()},
                **{f'{key}_low': value for key, value in
                   without_treatment_matched[attribute].describe().to_dict().items()},
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                "smd": round(mean_difference / denominator, 3),
                "smd_global": round(mean_difference_global / denominator_global, 3),
                "mean_difference": round(mean_difference, 3),
                "mean_difference_global": round(mean_difference_global, 3)
            }])
        ], ignore_index=True)
        # attributes_scores = attributes_scores.append({
        #     'attribute': attribute,
        #     **{f'{key}_high': value for key, value in with_treatment_matched[attribute].describe().to_dict().items()},
        #     **{f'{key}_low': value for key, value in without_treatment_matched[attribute].describe().to_dict().items()},
        #     'ks_statistic': ks_statistic,
        #     'p_value': p_value,
        #     "smd": round(mean_difference / denominator, 3),
        #     "smd_global": round(mean_difference_global / denominator_global, 3),
        #     "mean_difference": round(mean_difference, 3),
        #     "mean_difference_global": round(mean_difference_global, 3)
        # }, ignore_index=True)
    return attributes_scores.sort_values(by='p_value', ascending=False).round(2)


def chi2_test(data1, data2):
    # Create DataFrames for each sample
    df1 = pd.DataFrame(data1, columns=['Category'])
    df2 = pd.DataFrame(data2, columns=['Category'])

    # Compute category frequencies for each sample
    freq1 = df1['Category'].value_counts().sort_index()
    freq2 = df2['Category'].value_counts().sort_index()

    # Create a contingency table
    contingency_table = pd.concat([freq1, freq2], axis=1).fillna(0)
    contingency_table.columns = ['Sample1', 'Sample2']

    # Perform the Chi-squared test
    chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)

    return chi2_stat, p_value


def categorical_summary(categorical_attributes, with_treatment_matched, without_treatment_matched):
    attributes_scores = pd.DataFrame(columns=[
        'attribute', 'count_high', 'unique_high', 'top_high', 'freq_high', 'count_low', 'unique_low', 'top_low', 'freq_low',  'chi2', 'p_value'
    ])

    for attribute in categorical_attributes:
        print(f"Describe: {attribute}")
        # Perform the Chi-squared test
        chi2_stat, p_value = chi2_test(np.array(with_treatment_matched[attribute].astype(str)), np.array(without_treatment_matched[attribute].astype(str)))

        print("Chi-squared statistic:", chi2_stat)
        print("P-value:", p_value)

        attributes_scores = attributes_scores.append({
            'attribute': attribute,
            **{f'{key}_high': value for key, value in with_treatment_matched[attribute].describe().to_dict().items()},
            **{f'{key}_low': value for key, value in without_treatment_matched[attribute].describe().to_dict().items()},
            'chi2': chi2_stat,
            'p_value': p_value
        }, ignore_index=True)
    print('\n'.join([f'{metric} : {score}' for metric, score in attributes_scores.items()]))

    return attributes_scores.sort_values(by='p_value', ascending=False).round(2)
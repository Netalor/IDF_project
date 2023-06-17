import pandas as pd
from confs.conf import *

from src.data.functions import get_non_nan_cols, show_nulls


def get_df_desc(org_df):
    df_desc_ = org_df.describe(include='all')
    df_desc_.loc['dtype'] = org_df.dtypes
    df_desc_.loc['size'] = len(org_df)
    df_desc_.loc['% null'] = org_df.isnull().mean()
    df_desc_.loc['null'] = org_df.isnull().sum()
    return df_desc_


def get_likui_df(df):
    seif_likuy_df = pd.DataFrame()
    seif_likuy_df[f'mispar_ishi'] = df['mispar_ishi']
    sl_cols = df.filter(regex='seif_likuy').columns

    pp = pd.DataFrame(columns=['severity', 'seif_likuy_group'])
    for i in range(1, 11):
        seif_likuy_df[f'severity{i:02}'] = df[f'seif_likuy{i:02}'].fillna(0).astype(int).astype(str).str[-1]
        # seif_likuy_df['section'] = df[f'seif_likuy0{i}'].fillna(0).astype(int).astype(str).str[-2]
        # seif_likuy_df['version'] = df[f'seif_likuy0{i}'].fillna(0).astype(int).astype(str).str[-3]
        seif_likuy_df[f'group{i:02}'] = df[f'seif_likuy{i:02}'].fillna(0).astype(int).astype(str).str[:-3]

        pp = pd.concat([
            pp, pd.DataFrame(seif_likuy_df[['mispar_ishi', f'severity{i:02}', f'group{i:02}']].values,
                             columns=['mispar_ishi', 'severity', 'seif_likuy_group'])
        ], axis=0)

    sl_df = pp.pivot_table(index='mispar_ishi', columns='seif_likuy_group', values='severity')
    print(f'sl_df.shape: {sl_df.shape}')
    print(f'sl_df cols: {sl_df.columns}')
    sl_df.columns = [f'seif_likuy_group_{col}' for col in sl_df.columns]
    print(f'sl_df cols: {sl_df.columns}')
    sl_df['seif_likuy_max_severity'] = sl_df.filter(regex='seif_likuy_group').max(axis=1)
    sl_df['seif_likuy_sum_severity'] = sl_df.filter(regex='seif_likuy_group').sum(axis=1)
    sl_df['seif_likuy_total'] = sl_df.filter(regex='seif_likuy_group').count(axis=1)
    print(f'sl_df cols: {sl_df.columns}')
    sl_df.drop('seif_likuy_group_', axis=1, inplace=True)
    return sl_df, sl_cols


def get_bagrut_df(df):
    bagrut_cols = df.filter(regex='mprofesscode|unit').columns

    pp = pd.DataFrame(columns=['mispar_ishi', 'units', 'mpr'])
    for i in range(1, 41):
        pp = pd.concat([
            pp, pd.DataFrame(df[['mispar_ishi', f'units{i:02}', f'mprofesscode{i:02}']].values,
                             columns=['mispar_ishi', 'units', 'mpr'])
        ], axis=0)
    bagrut_df = pp.pivot_table(index='mispar_ishi', columns='mpr', values='units')
    print(f'bagrut_df.shape: {bagrut_df.shape}')
    print(f'bagrut_df cols: {bagrut_df.columns}')
    bagrut_df.columns = [f'mpr_code_{col}' for col in bagrut_df.columns]
    print(f'bagrut_df cols: {bagrut_df.columns}')
    bagrut_df['bagrut_max_units'] = bagrut_df.filter(regex='mpr_code').max(axis=1)
    bagrut_df['bagrut_sum_units'] = bagrut_df.filter(regex='mpr_code').sum(axis=1)
    bagrut_df['bagrut_total'] = bagrut_df.filter(regex='mpr_code').count(axis=1)

    return bagrut_df, bagrut_cols


def load_manila_data(path):
    manila_data = pd.read_csv(f'{path}/data/manila_data/manila_data.csv', encoding='ISO-8859-8')
    manila_data_desc = get_df_desc(manila_data)
    print(f'manila_data.shape: {manila_data.shape}')
    manila_data.columns = list(map(str.lower, manila_data.columns))

    com_data = manila_data.set_index('mispar_ishi').filter(regex="המחשב")
    # adding target variable
    manila_data['target'] = com_data[target_name].apply(classify)
    # separate manila cols and other
    hebrew_cols = manila_data.filter(regex="[\u0590-\u05FF\uFB1D-\uFB4F]+").columns.to_list()
    return manila_data, com_data, hebrew_cols


def load_students_data(path):
    students_data = pd.read_csv(f'{path}/data/manila_data/more_data_for_students.csv', encoding='ISO-8859-8')
    students_data_desc = get_df_desc(students_data)
    print(f'students_data.shape: {students_data.shape}')
    students_data.columns = map(str.lower, students_data.columns)
    students_data.sample(1)

    return students_data


def load_stat_socio(path):
    stat_socio = pd.read_csv(f'{path}/data/manila_data/stat_socio.csv', encoding='ISO-8859-8')
    stat_socio_desc = get_df_desc(stat_socio)
    print(f'stat_socio.shape: {stat_socio.shape}')
    stat_socio.columns = map(str.lower, stat_socio.columns)
    # stat_socio.set_index('encoded_mi', inplace=True)
    stat_socio.sample(1)

    return stat_socio


def load_yeshuv_napa(path):
    yeshuv_napa = pd.read_csv(f'{path}/data/manila_data/yeshuv_napa.csv', encoding='ISO-8859-8')
    yeshuv_napa_desc = get_df_desc(yeshuv_napa)
    print(f'yeshuv_napa.shape: {yeshuv_napa.shape}')
    yeshuv_napa.columns = ['city_code', 'napa_code', 'city_sotzio']
    yeshuv_napa.sample(1)

    return yeshuv_napa


def filter_df_by_target_cond(df: pd.DataFrame, name):
    print(name)
    # y = 'אשכול תפקידי מקצועות המחשב_חיילת מקצועות המחשוב'
    if name == 'mihshuv':
        filtered_data = df[(df['profil'] >= 45) &
                           (((df['mea_svivat_afaala'] >= 3) & (df['mea_svivat_ibud'] >= 4))
                            | ((df['mea_svivat_afaala'] >= 4) & (df['mea_svivat_ibud'] >= 3)))]

    # y = 'אשכול תפקידי מקצועות המחשב_חיילת מקצועות התקשוב'
    elif name == 'tikshuv':
        filtered_data = df[df['profil'] >= 64]

   # y = 'אשכול תפקידי מקצועות המחשב_אשכול מקצועות המחשב'
    elif name == 'mahshev':
        filtered_data = df[df['dapar'] >= 60]
    else:
        print('no condition for y - folder name')
        filtered_data = None
        exit()
    print(f'df.shape before filtering: {df.shape}')
    print(f'df_filtered.shape: {filtered_data.shape}')
    print(f'ratio of the data after filtering: {filtered_data.shape[0] / df.shape[0]}')

    return filtered_data


def get_math_5_in_city(data):
    math_pop = data.pivot_table(index='city_code', columns='yehidot_math', values='mispar_ishi', aggfunc='count').fillna(0)
    math_pop['math_5_ration'] = ((math_pop[5] + math_pop[10]) / math_pop.sum(axis=1)).fillna(0)
    data = data.merge(math_pop[['math_5_ration']], how='left', left_on='city_code', right_index=True)
    return data


def classify(x):
    if pd.isna(x):
        return 0
    elif x in [4.0, 5.0]:
        return 1
    else:
        return 0


def load_data(path):
    # load csv data
    manila_data, com_data, hebrew_cols = load_manila_data(path)
    students_data = load_students_data(path)
    stat_socio = load_stat_socio(path)
    yeshuv_napa = load_yeshuv_napa(path)

    df = manila_data.drop(hebrew_cols, axis=1)
    print(f'df shape: {df.shape}')

    # feature engineering for bagraut data
    if include_bagrut:
        bagrut_df, bagrut_cols = get_bagrut_df(df)
        print(f'dbagrut_df shape: {bagrut_df.shape}')

    if include_likui:
        likui_df, likui_cols = get_likui_df(df)
        print(f'likui_df shape: {likui_df.shape}')

    # merge into one dataset
    df_merged = df.join(students_data.set_index('encoded_mi'), how='left')
    print(f'df+students_data shape: {df_merged.shape}')
    # df_merged['city_code']
    df_merged = df_merged.join(stat_socio.set_index('encoded_mi'), how='left')
    print(f'df+students_data+stat_socio shape: {df_merged.shape}')

    df_merged = df_merged.merge(yeshuv_napa, how='left', left_on='yeshuv_code', right_on='city_code')
    print(f'df+students_data+stat_socio+yeshuv_napa shape: {df_merged.shape}')
    # data['city_code']
    if include_bagrut:
        df_merged = df_merged.merge(bagrut_df, how='left', left_index=True, right_index=True)
        print(f'df+students_data+stat_socio+yeshuv_napa+bagrut_df shape: {df_merged.shape}')

    if include_likui:
        df_merged = df_merged.merge(likui_df, how='left', left_index=True, right_index=True)
        print(f'df+students_data+stat_socio+yeshuv_napa+bagrut_df+likui_df shape: {df_merged.shape}')
    print(f'df_merged.columns: {df_merged.columns.to_list()}')

    # adding get_math_5_in_city
    df_merged = get_math_5_in_city(df_merged)
    print(f'df_merged.columns: {df_merged.columns.to_list()}')
    return df_merged


def filter_data(df_merged, threshold=0.5):
    # filter data by the target population
    df_filtered = filter_df_by_target_cond(df_merged, folder_name)
    print(f'df_filtered.shape: {df_filtered.shape}')

    # drop nan cols
    null_data_results = show_nulls(df_filtered)
    null_data_results.to_csv(f"output/{folder_name}/analytics/na_columns.csv")
    print(f'null_data_results: {null_data_results.head(10)}')

    not_nan_columns = get_non_nan_cols(null_data_results, threshold)
    print(f'len of not_nan_columns: {len(not_nan_columns)}')

    df_clean = df_filtered[not_nan_columns]
    print(f'df_clean.shape: {df_clean.shape}')
    # remove na
    print(f'df_clean.shape: {df_clean.shape}')
    return df_clean


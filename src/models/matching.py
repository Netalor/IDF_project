from numpy import sqrt, absolute
from tqdm import tqdm
from pandas import DataFrame, Series


def create_matched_df(df):
    # fuzzy matching by rounding
    df["propensity_score_round"] = df["propensity_score"].round(2)
    df_balanced = DataFrame()
    for score, df_score in tqdm(df.groupby("propensity_score_round"), desc="match scores"):
        same_score_cnt = df_score.groupby(
            "target")["propensity_score_round"].count().min()
        df_balanced = df_balanced.append(
            df_score.groupby("target").sample(same_score_cnt),
        )
    print(
        "balanced df shape\n",
        df_balanced.groupby("target")["propensity_score_round"].count()
    )
    return df_balanced


def calc_smd(df):
    feature_keep = df.columns.drop(["const", "formatted_credit_card_company_Visa"]).tolist()
    visa = df.query("formatted_credit_card_company_Visa == 1")[feature_keep]
    mastercard = df.query("formatted_credit_card_company_Visa == 0")[feature_keep]
    smd = (
            visa.values.mean(axis=0) - mastercard.values.mean(axis=0)
    ) / sqrt(
            (visa.values.var(axis=0) + mastercard.values.var(axis=0)) / 2
    )
    return Series(absolute(smd), index=feature_keep)
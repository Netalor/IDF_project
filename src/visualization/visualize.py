
import plotly.graph_objects as go
import shap
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from confs.conf import *

color_scheme = {"1": "#6473ff", "0": "#217883"}


# def create_model_plots(df, feature_plots, desc=''):
#     for feature_plot in feature_plots:
#         logger.info(feature_plot)
#         fig = go.Figure()
#         for card_name, card_df in df.groupby("target"):
#             logger.info(card_name)
#             fig.add_trace(
#                 go.Histogram(
#                     x=card_df[feature_plot],
#                     name=card_name,
#                     marker_color=color_scheme.get(1)
#                 )
#             )
#         # Overlay both histograms
#         fig.update_layout(
#             barmode='overlay',
#             title_text=f"{desc} {feature_plot.replace('_', ' ')}",
#         )
#         # Reduce opacity to see both histograms
#         fig.update_traces(opacity=0.75)
#         fig.write_image(f"output/{folder_name}/analytics/{feature_plot}_histogram.png")
#         if show_plot:
#             fig.show()

def get_favorite_job(manila_data, desc=''):
    # favorite_job = manila_data.idxmax(axis=1).rename('max_job').to_frame()
    # Find column names with maximum values for each row
    max_columns = manila_data.apply(lambda row: row.index[row == row.max()].tolist(), axis=1).rename('max_job')

    # Explode the list of column names into multiple rows
    favorite_job = max_columns.explode().reset_index()
    favorite_job_agg = favorite_job.groupby('max_job').agg(total_students=('index', 'count'))
    favorite_job_agg['mean_votes'] = (favorite_job_agg['total_students']/favorite_job_agg['total_students'].sum())\
        .round(2)
    favorite_job_agg['ratio'] = (favorite_job_agg['total_students']/manila_data.shape[0]).round(2)

    favorite_job_agg.sort_values(by='ratio', ascending=False).to_csv(f"output/{folder_name}/analytics/favorite_job{desc}.csv")
    return favorite_job_agg.sort_values(by='ratio', ascending=False)


def statistic_plot(manila_data):
    agg_student_statistic = manila_data.T.describe().T
    agg_student_statistic.to_csv(f"output/{folder_name}/analytics/agg_student_statistic.csv")
    # agg_student_statistic['mean'].hist()
    # plt.savefig(f"output/{folder_name}/analytics/mean.png")
    # manila_data.apply(lambda x: x.mean(), axis=0).hist()
    # plt.savefig(f"output/{folder_name}/analytics/mean.png")


def plot_hist(x0, x1, col):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x0, name=f'low_rating'))
    fig.add_trace(go.Histogram(x=x1, name=f'high_rating'))
    # Overlay both histograms
    fig.update_layout(barmode='overlay', title_text=f"histogram for {col} ")
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.5)
    fig.write_image(f"output/{folder_name}/plots/hist_{col}.png")
    if show_plot:
        fig.show()


def plot_smd(smd_scores, cols=["smd_global", "smd"], diff_name='smd'):

    fig = make_subplots(
        rows=len(smd_scores.index),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02
    )

    for row, feature in enumerate(smd_scores.index, 1):
        show_legend = True if row == 1 else False
        fig.add_trace(
            go.Scatter(
                x=smd_scores.loc[feature, cols], #,["mean_difference_global", "mean_difference"]],   #["smd_global", "smd"]
                y=[feature, feature],
                mode='lines+markers',
                showlegend=show_legend,
                name="unmatched",
                line=dict(color="#217883"),
                marker=dict(
                    size=[20, 20],
                    color=["#27c1d1", "#217883"]
                )

            ),
            row=row,
            col=1
        )
    fig.update_layout(
        height=1000, width=1200,
        title_text="differences for propensity score matching"
    )
    fig.update_traces(textposition="bottom right")
    fig.write_image(f"output/{folder_name}/plots/{diff_name}_differences.png")
    if show_plot:
        fig.show()


def shap_plot(clf, x, name):
    import matplotlib.pyplot as plt

    shap.initjs()
    explainer = shap.TreeExplainer(clf)
    shap.summary_plot(explainer.shap_values(x), x, show=False)
    plt.savefig(f"output/{folder_name}/model/shap_summary_{name}.png")

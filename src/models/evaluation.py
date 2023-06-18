
from typing import Union
from pandas import Series
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px


def get_array_only(arr: Union[np.array, Series]):
    return arr.values if isinstance(arr, Series) else arr


class Evaluation:
    """
    Evaluation
    Input: y, pred
    Output: auc, accuracy, precision, recall, f1, plots
    """

    def __init__(
            self,
            y: Union[np.array, Series],
            proba: Union[np.array, Series],
            threshold: float = 0.5,
            output_path: str = None
    ):

        self.y = get_array_only(y)
        self.proba = get_array_only(proba)
        self.threshold_ = threshold
        self.pred_ = np.where(proba > self.threshold_, 1, 0)
        self.output_path = output_path

    @property
    def threshold(self):
        return self.threshold_

    @threshold.setter
    def threshold(self, value):
        self.threshold_ = value

    @property
    def pred(self):
        return self.pred_

    def get_eval_metrics(self):
        eval_metrics = {
            'threshold': self.threshold,
            'auc': round(roc_auc_score(self.y, self.proba), 3),
            'accuracy_score': accuracy_score(y_true=self.y, y_pred=self.pred),
            'f1_score': f1_score(y_true=self.y, y_pred=self.pred),
            'precision_score': precision_score(y_true=self.y, y_pred=self.pred),
            'recall_score': recall_score(y_true=self.y, y_pred=self.pred)
        }

        return eval_metrics

    def confusion_matrix_curve(self,
                               count=True,
                               percent=True,
                               cbar=True,
                               xyticks=True,
                               xyplotlabels=True,
                               sum_stats=True,
                               figsize=None,
                               cmap='Blues',
                               title=None):
        """
        This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
        Arguments
        ---------
        count:         If True, show the raw number in the confusion matrix. Default is True.
        cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                       Default is True.
        xyticks:       If True, show x and y ticks. Default is True.
        xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
        sum_stats:     If True, display summary statistics below the figure. Default is True.
        figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
        cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                       See http://matplotlib.org/examples/color/colormaps_reference.html

        title:         Title for the heatmap. Default is None.
        """
        cf = confusion_matrix(self.y, self.pred)
        categories = ['Female', 'Male']
        group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        # CODE TO GENERATE TEXT INSIDE EACH SQUARE
        blanks = ['' for i in range(cf.size)]

        if group_names and len(group_names) == cf.size:
            group_labels = ["{}\n".format(value) for value in group_names]
        else:
            group_labels = blanks

        if count:
            group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        else:
            group_counts = blanks

        if percent:
            group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
        else:
            group_percentages = blanks

        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:
            # Accuracy is sum of diagonal divided by total observations
            accuracy = np.trace(cf) / float(np.sum(cf))
            print(f'cf: {cf}')
            print(f'len cf: {len(cf)}')
            # if it is a binary confusion matrix, show some more stats
            if len(cf) == 2:
                print('len')
                # Metrics for Binary Confusion Matrices
                auc = round(roc_auc_score(self.y, self.proba), 3)
                accuracy = accuracy_score(y_true=self.y, y_pred=self.pred)
                f1 = f1_score(y_true=self.y, y_pred=self.pred)
                precision = precision_score(y_true=self.y, y_pred=self.pred)
                recall = recall_score(y_true=self.y, y_pred=self.pred)

                # auc = round(roc_auc_score(self.y, self.proba), 3)
                # precision = cf[1, 1] / sum(cf[:, 1])
                # recall = cf[1, 1] / sum(cf[1, :])
                # f1_score = 2 * precision * recall / (precision + recall)
                stats_text = "\n\nAccuracy={:0.3f}\nAUC={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                    accuracy, auc, precision, recall, f1)
                print(f'stats_text 1: {stats_text}')
            else:
                stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
        else:
            stats_text = ""
        print(f'stats_text: {stats_text}')
        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if figsize == None:
            # Get default figure size if not set
            figsize = plt.rcParams.get('figure.figsize')

        if xyticks == False:
            # Do not show categories if xyticks is False
            categories = False

        # MAKE THE HEATMAP VISUALIZATION
        plt.figure(clear=True, figsize=figsize)
        sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

        if xyplotlabels:
            plt.ylabel('True label')
            plt.xlabel('Predicted label' + stats_text)
        else:
            plt.xlabel(stats_text)

        if title:
            plt.title(title)
        # if self.output_path is None:
        #
        if self.output_path is not None:
            os.makedirs(f'{self.output_path}/', exist_ok=True)
            plt.savefig(f'{self.output_path}/recall_precision_curve.png')
        else:
            plt.show()
        return plt

    def confusion_matrix_curve_plotly(self, normalize=None):
        conf_data = confusion_matrix(y_true=self.y,  y_pred=self.pred, normalize=normalize)
        fig = px.imshow(conf_data,
                        text_auto=True,
                        labels=dict(x="True label", y="Prediction", color="Productivity"),
                        x=['Negative', 'Positive'],
                        y=['Negative', 'Positive', ],
                        color_continuous_scale=px.colors.sequential.BuGn
                        )
        fig.update_xaxes(side="top")
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(textfont_size=65)
        fig.update_xaxes(title_font=dict(size=30))
        fig.update_yaxes(title_font=dict(size=30))

        if self.output_path is not None:
            os.makedirs(f'{self.output_path}/', exist_ok=True)
            fig.write_html(f'{self.output_path}/confusion_matrix.html')
        else:
            return fig

    def recall_precision_curve(self):
        precision, recall, thresholds = precision_recall_curve(self.y, self.proba)
        thresholds = np.append(thresholds, 1)

        f1_score = 2 * (recall * precision) / (recall + precision)

        fig = go.Figure()
        # Create and style traces
        fig.add_trace(go.Scatter(x=thresholds, y=precision, name='precision',
                                 line=dict(color='royalblue', width=4)))
        fig.add_trace(go.Scatter(x=thresholds, y=recall, name='recall',
                                 line=dict(color='gold', width=4)))
        fig.add_trace(go.Scatter(x=thresholds, y=f1_score, name='f1_score',
                                 line=dict(color='firebrick', width=4)))

        fig.add_vline(x=self.threshold, line_width=3, line_dash="dash", line_color="green", name="threshold")
        # Edit the layout
        fig.update_layout(
            xaxis_title='Threshold',
            yaxis_title='%')
        fig.update_layout(
            width=500,
            height=400
        )
        if self.output_path is not None:
            os.makedirs(f'{self.output_path}/', exist_ok=True)
            fig.write_html(f'{self.output_path}/recall_precision_curve.html')
        else:
            return fig

    def prediction_distribution_curve(self):
        fig = go.Figure()
        proba_np = np.asarray(self.proba)
        y_np = np.asarray(self.y)

        fig.add_trace(go.Histogram(x=proba_np[y_np == 1], name="positive"))
        fig.add_trace(go.Histogram(x=proba_np[y_np == 0], name="negative"))

        fig.add_trace(
            go.Scatter(
                x=[self.threshold, self.threshold],
                y=[0, 50],
                mode='lines', name='lines')
        )

        fig.update_layout(barmode='overlay')
        fig.update_traces(opacity=0.75)
        if self.output_path is not None:
            os.makedirs(f'{self.output_path}/', exist_ok=True)
            fig.write_html(f'{self.output_path}/prediction_distribution_curve.html')
        else:
            return fig

    def roc_curve(self):
        plt.figure(clear=True)
        fpr, tpr, _ = roc_curve(self.y, self.proba)
        auc = round(roc_auc_score(self.y, self.proba), 3)

        plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        plt.legend(loc=4)

        plt.plot([0, 1],  # Plotting a diagonal with AUC = 0.5
                 [0, 1],
                 linestyle='--',
                 lw=2, color='r',
                 )
        if self.output_path is not None:
            os.makedirs(f'{self.output_path}/', exist_ok=True)
            plt.savefig(f'{self.output_path}/roc_curve.png')
        else:
            plt.show()

    def eval_classification(self):
        self.confusion_matrix_curve()
        self.recall_precision_curve()
        self.prediction_distribution_curve()
        self.roc_curve()
        return self.get_eval_metrics()

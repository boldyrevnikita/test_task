import pandas as pd
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import qmc, norm
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import linearmodels
from linearmodels import PooledOLS
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.special import expit


class SigmoidPooledOLS:
    def __init__(self, function):
        self.model = None
        self.sigmoid_params = None
        self.function = function

    def fit(self, X, y, **kwargs):
        y_transform = np.log(y + 0.0001 / (1 - y + 0.0001))

        # Обучение модели PooledOLS
        self.model = self.function(dependent=y_transform, exog=X, **kwargs).fit()

    def predict(self, X):
        # Делаем предсказания с помощью PooledOLS
        predictions = self.model.predict(X)

        # Применение сигмоидальной функции
        sigmoid_predictions = expit(predictions)

        return sigmoid_predictions

    def evaluate(self, y_true, y_pred):
        roc_auc = roc_auc_score(y_true, y_pred)

        # plot ROC AUC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

        # return ROC AUC score 
        return roc_auc_score(y_true, y_pred)
    
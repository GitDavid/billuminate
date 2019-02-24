import os
from gensim import summarization
from lxml import etree
import re
import pandas as pd
import numpy as np
import sys
sys.path.append('../../')
from data_preparation import feature_utils,  bill_utils

MODEL_ROOT = '../../../models/'
NLP_MODEL_ROOT = '../../../nlp_models/'


def read_time(word_count, wpm=200):
    return np.divide(word_count, wpm).round(decimals=2)


def apply_read_time(X):

    # estimate words per minute for each line
    X['read_time'] = read_time(X['word_count'], wpm=200)
    X['predict_ranking'] = X['predict_proba1'].rank(ascending=False).astype(int)

    # Calculate the cumulative reading time with sentences ranked using predict_proba
    sum_ser = X.sort_values(by='predict_proba1',
                            ascending=False)['read_time'].cumsum()
    sum_ser.name = 'time_cumulative'

    X = pd.merge(X, pd.DataFrame(sum_ser), left_index=True, right_index=True)

    return X
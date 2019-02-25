import os
from gensim import summarization
from lxml import etree
import re
import pandas as pd
import numpy as np
import sys
sys.path.append('../../')
from data_preparation import bill_utils

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

def create_read_time_slider(X, read_time=None):
    read_time_slider = {'min': 0.5,
                       'max': np.ceil(X['time_cumulative'].max()).astype(int),
                       'current': read_time}
    if read_time_slider['max'] == read_time_slider['min']:
        read_time_slider['max'] += 0.5
    if not read_time_slider['current']:
        read_time_slider['current'] = X[X.prediction == 1]['time_cumulative'].max()
    read_time_slider['current'] = int(np.ceil(float(read_time_slider['current'])))

    return read_time_slider
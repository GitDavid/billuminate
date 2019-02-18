from gensim import summarization
from lxml import etree
import re
import pandas as pd
import numpy as np
import sys
sys.path.append('/../../src/')

from data_preparation import feature_utils,  bill_utils

MODEL_ROOT = '../../models/'
NLP_MODEL_ROOT = '../../nlp_models/'


def create_single_text_string(tree, tag='text'):
    text = ""
    for elt in tree.getiterator(tag):
        if isinstance(elt.text, str):
            text += elt.text + ' '
    return text


def do_summarization(string_xml):
    string_length = len(string_xml)
    print('text length: {}'.format(string_length))
    tree = etree.fromstring(string_xml)

    text = create_single_text_string(tree, tag='text')

    summarized = summarization.summarize(text)
    print('summary length: {}'.format(len(summarized)))

    return summarized


def read_time(word_count, wpm=200):
    return np.divide(word_count, wpm)

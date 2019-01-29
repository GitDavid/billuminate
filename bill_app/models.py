import sqlalchemy
import psycopg2
import os
import numpy 
import ast
import pandas as pd
import re
from lxml import etree
import numpy as np
from gensim.summarization import summarize, keywords


def do_summarization(string_xml):
    string_length = len(string_xml)
    print(string_length)
    tree = etree.fromstring(string_xml)

    text = ""
    for elt in tree.getiterator('text'):
        if isinstance(elt.text, str):
            text += elt.text + ' '

    summarized = summarize(text)
    print(len(summarized))

    return summarized
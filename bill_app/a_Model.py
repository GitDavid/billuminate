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



def ModelIt(fromUser='Default',
            bills=[]):
    in_month = len(bills)
    print('The number bills is %i' % in_month)
    print('Here is a summary of a random bill')

    string_tree = bills[10]
    tree = etree.fromstring(string_tree)

    text = ""
    for elt in tree.getiterator('text'):
        if isinstance(elt.text, str):
            text += elt.text + ' '

    summarized = summarize(text)
    print(summarized)
    result = in_month
    if fromUser != 'Default':
        return summarized
    else:
        return 'check your input'

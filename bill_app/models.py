import numpy as np
import pandas as pd
import re
from lxml import etree
import numpy as np
from gensim import summarization


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

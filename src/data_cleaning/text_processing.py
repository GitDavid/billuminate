# import matplotlib
# import matplotlib.pylab as plt

import random
import re
import string
import xml.etree.ElementTree as ET

import numpy as np

import pandas as pd
# import sqlalchemy_utils
import psycopg2
import spacy
import sqlalchemy
from sklearn import metrics


def _is_not_empty(maybe_text):
    if maybe_text and not maybe_text.isspace():
        return True


def _clean_extracted_list(txt_extract,
                          unwanted_tags=['external-xref',
                                         'after-quoted-block']):
    txt_extract = pd.DataFrame(txt_extract)
    txt_extract.columns = ['loc_ix', 'tag', 'text']

    first_ix = txt_extract.index[txt_extract['tag'] ==
                                 'short-title'].tolist()[0]
    try:
        last_ix = txt_extract.index[txt_extract['tag'] ==
                                    'attestation-date'].tolist()[0] - 1
    except:
        last_ix = txt_extract.index[txt_extract['tag'] == 'text'].tolist()[-1]
    else:
        last_ix = -1
    txt_extract = txt_extract.loc[first_ix:last_ix]

    return txt_extract[~txt_extract['tag'].isin(unwanted_tags)]


def _full_text_tostring(txt_extract, omit_tags=['header', 'short-title']):
    sentences = []
    enum_string = ""
    full_string = ""
    for ix, row in txt_extract[~txt_extract['tag'].isin(omit_tags)].iterrows():
        nextix = ix
        enum_string += row['text'] + ' '
        if row['tag'] == 'enum':
            nextix = ix + 1
        if nextix == ix:
            sentences.append(row['text'] + ' ')
            full_string += row['text'] + ' '
    return full_string, enum_string, sentences


def _bill_from_xml(xml_text_string):
    txt_tree = ET.ElementTree(ET.fromstring(xml_text_string))
    txt_root = txt_tree.getroot()

    # txt_descendants = list(txt_root.iter())
    txt_extract = [[ix, elem.tag, elem.text] for ix, elem
                   in enumerate(txt_root.iter())
                   if _is_not_empty(elem.text)]
    # and elem.tag in ['enum', 'header', 'text']]

    txt_extract = _clean_extracted_list(txt_extract)
    xml_output = _full_text_tostring(txt_extract)
    full_string, enum_string, extract_sentences = xml_output
    return full_string, enum_string, extract_sentences


def _tokenize_sentences(txt_string, nlp):
    doc = nlp(txt_string)
    txt_sent = [sent.string.strip() for sent in doc.sents]
    return doc, txt_sent


def _remove_whitespace(sentence_list):
    white_space = list(string.whitespace)[1:]
    for ix in range(len(sentence_list)):
        for bad_string in white_space:
            if bad_string in sentence_list[ix]:
                sentence_list[ix] = sentence_list[ix].replace(bad_string, "")
    return sentence_list


def _extract_entities(sentences, nlp, bad_ents=['WORK_OF_ART', 'CARDINAL']):
    df_ent = pd.DataFrame()
    for ix, sentence in enumerate(sentences):
        doc = nlp(sentence)
        for ent in doc.ents:

            df_ent = df_ent.append([[ix, ent.text, ent.start_char,
                                     ent.end_char, ent.label_]],
                                   ignore_index=True)

    df_ent.columns = ['sentence', 'text', 'start_char', 'end_char', 'label']
    df_ent = df_ent[(~df_ent.label.isin(bad_ents)) &
                    (~df_ent['text'].apply(str.isspace))]
    return df_ent


def _extract_embeddings(
        path_to_embedding='../nlp_models/glove.6B/glove.6B.300d.txt'):
    f = open(path_to_embedding, encoding='utf-8')
    word_embeddings = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    embedding_size = list(coefs.shape)[0]
    return word_embeddings, embedding_size


def _return_embeddings(full_sent):
    df_ent = _extract_entities(full_sent)
    return df_ent


def _calc_embedding(sen, word_embeddings, embedding_size):
    if embedding_size is None:
        embedding_size = random.choice(list(word_embeddings.values())).shape
    if len(sen) != 0:
        vector = sum([word_embeddings.get(w, np.zeros(embedding_size))
                      for w in sen.split()])/(len(sen.split())+0.001)
    else:
        vector = np.zeros(embedding_size)
    return vector


def _remove_punct_nums(sentences):
    # remove punctuations, numbers and special characters
    regex = re.compile(r"[^a-zA-Z]")
    return [regex.sub(" ", s) for s in sentences]


def _make_lowercase(sentences):
    return [s.lower() for s in sentences]


def _remove_custom(sentence_list, type='sec'):
    for ix in range(len(sentence_list)):
        s = sentence_list[ix]
        start_ix = s.find("(Sec.")
        if start_ix != -1:
            end_ix = start_ix + 8
            sentence_list[ix] = sentence_list[ix].replace(s[start_ix:end_ix],
                                                          " ")
    return sentence_list


def _apply_full_text_cleaning(full_sent):
    full_sent_clean = full_sent

    full_sent_clean = _remove_custom(full_sent_clean, type='sec')
    full_sent_clean = _remove_whitespace(full_sent_clean)
    full_sent_clean = _remove_punct_nums(full_sent_clean)
    full_sent_clean = _make_lowercase(full_sent_clean)

    full_sent_clean = [s for s in full_sent_clean if _is_not_empty(s)]
    return full_sent_clean


def _get_full_text_vectors(full_sent_clean, word_embeddings, embedding_size):
    full_vec = [_calc_embedding(s, word_embeddings, embedding_size)
                for s in full_sent_clean]
    return full_vec


def _apply_summary_text_cleaning(summ_sent):
    # summ_doc, summ_sent = _tokenize_sentences(summ_string, nlp)
    summ_sent_clean = summ_sent

    summ_sent_clean = _remove_custom(summ_sent_clean, type='sec')
    summ_sent_clean = _remove_whitespace(summ_sent)
    summ_sent_clean = _remove_punct_nums(summ_sent_clean)
    summ_sent_clean = _make_lowercase(summ_sent_clean)

    summ_sent_clean = [s for s in summ_sent_clean if _is_not_empty(s)]
    return summ_sent_clean


def _get_summary_text_vectors(summ_sent_clean, word_embeddings,
                              embedding_size):
    summ_vec = [_calc_embedding(s, word_embeddings, embedding_size)
                for s in summ_sent_clean]
    return summ_vec


if __name__ == "__main__":
    nlp = spacy.load('en_core_web_lg')

    # Connect to db wiht sqlalchemy
    dbname = 'congressional_bills'
    username = 'melissaferrari'
    engine = sqlalchemy.create_engine('postgres://%s@localhost/%s' %
                                      (username, dbname))
    print(engine.url)

    # Connect to make queries using psycopg2
    con = None
    con = psycopg2.connect(database=dbname, user=username)

    get_data = retrieve_data(engine)
    summary_table, bills_text_table, bill_inner_join, ed_bills = get_data

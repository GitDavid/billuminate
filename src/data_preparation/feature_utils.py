import sys
if sys.platform == "linux":
    sys.path.append('/home/ubuntu/repo/billuminate/src/')
    sys.path.append('/media/swimmers3/ferrari_06/repo/billuminate/src/')

elif sys.platform == "darwin":
    sys.path.append('/Users/melissaferrari/Projects/repo/billuminate/src/')

MODEL_ROOT = '../../models/'
NLP_MODEL_ROOT = '../../nlp_models/'
TRAINING_DATA_ROOT = '../../data/training_data/'


import numpy as np

from data_preparation import bill_utils
import pandas as pd
import pickle


def feature_generators(txt_df, feature_dict=None, joint_title=None):

    txt_df['clean_text'] = txt_df['clean_text'].fillna("")

    assert 'clean_text' in txt_df.columns
    #print(txt_df.columns)
    if not feature_dict:
        feature_dict = {
            'title_word_count': lambda x: len([wrd for wrd in x.split()
                                               if wrd in joint_title]),
            'char_count': len,
            'word_count': lambda x: len(x.split()),
            }
        feature_list = list(feature_dict.keys())

    # for feature_name in feature_dict.keys():
    #     txt_df[feature_name] = txt_df['clean_text'].apply(
    #         feature_dict[feature_name])

    txt_df['title_word_count'] = txt_df['clean_text'].apply(lambda x: len([wrd for wrd in x.split()
                                               if wrd in joint_title]))

    txt_df['char_count'] = txt_df['clean_text'].apply(len)

    txt_df['word_count'] = txt_df['clean_text'].apply( lambda x: len(x.split()))
    txt_df['word_density'] = np.divide(txt_df['char_count'], txt_df.word_count + 1)

    feature_list.append('word_density')
    # if all(x in txt_df.columns for x in ['char_count', 'word_count']):
    #     txt_df['word_density'] = txt_df.apply(
    #         lambda x: np.divide(x.char_count, x.word_count + 1))
    #     feature_list.append('word_density')

    return txt_df, feature_list

from sklearn.metrics.pairwise import cosine_similarity
from networkx import nx

def apply_pagerank(vector_embeddings):
    vlen = len(vector_embeddings)
    try:
        sim_mat = cosine_similarity(vector_embeddings)

        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph, max_iter=100)
        page_ranks = [x for k, x in scores.items()]
    except:
        page_ranks = [1/vlen]*vlen
    return page_ranks


import spacy
from spacy import displacy
from collections import Counter
import re

def apply_ents(X, nlp):
    ENT_TYPES = ['LAW', 'ORDINAL', 'GPE', 'DATE', 'MONEY', 'LAW', 'EVENT', 'PRODUCT', 'NORP']

    regex_list = [r"^\d+\.\s", r"\s[A]\s", r"\([i]+\)", r"\([I]+\)", r"[i]+\.", 
                  r"\([A-Z]\)", r"\([a-z]\)", r"\([a-z]\)", r"\([\d]\)",
                      r"\([A-Z][A-Z]\)", r"\([a-z][a-z]\)"]

    sents = X['text'].values
    for ix in range(len(sents)):
        s = sents[ix]
        if not isinstance(s, str):
            s = " "
        else:
            for r in regex_list:
                s = re.sub(r," ", s)
        sents[ix] = s

    ent_array = np.zeros((sents.shape[0], len(ENT_TYPES)), dtype=int)
    for ix, sent_list in enumerate(sents):
        for doc in nlp.pipe(sent_list):
            ent_list = [ent.label_ for ent in doc.ents]
            ent_array[ix, :] = [ent_list.count(ENT) for ENT in ENT_TYPES]

    df_ent = pd.DataFrame(ent_array, index=X.index, columns=['ent_{}'.format(ENT_TYPE) for ENT_TYPE in ENT_TYPES])
    df_ent['ent_TOTAL'] = np.sum(ent_array, axis=1)

    X = X.merge(df_ent, left_index=True, right_index=True)

    X['ent_DENSITY'] = X['ent_TOTAL'].div(X['word_count'].where(X['word_count'] != 0, np.nan))
    X['title_word_DENSITY'] = X['title_word_count'].div(X['word_count'].where(X['word_count'] != 0, np.nan))
    return X


def prepare_features(bill, train=False, word_embeddings=False, 
                     embedding_size=False, get_vecs=False, nlp_lib=False):

    official_title = bill['official_title'].lower()
    short_title = bill['short_title'].lower()
    joint_title = official_title + short_title
    print(type(bill))
    X, fvecs = bill_utils.generate_bill_data(bill, train=train,
                                             word_embeddings=word_embeddings, 
                                             embedding_size=embedding_size, 
                                             get_vecs=get_vecs)
  
    X['pagerank'] = apply_pagerank(fvecs)
    X, feature_list = feature_generators(X,
                                         joint_title=joint_title)
    print('after feat gen', X.shape)
    X = apply_ents(X, nlp=nlp_lib)
    print('after apply ent', X.shape)
    # print(joint_title)
    # feature_df_cols = ['clean_text', 'tag_rank', 'abs_loc', 'norm_loc']
    # feature_df_cols.extend(feature_list)
    # X = full_txt[feature_df_cols]
    X["doc_word_count"] = X["word_count"].sum()
    X['sent_DENSITY'] = X['word_count'].div(X['doc_word_count'].where(X['doc_word_count'] != 0, np.nan))
    X = X.fillna(0)
    print('afterdoc count', X.shape)

    return X

import sys
sys.path.append('../')

from scipy import sparse
import os
print(os.getcwd())
import numpy as np
from data_preparation import bill_utils
import pickle
import pandas as pd
import time

from sklearn.metrics.pairwise import cosine_similarity
from networkx import nx

import spacy
from spacy import displacy
from collections import Counter
import re
TRAINING_DATA_ROOT = '../../data/training_data/'
MODEL_ROOT = '../../models/'
NLP_MODEL_ROOT = '../../nlp_models/'


def load_trained_tfidf(file_path=None, subject='health'):

    if not file_path:
        file_name = 'tfidf_{}.npz'.format(subject.lower())
        file_path = os.path.join(MODEL_ROOT, file_name)

    tfidf_train = sparse.load_npz(file_path)

    return tfidf_train


def load_model(model_save_path):
    with open(model_save_path, 'rb') as training_model:
        model = pickle.load(training_model)
    return model


def title_word_count(x, joint_title):
    return len([wrd for wrd in x.split() if wrd in joint_title])


def char_count(x):
    return len(x)


def word_count(x):
    return len(x.split())


def word_density(x):
    return np.divide(char_count(x), word_count(x) + 1)


def title_word_density(x, joint_title):
    return np.divide(title_word_count(x, joint_title), word_count(x) + 1)


def doc_word_count(x):
    return sum(word_count(x))


def sent_density(x):
    return np.divide(word_count(x), doc_word_count(x) + 1)


#def apply_pagerank(x, vector_embeddings):
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


def apply_ents_torows(x, nlp):
    ENT_TYPES = ['LAW', 'ORDINAL', 'GPE', 'DATE', 'MONEY', 'LAW', 'EVENT', 'PRODUCT', 'NORP']

    for doc in nlp.pipe(x):
        ent_list = [ent.label_ for ent in doc.ents]
        ent_array = [ent_list.count(ENT) for ENT in ENT_TYPES]

    return ent_array


def apply_ents(sentences, nlp, df_index):
    ENT_TYPES = ['LAW', 'ORDINAL', 'GPE', 'DATE', 'MONEY', 'LAW', 'EVENT', 'PRODUCT', 'NORP']

    ent_array = np.zeros((sentences.shape[0], len(ENT_TYPES)), dtype=int)
    for ix, sentence in enumerate(sentences):
        for doc in nlp.pipe(sentence):
            ent_list = [ent.label_ for ent in doc.ents]
            ent_array[ix, :] = [ent_list.count(ENT) for ENT in ENT_TYPES]

    df_ent = pd.DataFrame(ent_array,
                          index=df_index,
                          columns=['ent_{}'.format(ENT_TYPE) for ENT_TYPE in ENT_TYPES])
    df_ent['ent_TOTAL'] = np.sum(ent_array, axis=1)

    return df_ent


def generate_feature_space(bill, feature_list, train=False, word_embeddings=False, 
                           embedding_size=False, get_vecs=False, nlp_lib=False, 
                           tfidf=None):

    official_title = bill['official_title'].lower()
    short_title = bill['short_title'].lower()
    joint_title = '{} {}'.format(official_title, short_title)
    
    bill_df, fvecs = bill_utils.generate_bill_data(bill, train=train,
                                                   word_embeddings=word_embeddings, 
                                                   embedding_size=embedding_size, 
                                                   get_vecs=get_vecs)
    bill_df['clean_text'] = bill_df['clean_text'].fillna("")
    
    feature_space = bill_df[['tag_rank', 'abs_loc', 'norm_loc']].copy()

    feature_dict = {
        'title_word_count': [title_word_count, 'clean_text', {'joint_title':joint_title}, False],
        'char_count': [char_count, 'clean_text', {}, False],
        'word_count': [word_count, 'clean_text', {}, False] ,
        'word_density': [word_density, 'clean_text', {}, False] ,
        'title_word_DENSITY': [title_word_density, 'clean_text', {'joint_title':joint_title}, False]
        }

    for feature in feature_list:

        if feature == 'ents':
            sentences = bill_df['clean_text'].values
            df_ent = apply_ents(sentences, nlp_lib, bill_df.index.values)
            feature_space = feature_space.merge(df_ent, left_index=True, right_index=True)
            feature_space['ent_DENSITY'] = feature_space['ent_TOTAL'].div(
                feature_space['word_count'].where(feature_space['word_count'] != 0, np.nan))
        elif feature == 'tfidf':
            tfidf_mat = tfidf.transform(bill_df['clean_text'])
            feature_space = feature_space.fillna(0)
            return bill_df, [feature_space, tfidf_mat,]
        elif feature == 'doc_word_count':
            # 'doc_word_count': [doc_word_count, 'clean_text', {}, False]
            feature_space["doc_word_count"] = feature_space["word_count"].sum()
        elif feature == 'sent_DENSITY':
            # 'sent_DENSITY': [sent_density, 'clean_text', {}, False]
            feature_space['sent_DENSITY'] = feature_space['word_count'].div(
                feature_space['doc_word_count'].where(feature_space['doc_word_count'] != 0, np.nan))
        elif feature == 'page_rank':
            #'page_rank': [apply_pagerank, '', {'vector_embeddings':fvecs}, False],
            feature_space[feature] = apply_pagerank(fvecs)
        else:
            func, col, args, _ = feature_dict[feature]
            #if not is_list:
            feature_space[feature] = bill_df[col].apply(func, **args)
            #'ents': [apply_ents, 'clean_text', [nlp], True]

    feature_space = feature_space.fillna(0)

    return bill_df, [feature_space,]


def join_features(feature_list):

    if any(sparse.issparse(x) for x in feature_list):
        all_features = sparse.hstack(feature_list)

    else:
        all_features = np.hstack(feature_list)

    return all_features


def get_bill_dict(bills_info, bill_id):
    bill = bills_info[bills_info['bill_id'] == bill_id].copy()
    bill = bill_utils._return_correct_version(bill, as_dict=True)
    return bill

##quick requests/beautifulsoup
def apply_model(bills_info, bill_id, model=None, feature_list=None, tfidf=None,
                train=False, word_embeddings=False, embedding_size=False, get_vecs=False, nlp_lib=False):

    bill = get_bill_dict(bills_info, bill_id)
    bill_df, feature_space = generate_feature_space(bill, feature_list, 
                                                    train=train, word_embeddings=word_embeddings, 
                                                    embedding_size=embedding_size, get_vecs=get_vecs,nlp_lib=nlp_lib, tfidf=tfidf)
    X_features = join_features(feature_space)
    bill_df = bill_df.merge(feature_space[0],
                            on=['tag_rank', 'abs_loc', 'norm_loc'])

    y_pred = model.predict(X_features)
    y_probs = model.predict_proba(X_features)

    bill_df['prediction'] = y_pred
    bill_df['predict_proba0'] = y_probs[:,0]
    bill_df['predict_proba1'] = y_probs[:,1]

    return bill_df, bill

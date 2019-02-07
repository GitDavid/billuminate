# import matplotlib
# import matplotlib.pylab as plt

import os
import random
import re
import string
import xml.etree.ElementTree as ET
import tqdm
import numpy as np
import time

import pandas as pd
# import sqlalchemy_utils
# import psycopg2
import spacy
import sqlalchemy
from sklearn import metrics

import text_processing


DATA_ROOT = '../../data/training_data/'


def retrieve_data(engine, subject=None):
    print(subject)
    summary_table = pd.read_sql_table('summaries', con=engine, columns=None)
    bills_info = pd.read_sql_table('bills', con=engine)
    bills_txt_table = pd.read_sql_table('bill_text', con=engine, columns=None)

    query = """
            SELECT
            sm.text AS summary_text,
            sm.bill_ix, sm.as, sm.date,
            bv.code,
            bills.subjects_top_term,
            bt.text AS full_text
            FROM summaries sm
            INNER JOIN bill_text bt
            ON sm.bill_ix=bt.bill_ix
            INNER JOIN bill_versions bv
            ON bv.id=bt.bill_version_id
            ;
            """
    if subject:
        subject_query = """
                        INNER JOIN bills
                        ON bills.id=sm.bill_ix
                        WHERE bills.subjects_top_term='%s';
                        """
        query = query[:query.find(';')] + subject_query % subject
        # print(query)

    bill_inner_join = pd.read_sql_query(query, engine)

    if subject:

        select_cols = ['id', 'official_title', 'bill_type', 'status_at',
                       'sponsor', 'status', 'subjects_top_term', 'bill_id',
                       'introduced_at', 'congress', 'short_title']

        filter_bills = bills_info[bills_info['subjects_top_term'] ==
                                  subject][select_cols]

        filter_bills = filter_bills.rename(columns={'id': 'bill_ix'})
        filter_bills_join = filter_bills.merge(bill_inner_join,
                                               on='bill_ix', how='inner')

    return summary_table, bills_txt_table, bill_inner_join, filter_bills_join


def _return_correct_bill_version(df_bills):
    num_rows = len(df_bills)
    if num_rows == 0:
        raise Exception('Oh no! This bill is not in the database.')
    elif num_rows > 1:
        rank_codes = ['ENR', 'EAS', 'EAH', 'RS', 'ES',
                      'PCS', 'EH', 'RH', 'IS', 'IH']
        code = next(i for i in rank_codes if i in df_bills['code'].unique())
        df_bills = df_bills[df_bills['code'] == code]
    return df_bills.iloc[0].to_dict()


def _check_if_unfavorable_bill(bill_dict):
    if 'to amend the' in bill_dict['official_title'].lower():
        print('this is an ammendment, proceed wiht caution')
        return True


def _preprocess_bill_data(ed_bills_join, bill_ix):

    test_bill = ed_bills_join[(ed_bills_join['bill_ix'])].copy()
    test_bill = _return_correct_bill_version(test_bill)
    is_bad = _check_if_unfavorable_bill(test_bill)
    print(is_bad)

    short_title = test_bill['short_title']
    official_title = test_bill['official_title']
    summ_string = test_bill['summary_text']
    print(short_title)
    print(official_title)
    return test_bill, short_title, official_title, summ_string


def _create_sim_mat(sent_vecs_1, sent_vecs_2,
                    embedding_size):
    sim_mat = np.zeros([len(sent_vecs_1), len(sent_vecs_2)])
    vlen = embedding_size
    for i in range(len(sent_vecs_1)):
        for j in range(len(sent_vecs_2)):
            sim_mat[i][j] = metrics.pairwise.cosine_similarity(sent_vecs_1[i].
                                                               reshape(1,
                                                                       vlen),
                                                               sent_vecs_2[j].
                                                               reshape(1,
                                                                       vlen))[0,
                                                                              0]
    return sim_mat


def _get_closest_sent_match_ix(sim_mat):

    # ix_match = np.argmax(sim_mat, axis=0)
    ix_sort = (-sim_mat).argsort(axis=0)[:1][0]
    ix_match = np.sort(ix_sort)
    ix_match = np.unique(ix_match)
    # closest_match = [full_sent[i] for i in ix_match]
    return ix_match  # closest_match


def label_important_sentences(full_vec, summ_vec, version, embedding_size):

    assert version in [1], 'version does not exist'

    if version == 1:

        # embedding_size = np.shape(full_vec)[1]
        # print('embedding_size is {}'.format(embedding_size))
        # takes in sentence level embeddings for full text and summaries
        sim_mat = _create_sim_mat(full_vec, summ_vec,
                                  embedding_size=embedding_size)

        ix_match = _get_closest_sent_match_ix(sim_mat)
        
        data = pd.DataFrame(full_vec, columns=['embed_{:03}'.format(i)
                            for i in range(embedding_size)])
        data['in_summary'] = 0
        data.loc[ix_match, 'in_summary'] = 1

        return data


def get_feature_vector(text_string, word_embeddings, embedding_size, nlp,
                       text_type='full_text'):

    assert text_type in ['summary', 'full_text']

    text_doc, text_sent = text_processing._tokenize_sentences(text_string, nlp)

    if text_type == 'summary':
        # tokenize and clean
        summ_sent_clean = text_processing._apply_summary_text_cleaning(text_sent)
        summ_vec = text_processing._get_summary_text_vectors(summ_sent_clean,
                                                             word_embeddings,
                                                             embedding_size)
        return summ_vec

    if text_type == 'full_text':
        # tokenize and clean
        full_sent_clean = text_processing._apply_full_text_cleaning(text_sent)
        full_vec = text_processing._get_full_text_vectors(full_sent_clean,
                                                          word_embeddings,
                                                          embedding_size)
        return full_vec


def aggregate_training_data(bills_df, version, path_to_embedding, nlp):

    print('loading word embeddings')
    (word_embeddings,
     embedding_size) = text_processing._extract_embeddings(path_to_embedding)
    print('finished loading')
    all_data = pd.DataFrame()

    unique_bills = bills_df.bill_ix.unique()
    num_rows = 0
    for bill_ix in tqdm.tqdm(unique_bills):
        try:
            bill = bills_df[(bills_df['bill_ix'] == bill_ix)].copy()
            bill = _return_correct_bill_version(bill)
            xml_text_string = bill['full_text']
            (full_string,
             enum_string,
             sentences) = text_processing._bill_from_xml(xml_text_string)

            full_vec = get_feature_vector(full_string, word_embeddings,
                                          embedding_size, nlp,
                                          text_type='full_text')

            summ_vec = get_feature_vector(full_string, word_embeddings,
                                          embedding_size, nlp,
                                          text_type='summary')

            data = label_important_sentences(full_vec=full_vec,
                                             summ_vec=summ_vec,
                                             version=version,
                                             embedding_size=embedding_size)

            assert(len(data[data['in_summary'] != 1]) +
                   len(data[data['in_summary'] == 1]) ==
                   len(data))

            all_data = all_data.append(data)
            num_rows += len(data) 
        except:
            print('{} failed'.format(bill_ix))
    print('The number of rows in the dataset should be {}'.format(num_rows))
    return all_data


def main():

    # Connect to db with sqlalchemy
    dbname = 'congressional_bills'
    username = 'melissaferrari'
    engine = sqlalchemy.create_engine('postgres://%s@localhost/%s' %
                                      (username, dbname))
    print(engine.url)

    subject = 'Education'
    version = 1

    print('querying data')
    start_time = time.time()
    get_data = retrieve_data(engine, subject=subject)
    summary_table, bills_text_table, bill_inner_join, filter_bills = get_data
    print("--- That took {} seconds ---".format(time.time() - start_time))

    embedding_size = 100
    path_to_embedding = '../../nlp_models/glove.6B/glove.6B.{}d.txt'.format(embedding_size)

    print('loading spacy en_core_web_sm')
    start_time = time.time()
    nlp = spacy.load('en_core_web_sm')
    print("--- That took a {} seconds ---".format(time.time() - start_time))

    filter_bills = filter_bills[:50]
    all_data = aggregate_training_data(filter_bills, version=version,
                                       path_to_embedding=path_to_embedding,
                                       nlp=nlp)

    file_name = 'trainingdata_v{}_emb{}_{}.csv'.format(version, embedding_size, subject)
    save_path = os.path.join(DATA_ROOT,
                             file_name)
    all_data.to_csv(save_path)



if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- That took a total of {} seconds ---".format(time.time() - start_time))

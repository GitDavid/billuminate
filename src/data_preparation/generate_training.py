import datetime
import os
import time

import numpy as np

import pandas as pd
import spacy
import sqlalchemy
import tqdm

import sys
sys.path.append('../')
from data_preparation import bill_utils, text_utils

TRAINING_DATA_ROOT = '../../data/training_data/'
NLP_MODEL_ROOT = '../../nlp_models/'


def aggregate_training_data(bills_df,
                            word_embeddings, nlp):

    all_text_data = pd.DataFrame()
    all_labeled = pd.DataFrame()
    all_sum_text = pd.DataFrame()

    unique_bills = bills_df.bill_id.unique()

    num_rows = 0
    for bill_id in tqdm.tqdm(unique_bills):
        try:
            bill = bills_df[(bills_df['bill_id'] == bill_id)].copy()
            bill = bill_utils._return_correct_version(bill, as_dict=True)
            get_data = bill_utils.generate_bill_data(bill,
                                                     word_embeddings,
                                                     nlp,
                                                     train=True)

            label_df, full_txt, sum_df = get_data
            all_labeled = all_labeled.append(label_df).copy()
            all_text_data = all_text_data.append(full_txt).copy()
            all_sum_text = all_sum_text.append(sum_df).copy()
            num_rows += len(label_df)

        except ValueError:
            print('{} failed'.format(bill_id))
            pass

    return all_labeled, all_text_data, all_sum_text


def main():

    for subject in ['Armed forces and national security',
                    'Government operations and politics',
                    'Taxation']:
        # ['Health', 'Education', 'Crime and law enforcement']

        date = datetime.datetime.today().strftime('%Y%m%d')

        custom = 'leglemno'
        all_text_filename = '{}_structuredtext_{}_{}.csv'.format(date,
                                                                 custom,
                                                                 subject)

        all_text_savepath = os.path.join(TRAINING_DATA_ROOT, all_text_filename)
        print(os.path.isfile(all_text_savepath))

        # Connect to database
        dbname = 'congressional_bills'
        username = 'postgres'
        password = 'password'
        engine = sqlalchemy.create_engine('postgres://%s:%s@localhost/%s' %
                                          (username, password, dbname))

        print(engine.url)
        print('Querying data ...')
        start_time = time.time()

        bill_df = bill_utils.retrieve_data(engine, subject=subject)
        bill_df = bill_df.reset_index(drop=True)

        print("--- That took {} seconds ---".format(time.time() - start_time))

        nlp_models = ['en_vectors_web_lg', 'en_core_web_lg', 'en']
        nlp_model = nlp_models[1]

        print('Loading NLP model {}'.format(nlp_model))
        start_time = time.time()
        nlp = spacy.load('en')
        print("--- That took {} seconds ---".format(time.time() - start_time))

        NLP_ext = 'word2vec-legal/lemmatized-legal/no replacement/'
        emb_name = 'legal_lemmatized_no_replacement.bin'
        path_to_embedding = NLP_MODEL_ROOT + NLP_ext + emb_name

        print('Loading word embeddings from {} ...'.format(path_to_embedding))
        start_time = time.time()
        word_embeddings = text_utils._load_embeddings_other(path_to_embedding)
        print("--- That took {} seconds ---".format(time.time() - start_time))

        # We want to run the analysis in chunks for large processes.
        chunk_size = 100
        incrament = np.arange(0, len(bill_df)+20, chunk_size)
        min_ix = incrament[:-1]
        max_ix = incrament[1:]
        num_chunks = len(incrament[:-1])
        print('Working in chunks of {} rows'.format(chunk_size))
        print('There will be {} chunks total'.format(num_chunks))

        for ix in range(num_chunks):
            bill_df_chunk = bill_df.loc[min_ix[ix]:max_ix[ix]].copy()

            get_data = aggregate_training_data(
                bill_df_chunk, word_embeddings=word_embeddings, nlp=nlp)

            all_labeled, all_text_data, all_sum_text = get_data

            # Save labeled data
            alabeled_filename = '{}_training_labeled_{}_{}.csv'.format(date,
                                                                       custom,
                                                                       subject)
            alabeled_path = os.path.join(TRAINING_DATA_ROOT, alabeled_filename)
            bill_utils.to_csv_append_mode(all_labeled, alabeled_path)

            # Save text
            atext_filename = '{}_structuredtext_{}_{}.csv'.format(date,
                                                                  custom,
                                                                  subject)
            atext_savepath = os.path.join(TRAINING_DATA_ROOT, atext_filename)
            bill_utils.to_csv_append_mode(all_text_data, atext_savepath)

            # Save summaries
            asum_filename = '{}_structuredsummaries_{}_{}.csv'.format(date,
                                                                      custom,
                                                                      subject)
            asum_savepath = os.path.join(TRAINING_DATA_ROOT, asum_filename)
            bill_utils.to_csv_append_mode(all_sum_text, asum_savepath)


if __name__ == "__main__":
    start_time = time.time()
    main()
    seconds = time.time() - start_time
    minutes = int(np.divide(seconds, 60))
    print("--- Generating training set took {} minutes ---".format(minutes))

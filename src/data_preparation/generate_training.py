import sys
if sys.platform == "linux":
    sys.path.append('/home/ubuntu/repo/billuminate/src/')
    sys.path.append('/media/swimmers3/ferrari_06/repo/billuminate/src/')

    TRAINING_DATA_ROOT = '/media/swimmers3/ferrari_06/repo/billuminate/data/training_data/'
    
elif sys.platform == "darwin":
    sys.path.append('/Users/melissaferrari/Projects/repo/billuminate/src/')


import datetime
import os
import time

import numpy as np

import pandas as pd
import spacy
import sqlalchemy
from data_preparation import bill_utils, text_utils
import tqdm

#TRAINING_DATA_ROOT = '../../data/training_data/'


def aggregate_training_data(bills_df,
                            word_embeddings, embedding_size, nlp):

    all_text_data = pd.DataFrame()
    all_labeled = pd.DataFrame()
    #all_embed_data = pd.DataFrame()
    all_sum_text = pd.DataFrame()

    unique_bills = bills_df.bill_id.unique()

    # if len(unique_bills) == len(bills_df):
    #     multiple_versions_exist = False
    # else:
    #     multiple_versions_exist = True

    num_rows = 0
    for bill_id in tqdm.tqdm(unique_bills):
        try:
            #print(bill_id)
            #print(bill_id)
            bill = bills_df[(bills_df['bill_id'] == bill_id)].copy()
            bill = bill_utils._return_correct_version(bill, as_dict=True)
            get_data = bill_utils.generate_bill_data(bill, word_embeddings,
                                                     embedding_size, nlp,
                                                     train=True)
            #label_df, embed_data, full_txt, sum_df = get_data
            label_df, full_txt, sum_df = get_data
            all_labeled = all_labeled.append(label_df).copy()
            #all_embed_data = all_embed_data.append(embed_data)
            all_text_data = all_text_data.append(full_txt).copy()
            all_sum_text = all_sum_text.append(sum_df).copy()
            num_rows += len(label_df)
            #print('The number of rows in the dataset should be {}'.format(num_rows))
            #return all_labeled, all_embed_data, all_text_data, all_sum_text

        except:
            print('{} failed'.format(bill_id))
            pass

    return all_labeled, all_text_data, all_sum_text



def main():
    
    for subject in ['Armed forces and national security', 
                   'Government operations and politics',
                   'Taxation']:
    #for subject in ['Health', 'Education', 'Crime and law enforcement']:
        TRAINING_DATA_ROOT = '/media/swimmers3/ferrari_06/repo/billuminate/data/training_data/'
        date = datetime.datetime.today().strftime('%Y%m%d')

        #subject = 'Crime and law enforcement'
        #subject = 'Education'
        custom = 'leglemno'
        #custom = 'glove200'
        all_text_filename = '{}_structuredtext_{}_{}.csv'.format(date,custom,
                                                                  subject)
        all_text_savepath = os.path.join(TRAINING_DATA_ROOT, all_text_filename)
        print(TRAINING_DATA_ROOT)
        print(all_text_savepath)
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
        #bill_df = bill_df[bill_df.subject_top_term.isin([ 'Crime and law enforcement', 'Education', 'Health', 'Taxation'])]
        bill_df = bill_df.reset_index(drop=True)

        print("--- That took {} seconds ---".format(time.time() - start_time))

        nlp_models = ['en_vectors_web_lg', 'en_core_web_lg']
        nlp_model = nlp_models[1]

        print('Loading NLP model {}'.format(nlp_model))
        start_time = time.time()
        #nlp = spacy.load('en_core_web_sm')
        nlp = spacy.load('en_core_web_lg')
        print("--- That took {} seconds ---".format(time.time() - start_time))

        embedding_size = 200
        #path_to_embedding = 'glove.6B/glove.6B.{}d.txt'.format(embedding_size)
        #path_to_embedding = '/media/swimmers3/ferrari_06/repo/billuminate/nlp_models/glove.6B/glove.6B.{}d.txt'.format(embedding_size)
        #print('Loading word embeddings from {} ...'.format(path_to_embedding))
        path_to_embedding = '/media/swimmers3/ferrari_06/repo/billuminate/nlp_models/word2vec-legal/lemmatized-legal/no replacement/legal_lemmatized_no_replacement.bin'
        print('Loading word embeddings from {} ...'.format(path_to_embedding))
        start_time = time.time()
        #word_embeddings, _ = text_utils._load_embeddings(path_to_embedding)
        word_embeddings, _ = text_utils._load_embeddings_other(path_to_embedding)
        print(len(word_embeddings.keys()))
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
            #print(TRAINING_DATA_ROOT)
            #bill_df_chunk = bill_df[min_ix[ix]:max_ix[ix]].copy()
            bill_df_chunk = bill_df.loc[min_ix[ix]:max_ix[ix]].copy()#bill_df[min_ix[ix]:max_ix[ix]].copy()

            get_data = aggregate_training_data(bill_df_chunk,word_embeddings=word_embeddings, embedding_size=embedding_size, nlp=nlp)
            #get_data = aggregate_training_data(bill_df_chunk, nlp=nlp)
            #all_labeled, all_embed_data, all_text_data, all_sum_text = get_data
            all_labeled, all_text_data, all_sum_text = get_data

            # Save labeled data
            labeled_filename = '{}_training_labeled_{}_{}.csv'.format(date,custom,
                                                                  subject)
            labeled_savepath = os.path.join(TRAINING_DATA_ROOT, labeled_filename)
            bill_utils.to_csv_append_mode(all_labeled, labeled_savepath)

            # Save embeddings for text and summaries
    #         embeddings_filename = '{}_allembeddings_leglemno_{}.csv'.format(date,
    #                                                                      subject)
    #         embeddings_savepath = os.path.join(TRAINING_DATA_ROOT,
    #                                            embeddings_filename)
    #         bill_utils.to_csv_append_mode(all_embed_data, embeddings_savepath)

            # Save text
            all_text_filename = '{}_structuredtext_{}_{}.csv'.format(date,custom,
                                                                  subject)
            all_text_savepath = os.path.join(TRAINING_DATA_ROOT, all_text_filename)
            bill_utils.to_csv_append_mode(all_text_data, all_text_savepath)

            # Save summaries
            all_sum_filename = '{}_structuredsummaries_{}_{}.csv'.format(date,custom,
                                                                  subject)
            all_sum_savepath = os.path.join(TRAINING_DATA_ROOT, all_sum_filename)
            bill_utils.to_csv_append_mode(all_sum_text, all_sum_savepath)


if __name__ == "__main__":
    start_time = time.time()
    main()
    seconds = time.time() - start_time
    minutes = int(np.divide(seconds, 60))
    print("--- Generating training set took {} minutes ---".format(minutes))

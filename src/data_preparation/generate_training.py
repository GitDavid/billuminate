import datetime
import os
import time

import numpy as np

import bill_utils
import pandas as pd
import spacy
import sqlalchemy
import text_utils
import tqdm
import training_utils

TRAINING_DATA_ROOT = '../../data/training_data/'


def aggregate_training_data(bills_df, version,
                            word_embeddings, embedding_size, nlp):

    all_text_data = pd.DataFrame()
    all_labeled = pd.DataFrame()
    all_embed_data = pd.DataFrame()
    all_sum_text = pd.DataFrame()

    unique_bills = bills_df.bill_id.unique()

    # if len(unique_bills) == len(bills_df):
    #     multiple_versions_exist = False
    # else:
    #     multiple_versions_exist = True

    num_rows = 0
    for bill_id in tqdm.tqdm(unique_bills):
        try:
            bill = bills_df[(bills_df['bill_id'] == bill_id)].copy()
            bill = bill_utils._return_correct_bill_version(bill, as_dict=True)

            short_title = bill['short_title']
            full_string = bill['full_text']
            summ_string = bill['summary_text']

            full_string = bill['full_text']
            full_txt, fsents = bill_utils.get_clean_text(full_string,
                                                         text_type='full_text')

            full_txt['bill_id'] = bill_id
            full_txt['clean_text'] = fsents

            locs = full_txt['loc_ix']
            full_txt['abs_loc'] = (locs - locs.min()).values
            full_txt['norm_loc'] = (np.divide(locs - locs.min(),
                                              locs.max() - locs.min())).values

            fvecs = text_utils._calc_embeddings_set(fsents,
                                                    word_embeddings,
                                                    embedding_size)

            sum_df, ssents = bill_utils.get_clean_text(summ_string,
                                                       text_type='summary',
                                                       short_title=short_title,
                                                       nlp=nlp)

            sum_df['bill_id'] = bill_id
            sum_df['clean_text'] = ssents

            svecs = text_utils._calc_embeddings_set(ssents,
                                                    word_embeddings,
                                                    embedding_size)

            label_df, ix_match = training_utils.label_important(fvecs, svecs,
                                                                embedding_size,
                                                                max_sim=0.5)

            summ_data = pd.DataFrame(svecs, columns=['embed_{:03}'.format(i)
                                     for i in range(embedding_size)])

            summ_data = summ_data.reset_index()
            summ_data = summ_data.rename(columns={'index': 'loc_ix'})

            summ_data['in_summary'] = 2
            summ_data['bill_id'] = bill_id

            label_df = label_df.reset_index()
            label_df = label_df.rename(columns={'index': 'loc_ix'})
            label_df['bill_id'] = bill_id

            embed_data = pd.concat([label_df, summ_data], sort=False)

            assert(len(label_df[label_df['in_summary'] != 1]) +
                   len(label_df[label_df['in_summary'] == 1]) ==
                   len(label_df))

            all_labeled = all_labeled.append(label_df)
            all_embed_data = all_embed_data.append(embed_data)
            all_text_data = all_text_data.append(full_txt)
            all_sum_text = all_sum_text.append(sum_df)

            num_rows += len(label_df)

        except TypeError:
            print('{} failed'.format(bill_id))

    print('The number of rows in the dataset should be {}'.format(num_rows))
    return all_labeled, all_embed_data, all_text_data, all_sum_text


def main():

    date = datetime.datetime.today().strftime('%Y%m%d')

    # Connect to database
    dbname = 'congressional_bills'
    username = 'melissaferrari'
    engine = sqlalchemy.create_engine('postgres://%s@localhost/%s' %
                                      (username, dbname))
    print(engine.url)

    subject = 'Health'

    print('Querying data ...')
    start_time = time.time()
    bill_df = bill_utils.retrieve_data(engine, subject=subject)
    print("--- That took {} seconds ---".format(time.time() - start_time))

    nlp_models = ['en_vectors_web_lg', 'en_core_web_lg']
    nlp_model = nlp_models[1]

    print('Loading NLP model {}'.format(nlp_model))
    start_time = time.time()
    nlp = spacy.load('en_core_web_lg')
    print("--- That took {} seconds ---".format(time.time() - start_time))

    embedding_size = 100
    path_to_embedding = 'glove.6B/glove.6B.{}d.txt'.format(embedding_size)

    print('Loading word embeddings from {} ...'.format(path_to_embedding))
    start_time = time.time()
    word_embeddings, _ = text_utils._load_embeddings(path_to_embedding)
    print("--- That took {} seconds ---".format(time.time() - start_time))

    # We want to run the analysis in chunks for large processes.
    chunk_size = 200
    incrament = np.arange(0, len(bill_df)+20, chunk_size)
    min_ix = incrament[:-1]
    max_ix = incrament[1:]
    num_chunks = len(incrament[:-1])
    print('Working in chunks of {} rows'.format(chunk_size))
    print('There will be {} chunks total'.format(num_chunks))
    for ix in range(num_chunks):

        bill_df_chunk = bill_df[min_ix[ix]:max_ix[ix]].copy()

        get_data = aggregate_training_data(bill_df_chunk, nlp=nlp)
        all_labeled, all_embed_data, all_text_data, all_sum_text = get_data

        # Save labeled data
        labeled_filename = '{}_training_labeled_{}.csv'.format(date, subject)
        labeled_savepath = os.path.join(TRAINING_DATA_ROOT, labeled_filename)
        bill_utils.to_csv_append_mode(all_labeled, labeled_savepath)

        # Save embeddings for text and summaries
        embeddings_filename = '{}_allembeddings_Glove_{}.csv'.format(date,
                                                                     subject)
        embeddings_savepath = os.path.join(TRAINING_DATA_ROOT,
                                           embeddings_filename)
        bill_utils.to_csv_append_mode(all_embed_data, embeddings_savepath)

        # Save text
        all_text_filename = '{}_structuredtext_{}.csv'.format(date,
                                                              subject)
        all_text_savepath = os.path.join(TRAINING_DATA_ROOT, all_text_filename)
        bill_utils.to_csv_append_mode(all_text_data, all_text_savepath)

        # Save summaries
        all_sum_text = '{}_structuredsummaries_{}.csv'.format(date,
                                                              subject)
        all_sum_savepath = os.path.join(TRAINING_DATA_ROOT, all_sum_text)
        bill_utils.to_csv_append_mode(all_sum_text, all_sum_savepath)


if __name__ == "__main__":
    start_time = time.time()
    main()
    seconds = time.time() - start_time
    minutes = int(np.divide(seconds, 60))
    print("--- Generating training set took {} minutes ---".format(minutes))

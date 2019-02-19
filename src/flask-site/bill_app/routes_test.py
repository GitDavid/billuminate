import sys
if sys.platform == "linux":
    sys.path.append('/home/ubuntu/repo/billuminate/src/')

    MODEL_ROOT = '/home/ubuntu/repo/billuminate/models/'
    NLP_MODEL_ROOT = '/home/ubuntu/repo/billuminate/nlp_models/'

elif sys.platform == "darwin":
    sys.path.append('/Users/melissaferrari/Projects/repo/billuminate/src/')

import pandas as pd
import psycopg2
import json
import numpy as np
from modeling import model_utils
from data_preparation import bill_utils, text_utils
import spacy
import pickle
import os
import en_core_web_lg
nlp = en_core_web_lg.load()

MODEL_ROOT = '../../../models/'
NLP_MODEL_ROOT = '../../../nlp_models/'
#NLP_MODEL_ROOT = '../../nlp_models/'


embedding_size = 200
path_to_embedding = NLP_MODEL_ROOT + 'lemmatized-legal/no replacement/legal_lemmatized_no_replacement.bin'
word_embeddings, _ = text_utils._load_embeddings_other(path_to_embedding)

user = 'melissaferrari'  # add your Postgres username here
host = 'localhost'
dbname = 'congressional_bills'
con = psycopg2.connect(database=dbname, user=user)

model_save_path = os.path.join(
    MODEL_ROOT, 'undersampled_RandomForestClassifier10_tfidf10000_other22_linux.pickle')
with open(model_save_path, 'rb') as trained_model:
    current_model = pickle.load(trained_model)

tfidf_save_path = os.path.join(MODEL_ROOT, 'tfidf_linux.pickle')
with open(tfidf_save_path, 'rb') as trained_model:
    tfidf_model = pickle.load(trained_model)

print('done loading stuff')

def bills_output():
    bill_title = None
    read_time = 2
    bill_id = 'hr4764-114'
    print('BILL ID = {}'.format(bill_id))

    if any(x for x in [bill_id, bill_title]):

        bill_df = bill_utils.retrieve_data(
            con, bill_id=bill_id, bill_title=bill_title, subject=None)

        if bill_df.empty:
            print('no bil')
            return

        bill_id = bill_df.bill_id.unique()[0]
        X, info_dict = model_utils.apply_model(bill_df, bill_id, model=current_model, 
                tfidf_train=tfidf_model, train=False, 
                word_embeddings=word_embeddings, 
                embedding_size=embedding_size, get_vecs=True, nlp_lib=nlp)

        wpm = 200  # words per minute
        X['read_time'] = np.divide(X['word_count'], wpm).round(decimals=2)
        X['predict_ranking'] = X['predict_proba1'].rank(
            ascending=False).astype(int)

        sum_ser = X.sort_values(by='predict_proba1',
                                ascending=False)['read_time'].cumsum()
        sum_ser.name = 'time_cumulative'

        X = pd.merge(X, pd.DataFrame(sum_ser),
                     left_index=True, right_index=True)

        min_slide_val = 1
        max_slide_val = np.ceil(X['read_time'].sum()).astype(int)
        if max_slide_val == min_slide_val:
            max_slide_val += 1
        if not read_time:
            read_time = X[X.prediction == 1]['read_time'].sum()

        pred_results = X[(X.time_cumulative <= (float(read_time) + .01))
                         | (X.tag == 'section')].copy()
        read_time = int(np.ceil(float(read_time)))

        print(pred_results.columns)
        print(pred_results.head())
        return pred_results
    else:
        empty_df = pd.DataFrame(columns=['tag', 'tag_rank', 'text'])
        print('empty_df')
        return


if __name__ == "__main__":
    bills_output()

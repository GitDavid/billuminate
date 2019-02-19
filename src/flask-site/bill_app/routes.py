import sys
if sys.platform == "linux":
     sys.path.append('/home/ubuntu/repo/billuminate/src/')

     MODEL_ROOT = '/home/ubuntu/repo/billuminate/models/'
     NLP_MODEL_ROOT = '/home/ubuntu/repo/billuminate/nlp_models/'

elif sys.platform == "darwin":
    sys.path.append('/Users/melissaferrari/Projects/repo/billuminate/src/')

#sys.path.append('/../../src/')

import os
import pickle
from flask import render_template, request, jsonify
from bill_app import con
import spacy
from data_preparation import bill_utils, text_utils
from modeling import model_utils
import numpy as np
from wtforms import TextField, Form
import json
import psycopg2
import pandas as pd
from bill_app import con
from bill_app import app
from sqlalchemy import create_engine
import en_core_web_lg
nlp = en_core_web_lg.load()


# elif sys.platform == "darwin":

MODEL_ROOT = '../../models/'
NLP_MODEL_ROOT = '../../nlp_models/'


#print('loading models')
model_save_path = os.path.join(MODEL_ROOT, 'undersampled_RandomForestClassifier10_tfidf10000_other22_linux.pickle')
with open(model_save_path, 'rb') as trained_model:
    current_model = pickle.load(trained_model)

# model_name = 'over_RandomForestClassifier_on_health_nestimators100_random_state0.pkl'
#current_model = model_utils.load_model(MODEL_ROOT + model_name)
tfidf_save_path = os.path.join(MODEL_ROOT, 'tfidf_linux.pickle')
with open(tfidf_save_path, 'rb') as trained_model:
    tfidf_model = pickle.load(trained_model)

embedding_size = 200
path_to_embedding = NLP_MODEL_ROOT + 'lemmatized-legal/no replacement/legal_lemmatized_no_replacement.bin'
word_embeddings, _ = text_utils._load_embeddings_other(path_to_embedding)

@app.route('/', methods=['GET', 'POST'])
@app.route('/bills_output', methods=['GET'])
def bills_output():
    bill_id = None
    bill_title = None

    bill_id = request.args.get('bill_id')
    print('BILL ID = {}'.format(bill_id))

    bill_title = request.args.get('bill_title')
    print('BILL Title = {}'.format(bill_title))

    read_time = request.args.get('reading_time')
    print('Read time = {}'.format(read_time))

    if any(x for x in [bill_id, bill_title]):
        bill_df = bill_utils.retrieve_data(
            con, bill_id=bill_id, bill_title=bill_title, subject=None)

        if bill_df.empty:

            return render_template("bill_not_found.html",
                                   bill_id=bill_id,
                                   bill_title=bill_title)

        bill_id = bill_df.bill_id.unique()[0]
        X, info_dict = model_utils.apply_model(
            bill_df, bill_id, model=current_model, 
                tfidf_train=tfidf_model, train=False, 
                word_embeddings=word_embeddings, 
                embedding_size=embedding_size, get_vecs=True, nlp_lib=nlp)

        wpm = 200 #words per minute
        X['read_time'] = np.divide(X['word_count'], wpm).round(decimals=2)
        X['predict_ranking'] = X['predict_proba1'].rank(ascending=False).astype(int)

        sum_ser = X.sort_values(by='predict_proba1',
                                ascending=False)['read_time'].cumsum()
        sum_ser.name = 'time_cumulative'

        X = pd.merge(X, pd.DataFrame(sum_ser), left_index=True, right_index=True)
        
        min_slide_val = 1
        max_slide_val = np.ceil(X['read_time'].sum()).astype(int)
        if max_slide_val == min_slide_val:
            max_slide_val += 1
        if not read_time:
            read_time = X[X.prediction == 1]['read_time'].sum()

        pred_results = X[(X.time_cumulative <= (float(read_time) + .01))
                         | (X.tag == 'section')].copy()
        read_time = int(np.ceil(float(read_time)))

        return render_template("output.html",
                               summarization_result=pred_results[['tag', 'tag_rank', 'text']],
                               bill_info=info_dict,
                               min_slide_val=min_slide_val,
                               max_slide_val=max_slide_val,
                               init_slide_val=read_time)
    else:
        empty_df = pd.DataFrame(columns=['tag', 'tag_rank', 'text'])
        return render_template("output.html",
                               summarization_result=empty_df,
                               bill_info=None,
                               min_slide_val=0,
                               max_slide_val=10,
                               init_slide_val=5)


@app.route('/api/bills/id/<bill_id>', methods=['GET'])
def get_bills_by_id(bill_id):

    query = "SELECT bill_id FROM bills WHERE bill_id LIKE '%" + bill_id + "%' LIMIT 10;"
    query_results = pd.read_sql_query(query, con)
    output_list = list(query_results['bill_id'].values)
    print(output_list)
    return jsonify(output_list)


@app.route('/api/bills/title/<title>', methods=['GET'])
def get_bills_by_title(title):

    query = "SELECT official_title FROM bills WHERE official_title LIKE '%" + \
        title + "%' LIMIT 10;"
    query_results = pd.read_sql_query(query, con)
    output_list = list(query_results['official_title'].values)
    print(output_list)
    return jsonify(output_list)


@app.route('/api/bills/subject/<subject>', methods=['GET'])
def get_bills_by_subject(subject):

    query = "SELECT subjects_top_term FROM bills WHERE subjects_top_term LIKE '%" + \
        subject + "%' LIMIT 10;"
    query_results = pd.read_sql_query(query, con)
    output_list = list(query_results['subjects_top_term'].values)
    print(output_list)
    return jsonify(output_list)


@app.route('/db_fancy')
def bills_page_fancy():
    sql_query = """
                SELECT bill_id, official_title, subjects_top_term
                FROM bills WHERE status=1;
                """
    query_results = pd.read_sql_query(sql_query, con)
    bills = []
    for i in range(0, query_results.shape[0]):
        d = dict(bill_id=query_results.iloc[i]['bill_id'],
                 official_title=query_results.iloc[i]['official_title'],
                 subjects_top_term=query_results.iloc[i]['subjects_top_term'])
        bills.append(d)
    return render_template('bill_list.html', bills=bills)

import json
import os
print(os.getcwd())
import pickle
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import psycopg2
import spacy
from flask import jsonify, render_template, request
from sqlalchemy import create_engine
from wtforms import Form, TextField

from bill_app import app, con
from bill_app.site_utils import apply_read_time, create_read_time_slider
from data_preparation import bill_utils, text_utils
from modeling import model_utils


# spacy en_core_web_lg is too large for AWS server
# disable unnecessary steps in spacy pipeline for speed
nlp = spacy.load('en', disable=['parser', 'tagger', 'textcat'])

MODEL_ROOT = '../../models/'
NLP_MODEL_ROOT = '../../nlp_models/'

model_save_path = os.path.join(MODEL_ROOT, 'undersampled_RandomForestClassifier10_tfidf10000_other22_linux.pickle')
with open(model_save_path, 'rb') as trained_model:
    current_model = pickle.load(trained_model)

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
    read_time = None

    bill_id = request.args.get('bill_id')
    print('BILL ID = {}'.format(bill_id))

    bill_title = request.args.get('bill_title')
    read_time = request.args.get('reading_time')

    if any(x for x in [bill_id, bill_title]):
        bill_df = bill_utils.retrieve_data(
            con, bill_id=bill_id, bill_title=bill_title, subject=None)

        if bill_df.empty:
            # The bill was not found
            return render_template("bill_not_found.html",
                                   bill_id=bill_id,
                                   bill_title=bill_title)

        bill_id = bill_df.bill_id.unique()[0]
        X, info_dict = model_utils.apply_model(
            bill_df, bill_id, model=current_model, 
                tfidf_train=tfidf_model, train=False, 
                word_embeddings=word_embeddings, 
                embedding_size=embedding_size, get_vecs=True, nlp_lib=nlp)

        # Determine approximate read time properties
        X = apply_read_time(X)
        
        # Set read_time slider properties
        read_time_slider = create_read_time_slider(X, read_time)

        print(info_dict.keys())
        return render_template("output.html",
                               summarization_result=X[['time_cumulative', 'tag',
                                                       'tag_rank', 'text']],
                               bill_info=info_dict,
                               read_time_slider=read_time_slider)
    else:
        # If no summary returned
        empty_df = pd.DataFrame(columns=['tag', 'tag_rank', 'text'])
        read_time_slider = {'min': 0.5, 'max': 10, 'current': 5}
        return render_template("output.html",
                               summarization_result=empty_df,
                               bill_info=None,
                               read_time_slider=read_time_slider)


@app.route('/api/bills/id/<bill_id>', methods=['GET'])
def get_bills_by_id(bill_id):

    query = "SELECT bill_id FROM bills WHERE bill_id LIKE '%" + bill_id + "%' LIMIT 10;"
    query_results = pd.read_sql_query(query, con)
    output_list = list(query_results['bill_id'].values)
    return jsonify(output_list)


@app.route('/api/bills/title/<title>', methods=['GET'])
def get_bills_by_title(title):

    query = "SELECT official_title FROM bills WHERE official_title LIKE '%" + \
        title + "%' LIMIT 10;"
    query_results = pd.read_sql_query(query, con)
    output_list = list(query_results['official_title'].values)
    return jsonify(output_list)


@app.route('/api/bills/subject/<subject>', methods=['GET'])
def get_bills_by_subject(subject):

    query = "SELECT subjects_top_term FROM bills WHERE subjects_top_term LIKE '%" + \
        subject + "%' LIMIT 10;"
    query_results = pd.read_sql_query(query, con)
    output_list = list(query_results['subjects_top_term'].values)
    return jsonify(output_list)

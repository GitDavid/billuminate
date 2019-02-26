from bill_app.site_utils import apply_read_time, create_read_time_slider
from bill_app import app, con
from flask import jsonify, render_template, request
import spacy
import pandas as pd
import os
import sys
sys.path.append('../')
from modeling import model_utils
from data_preparation import bill_utils, text_utils


# spacy en_core_web_lg is too large for AWS server
# disable unnecessary steps in spacy pipeline for speed
nlp = spacy.load('en', disable=['parser', 'tagger', 'textcat'])

MODEL_ROOT = '../../models/'
NLP_MODEL_ROOT = '../../nlp_models/'

mname = 'RFC10_tfidf_numfeat22_linux.pickle'
current_model = model_utils.load_model(os.path.join(MODEL_ROOT, mname))

# This should be saved with model in future!
feature_list = ['page_rank', 'title_word_count', 'char_count', 'word_count',
                'word_density', 'ents', 'title_word_DENSITY', 'doc_word_count',
                'sent_DENSITY', 'tfidf']

tfidf_file_name = 'tfidf_linux.pickle'
tfidf_model = model_utils.load_model(os.path.join(MODEL_ROOT, tfidf_file_name))

em_name = 'lemmatized-legal/no replacement/legal_lemmatized_no_replacement.bin'
word_embeddings = text_utils._load_embeddings_other(
    os.path.join(NLP_MODEL_ROOT, em_name))


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
            bill_df, bill_id, model=current_model, feature_list=feature_list,
            tfidf=tfidf_model, train=False,
            word_embeddings=word_embeddings,
            nlp_lib=nlp)

        # Determine approximate read time properties
        X = apply_read_time(X)

        # Set read_time slider properties
        read_time_slider = create_read_time_slider(X, read_time)

        print(info_dict.keys())
        return render_template("output.html",
                               summarization_result=X[['time_cumulative',
                                                       'tag',
                                                       'tag_rank',
                                                       'text']],
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

    query = "SELECT bill_id FROM bills WHERE bill_id LIKE '%" + bill_id
    query = query + "%' LIMIT 10;"
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

    que = "SELECT subjects_top_term FROM bills WHERE subjects_top_term LIKE '%"
    query = que + subject + "%' LIMIT 10;"
    query_results = pd.read_sql_query(query, con)
    output_list = list(query_results['subjects_top_term'].values)
    return jsonify(output_list)

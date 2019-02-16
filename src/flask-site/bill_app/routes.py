import sys
sys.path.append('/home/ubuntu/repo/billuminate/src/')
from flask import render_template, request, jsonify  # Response
from bill_app import app
import pandas as pd
import psycopg2
from bill_app import models  # feature_utils, bill_utils, apply_model
import json
from wtforms import TextField, Form
from modeling import model_utils
from data_preparation import bill_utils
import spacy


MODEL_ROOT = '../../models/'
NLP_MODEL_ROOT = '../../nlp_models/'
MODEL_ROOT = '/home/ubuntu/insight/bill-summarization/models/'
#'/Users/melissaferrari/Projects/repo/bill-summarization/models/'

print('loading models')
#nlp = spacy.load(NLP_MODEL_ROOT + 'en_core_web_lg')
model_name = 'over_RandomForestClassifier_on_health_nestimators100_random_state0.pkl'
current_model = load_model(MODEL_ROOT + model_name)
tfidf_train = load_model(MODEL_ROOT + 'tifidf_trained.pkl')

print('done loading models')

# Python code to connect to Postgres
user = 'postgres'  # 'melissaferrari'  # add your Postgres username here
# host = '/run/postgresql/' #'localhost'
host = 'localhost'
dbname = 'congressional_bills'
con = psycopg2.connect(database=dbname, user=user, host=host,
                       password='password')


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("bill_search.html",)


@app.route('/output', methods=['GET'])
def bills_output():

    bill_id = request.args.get('bill_id')
    print('BILL ID = {}'.format(bill_id))

    bill_df = retrieve_data(con, bill_id=bill_id, subject=None)

    X, info_dict = apply_model(bill_df, bill_id, model=current_model)

    pred_results = X[(X.prediction == 1) | (X.tag == 'section')].copy()

    return render_template("output.html",
                           summarization_result=pred_results[['tag', 'tag_rank', 'text']],
                           bill_info=info_dict)


@app.route('/api/bills/id/<bill_id>', methods=['GET'])
def get_bills_by_id(bill_id):

    query = "SELECT bill_id FROM bills WHERE bill_id LIKE '%" + bill_id + "%' LIMIT 10;"
    query_results = pd.read_sql_query(query, con)
    output_list = list(query_results['bill_id'].values)
    print(output_list)
    return jsonify(output_list)


@app.route('/api/bills/title/<title>', methods=['GET'])
def get_bills_by_title(title):

    query = "SELECT official_title FROM bills WHERE official_title LIKE '%" + title + "%' LIMIT 10;"
    query_results = pd.read_sql_query(query, con)
    output_list = list(query_results['official_title'].values)
    print(output_list)
    return jsonify(output_list)


@app.route('/api/bills/subject/<subject>', methods=['GET'])
def get_bills_by_subject(subject):

    query = "SELECT subjects_top_term FROM bills WHERE subjects_top_term LIKE '%" + subject + "%' LIMIT 10;"
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

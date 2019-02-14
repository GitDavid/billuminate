from flask import render_template, request, jsonify, Response
from bill_app import app
from sqlalchemy import create_engine
import pandas as pd
import psycopg2
from bill_app import models
import json
from wtforms import TextField, Form


# Python code to connect to Postgres
user = 'postgres'  #'melissaferrari'  # add your Postgres username here
# host = '/run/postgresql/' #'localhost'
host = 'localhost'
dbname = 'congressional_bills'
#db = create_engine('postgres://%s%s/%s' % (user, host, dbname))
#con = None
con = psycopg2.connect(database=dbname, user=user, host=host, password='password')# , port=5433) # host="/var/run/postgresql/" ,password='postgres')




@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("bill_search.html",)

@app.route('/output', methods=['GET'])
def bills_output():

    bill_id = request.args.get('bill_id')
    print('BILL ID = {}'.format(bill_id))

    # just select the bill from the database
    q_info = """
             SELECT bill_id, official_title, subjects_top_term
             FROM bills WHERE bill_id='%s';
             """
    print(q_info)
    q_text = """
             SELECT bill_ix, text FROM bill_text bt
             INNER JOIN bills b ON bt.bill_ix = b.id WHERE b.bill_id='%s';
             """

    q_info_results = pd.read_sql_query(q_info % (bill_id,), con)
    q_text_results = pd.read_sql_query(q_text % (bill_id,), con)

    if len(q_text_results) == 0:
        raise Exception('Oh no! This bill is not in the database.')

    if len(q_text_results) > 1:
        rank_codes = ['ENR', 'EAS', 'EAH', 'RS', 'ES',
                      'PCS', 'EH', 'RH', 'IS', 'IH']
        code = next(i for i in rank_codes if i in
                    q_text_results['code'].unique())
        q_text_results = q_text_results[q_text_results['code'] == code]

    info_dict = q_info_results.loc[0].to_dict()
    text_dict = q_text_results.loc[0].to_dict()

    string_xml = text_dict['text']
    summarization_result = models.do_summarization(string_xml)

    return render_template("output.html",
                           summarization_result=summarization_result,
                           bill_info=info_dict,
                           bill_text=text_dict,)


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

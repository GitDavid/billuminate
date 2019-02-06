from flask import render_template, request, jsonify, Response
from bill_app import app
from sqlalchemy import create_engine
import pandas as pd
import psycopg2
from bill_app import models
import json
from wtforms import TextField, Form


# Python code to connect to Postgres
user = 'melissaferrari'  # add your Postgres username here
host = 'localhost'
dbname = 'congressional_bills'
db = create_engine('postgres://%s%s/%s' % (user, host, dbname))
con = None
con = psycopg2.connect(database=dbname,
                       user=user)


class SearchForm(Form):
    bill_id = TextField('Bill ID', id='bill_autocomplete')


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

    form = SearchForm(request.form)

    return render_template("output.html",
                           summarization_result=summarization_result,
                           bill_info=info_dict,
                           bill_text=text_dict,
                           form=form)


@app.route('/api/bills/<bill_id>', methods=['GET'])
def get_bills(bill_id):

    query = "SELECT bill_id FROM bills WHERE bill_id LIKE '%" + bill_id + "%' LIMIT 10;"
    query_results = pd.read_sql_query(query, con)
    #output_list = list(query_results.bill_id.astype(str).str.cat(query_results.official_title.astype(str), sep=' - '))
    output_list = list(query_results['bill_id'].values)
    print(output_list)
    return jsonify(output_list)


@app.route('/', methods=['GET', 'POST'])
def index():
    form = SearchForm(request.form)
    return render_template("bill_search.html",
                           form=form)


NAMES = ["hr2-114", "hr4-115", "s200-113", "hr20-114"]


@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    return Response(json.dumps(NAMES), mimetype='application/json')
    # return jsonify(matching_results=NAMES)


"""
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search = request.args.get('autocomplete')
    query_list = "SELECT bill_id FROM bills WHERE bill_id LIKE '%"+ search+"%' LIMIT 10;""
    print(query_list)
    # query_list = query_list.replace("%%", "% %s %")
    query_list_results = pd.read_sql_query(query_list % (search,), con)

    df_dict = query_list_results.to_dict()
    print(df_dict)
    return Response(json.dumps(df_dict), mimetype='application/json')
    # jsonify(matching_results=df_dict)

"""


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


# @app.route('/')
@app.route('/bill_input')
def bill_input():
    query_list = "SELECT bill_id FROM bills WHERE (id>=9000) AND (id<10000);"
    query_list_results = pd.read_sql_query(query_list, con)
    bill_id_list = list(query_list_results['bill_id'])
    return render_template("bill_search.html",
                           title='Bill selector',
                           bill_list=bill_id_list)

from flask import render_template, request
from bill_app import app
from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from bill_app import models 


# Python code to connect to Postgres
# You may need to modify this based on your OS,
# as detailed in the postgres dev setup materials.
user = 'melissaferrari'  # add your Postgres username here
host = 'localhost'
dbname = 'congressional_bills'
db = create_engine('postgres://%s%s/%s' % (user, host, dbname))
con = None
con = psycopg2.connect(database=dbname,
                       user=user)


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
                           title='Home',
                           user={'nickname': 'Melissa'},
                           )


@app.route('/db')
def bill_page():
    sql_query = """
                SELECT * FROM bills WHERE status=1;
                """
    
    query_results = pd.read_sql_query(sql_query, con)
    bills = ""
    print(query_results[:10])
    for i in range(0, 10):
        bills += query_results.iloc[i]['official_title']
        bills += "<br>"
    return bills


@app.route('/db_fancy')
def bills_page_fancy():
    sql_query = """
                SELECT bill_id, official_title, subjects_top_term FROM bills WHERE status=1;
                """
    query_results = pd.read_sql_query(sql_query, con)
    bills = []
    for i in range(0, query_results.shape[0]):
        bills.append(dict(bill_id=query_results.iloc[i]['bill_id'],
                          official_title=query_results.iloc[i]['official_title'],
                          subjects_top_term=query_results.iloc[i]['subjects_top_term']))
    return render_template('bill_list.html', bills=bills)


@app.route('/input')
def bills_input():
    query_list = "SELECT bill_id FROM bills;"
    query_list_results = pd.read_sql_query(query_list, con)
    bill_id_list = list(query_list_results['bill_id'])
    return render_template("input.html",
                           title='Bill selector',
                           bill_list=bill_id_list)


@app.route('/output')
def bills_output():
    # pull 'bill_id' from input field and store it
    bill_id = request.args.get('bill_id')
    print('BILL ID = {}'.format(bill_id))
    #bill_id = 'hr4764-114'

    # just select the bill from the database
    query_info = "SELECT bill_id, official_title, subjects_top_term FROM bills WHERE bill_id='%s';" % bill_id
    #query_text = "SELECT bill_ix, text FROM bill_text WHERE bill_ix = (SELECT ls.id FROM bills ls WHERE bill_id='%s');" % bill_id
    query_text = "SELECT bill_ix, text FROM bill_text bt INNER JOIN bills b ON bt.bill_ix = b.id WHERE b.bill_id = '%s';" % bill_id
    
    query_info_results = pd.read_sql_query(query_info, con)
    query_text_results = pd.read_sql_query(query_text, con)
    
    # assert len(query_text_results) == 0 ## Asserts should never happen
    if len(query_text_results) == 0:
        raise Exception('Oh no! This bill is not in the database.')

    if len(query_text_results) > 1:
        raise Exception('More than one bill mapped to this idea. Something is wrong!')

    bills = []
    i = 0 
    for ix in range(1):
        bills.append(dict(bill_id=query_info_results.iloc[i]['bill_id'],
                          official_title=query_info_results.iloc[i]['official_title'],
                          subjects_top_term=query_info_results.iloc[i]['subjects_top_term']))
    
    string_xml = query_text_results.iloc[0]['text']
    summarization_result = models.do_summarization(string_xml)

    return render_template("output.html",
                           summarization_result=summarization_result, 
                           bills=bills)

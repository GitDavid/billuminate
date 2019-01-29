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
    return render_template("input.html")


@app.route('/output')
def bills_output():
    # pull 'bill_id' from input field and store it
    # bill_id = request.args.get('bill_id')
    bill_id = 'hr4764-114'
    # just select the bill from the database
    query_info = "SELECT bill_id, official_title, subjects_top_term FROM bills WHERE bill_id='%s';" % bill_id
    query_text = "SELECT bill_ix, text FROM bill_text WHERE bill_ix = (SELECT ls.id FROM bills ls WHERE bill_id='%s');" % bill_id
    #query = "SELECT bill_ix, text FROM bill_text bt INNER JOIN bills b ON bt.bill_ix = b.id WHERE b.bill_id = '%s';" % bill_id
    
    query_info_results = pd.read_sql_query(query_info, con)
    query_text_results = pd.read_sql_query(query_text, con)
    
    bill_id = query_info_results.iloc[0]['bill_id'],
    official_title = query_info_results.iloc[0]['official_title'],
    subjects_top_term = query_info_results.iloc[0]['subjects_top_term']
    
    string_xml = query_text_results.iloc[0]['text']
    summarization_result = models.do_summarization(string_xml)

    return render_template("output.html",
                           summarization_result=summarization_result, 
                           bill_id=bill_id,
                           official_title=official_title,
                           subjects_top_term=subjects_top_term)

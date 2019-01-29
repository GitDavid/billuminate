from flask import render_template, request
from bill_app import app
from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from bill_app.a_Model import ModelIt


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
def birth_page():
    sql_query = """
                SELECT * FROM bills WHERE status=1;
                """
    # SELECT * FROM birth_data_table WHERE delivery_method='Cesarean';
    query_results = pd.read_sql_query(sql_query, con)
    births = ""
    print(query_results[:10])
    for i in range(0, 10):
        births += query_results.iloc[i]['official_title']
        births += "<br>"
    return births


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
def cesareans_input():
    return render_template("input.html")


@app.route('/output')
def cesareans_output():
    # pull 'birth_month' from input field and store it
    patient = request.args.get('birth_month')
    # just select the Cesareans  from the birth dtabase for
    # the month that the user inputs
    query = "SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean' AND birth_month='%s'" % patient
    print(query)
    query_results = pd.read_sql_query(query, con)
    print(query_results)
    births = []
    for i in range(0, query_results.shape[0]):
        births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
        the_result = ModelIt(patient, births)
    return render_template("output.html",
                           births=births,
                           the_result=the_result)

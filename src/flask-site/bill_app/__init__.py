from flask import Flask
from config import Config
import psycopg2
import os

app = Flask(__name__)
app.config.from_object(Config)
url = app.config['URL']
db_conn_string = "dbname={} user={} password={} host={}"
db_conn = db_conn_string.format(url.path[1:], url.username, url.password, url.hostname)

con = psycopg2.connect(db_conn)

from bill_app import routes, errors, site_utils

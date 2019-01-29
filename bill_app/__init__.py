from flask import Flask

app = Flask(__name__)

from bill_app import routes

import os
from dotenv import load_dotenv
import urllib

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.flaskenv'))

class Config(object):
    CSRF_ENABLED = True
    SECRET_KEY = os.environ.get('SECRET_KEY')
    URL = urllib.parse.urlparse(os.environ.get('DATABASE_URL'))
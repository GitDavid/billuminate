import sys
if sys.platform == "linux":
    sys.path.append('/home/ubuntu/repo/billuminate/src/')
    sys.path.append('/media/swimmers3/ferrari_06/repo/billuminate/src/')

elif sys.platform == "darwin":
    sys.path.append('/Users/melissaferrari/Projects/repo/billuminate/src/')

MODEL_ROOT = '../../models/'
NLP_MODEL_ROOT = '../../nlp_models/'
TRAINING_DATA_ROOT = '../../data/training_data/'






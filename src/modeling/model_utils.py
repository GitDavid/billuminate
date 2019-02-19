import sys
if sys.platform == "linux":
    sys.path.append('/home/ubuntu/repo/billuminate/src/')
elif sys.platform == "darwin":
    sys.path.append('/Users/melissaferrari/Projects/repo/billuminate/src/')


from scipy import sparse
import os
import numpy as np
from data_preparation import feature_utils, bill_utils
import pickle


TRAINING_DATA_ROOT = '../../data/training_data/'
MODEL_ROOT = '../../models/'
NLP_MODEL_ROOT = '../../nlp_models/'


def load_trained_tfidf(file_path=None, subject='health'):

    if not file_path:
        file_name = 'tfidf_{}.npz'.format(subject.lower())
        file_path = os.path.join(MODEL_ROUTE, file_name)

    tfidf_train = sparse.load_npz(file_path)

    return tfidf_train


def load_model(model_save_path):
    with open(model_save_path, 'rb') as training_model:
        model = pickle.load(training_model)
    return model


def _apply_tfidf(text_col, tfidf_train):
    tfidf = tfidf_train.transform(text_col)
    return tfidf


def join_features(feature_list):

    if any(sparse.issparse(x) for x in feature_list):
        all_features = sparse.hstack(feature_list)

    else:
        all_features = np.hstack(feature_list)

    return all_features


def get_bill_dict(bills_info, bill_id):
    bill = bills_info[bills_info['bill_id'] == bill_id].copy()
    bill = bill_utils._return_correct_version(bill, as_dict=True)
    return bill

##quick requests/beautifulsoup
##https://glowingpython.blogspot.com/2014/09/text-summarization-with-nltk.html
def apply_model(bills_info, bill_id, model=None, tfidf_train=None, train=False, 
                word_embeddings=False, embedding_size=False, get_vecs=False, nlp_lib=False):

    bill = get_bill_dict(bills_info, bill_id)

    X = feature_utils.prepare_features(bill, train=train, 
                                       word_embeddings=word_embeddings, 
                                       embedding_size=embedding_size, 
                                       get_vecs=get_vecs, nlp_lib=nlp_lib)
    #print(X.columns)
    #print(X.shape)
    # CURRENT
    # ['loc_ix', 'tag', 'text', 'tag_rank', 'bill_id', 'clean_text', 'abs_loc',
    #    'norm_loc', 'title_word_count', 'char_count', 'word_count',
    #    'word_density']

    #print(X['clean_text'])

    features = X.drop(columns=['loc_ix', 'tag', 'text',
                            'clean_text', 'bill_id']).copy()


    if tfidf_train:
        tfidf_mat = tfidf_train.transform(X['clean_text'])
        feature_list = [features, tfidf_mat]
    else:
        feature_list = [features]
    X_numeric = join_features(feature_list)

    #assert X_numeric.shape[1] == model.n_features_
    y_pred = model.predict(X_numeric)
    y_probs = model.predict_proba(X_numeric)

    #print(y_pred.shape)
    #print(y_probs.shape)
    X['prediction'] = y_pred
    X['predict_proba0'] = y_probs[:,0]
    X['predict_proba1'] = y_probs[:,1]

    return X, bill


def render_final_text(X):

    pred_results = X[(X.prediction == 1) | (X.tag == 'section')].copy()

    for ix, row in pred_results.iterrows():
        print(row['abs_loc'], row['tag'], int(row['tag_rank']))
        print(row['text'])
        print()

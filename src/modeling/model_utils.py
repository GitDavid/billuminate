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


def apply_model(bills_info, bill_id, model=None, tfidf_train=None):

    bill = get_bill_dict(bills_info, bill_id)

    X = feature_utils.prepare_features(bill)
    print(X.shape)
    features = X.drop(columns=['loc_ix', 'tag', 'text',
                               'clean_text', 'bill_id'])

    feature_list = [features, ]
    if tfidf_train:
        tfidf_mat = tfidf_train.transform(X['clean_text'])
        feature_list.append(tfidf_mat)

    X_numeric = join_features(feature_list)

    assert X_numeric.shape[1] == model.n_features_
    y_pred = model.predict(X_numeric)

    X['prediction'] = y_pred

    return X, bill


def render_final_text(X):

    pred_results = X[(X.prediction == 1) | (X.tag == 'section')].copy()

    for ix, row in pred_results.iterrows():
        print(row['abs_loc'], row['tag'], int(row['tag_rank']))
        print(row['text'])
        print()

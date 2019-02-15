from scipy import sparse
import os
import numpy as np


TRAINING_DATA_ROOT = '../../data/training_data/'
MODEL_ROUTE = '../../models/'
NLP_MODEL_ROOT = '../../nlp_models/'


def load_trained_tfidf(file_path=None, subject='health'):

    if not file_path:
        file_name = 'tfidf_{}.npz'.format(subject.lower())
        file_path = os.path.join(MODEL_ROUTE, file_name)

    tfidf_train = sparse.load_npz(file_path)

    return tfidf_train


def apply_tfidf(text_col, tfidf_train):

    tfidf = tfidf_train.fit(text_col)

    return tfidf


def join_features(feature_list):

    if any(sparse.issparse(x) for x in feature_list):
        all_features = sparse.hstack(feature_list)

    else:
        all_features = np.hstack(feature_list)

    return all_features


def main(df):

    tfidf_train = load_trained_tfidf(file_path=None, subject='health')

    tfidf_mat = apply_tfidf(df['clean_text'], tfidf_train)

    feature_list = [tfidf_mat, X_match.drop(columns=['bill_id'])]




if __name__ == "__main__":
    main()

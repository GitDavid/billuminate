import pickle

import sys
sys.path.append('../')
from data_preparation import bill_utils, feature_utils


def load_model(model_save_path):
    with open(model_save_path, 'rb') as training_model:
        model = pickle.load(training_model)
    return model


def get_bill_dict(bills_info, bill_id):
    bill = bills_info[bills_info['bill_id'] == bill_id].copy()
    bill = bill_utils._return_correct_version(bill, as_dict=True)
    return bill


def apply_model(bills_info, bill_id, model=None,
                feature_list=None, tfidf=None,
                train=False, word_embeddings=False,
                nlp_lib=False):

    bill = get_bill_dict(bills_info, bill_id)
    bill_df, feature_space = feature_utils.generate_feature_space(
        bill, feature_list, train=train, word_embeddings=word_embeddings,
        nlp_lib=nlp_lib, tfidf=tfidf)

    X_features = feature_utils.join_features(feature_space)
    bill_df = bill_df.merge(feature_space[0],
                            on=['tag_rank', 'abs_loc', 'norm_loc'])

    y_pred = model.predict(X_features)
    y_probs = model.predict_proba(X_features)

    bill_df['prediction'] = y_pred
    bill_df['predict_proba0'] = y_probs[:, 0]
    bill_df['predict_proba1'] = y_probs[:, 1]

    return bill_df, bill

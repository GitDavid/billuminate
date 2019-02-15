import os
import pickle

import numpy as np
import itertools
import matplotlib.pylab as plt

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import (CountVectorizer,
                                             # HashingVectorizer,
                                             TfidfEmbeddingVectorizer,
                                             TfidfTransformer)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (classification_report,
                             confusion_matrix)
# from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline


TRAINING_DATA_ROOT = '../../data/training_data/'
MODEL_ROUTE = '../../models/'
NLP_MODEL_ROOT = '../../nlp_models/'


def train_basic_classifier(classifier,
                           X_train, X_test, y_train, y_test):

    classifier.fit(X_train, y_train)
    print("Accuracy: {} ".format(classifier.score(X_test, y_test)))
    return classifier


# def feature_preparation(X_match, df_match_):
#     # all_feat = sparse.hstack([X_match.drop(columns=['bill_id']), tfidf_mat,  ])
#     # feat_array = all_feat.toarray()
#     feat_array = np.hstack([X_match.drop(columns=['bill_id']), df_match_])
#     feat_array_ = np.nan_to_num(feat_array)
#     return feat_array_


# def custom_pipelines(MeanEmbeddingVectorizer, embeddings):
#     etree_w2v = Pipeline([
#         ("word2vec vectorizer", MeanEmbeddingVectorizer(embeddings)),
#         ("extra trees", ExtraTreesClassifier(n_estimators=200))])
#     etree_w2v = Pipeline([
#         ("word2vec vectorizer", TfidfEmbeddingVectorizer(embeddings)),
#         ("extra trees", ExtraTreesClassifier(n_estimators=200))])
#     return etree_w2v, etree_w2v


def calculate_metrics(model, X_test, y_test, to_df=True):
    report = classification_report(y_test, model.predict(X_test),
                                   output_dict=True)
    if to_df:
        report = pd.DataFrame(report).transpose()
    return report


def generate_confusion_matrix(y_test, y_pred):
    return confusion_matrix(y_test, y_pred)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def save_model(model, model_save_path, save_report=True):
    with open(model_save_path, 'wb') as picklefile:
        pickle.dump(model, picklefile)
    if save_report:
        report_name = model_save_path.replace(model_save_path.split(".")[-1],
                                              '_classification_report.csv')
        save_report.to_csv(report_name)


def save_sparse_matrix(sparse_matrix, save_path):
    sparse.save_npz(save_path, sparse_matrix)


def load_model(model_save_path):
    with open(model_save_path, 'rb') as training_model:
        model = pickle.load(training_model)
    return model


def create_pipeline():
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                              alpha=1e-3, random_state=42,
                              max_iter=5, tol=None)), ])
    return text_clf


def parameter_tuning(text_clf):
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3),
        }
    gs_clf = GridSearchCV(text_clf, parameters,
                          cv=5, iid=False, n_jobs=-1)
    best_score = gs_clf.best_score_
    return gs_clf, best_score


def get_sense_spec_ixs(y_test):

    assert all(col in y_test.columns for col in ['in_summary', 'predictions'])

    true_positives = y_test[(y_test['in_summary'] == y_test['predictions']) &
                            (y_test['in_summary'] == 1)].copy()
    ix_tp = true_positives.index.values

    true_negatives = y_test[(y_test['in_summary'] == y_test['predictions']) &
                            (y_test['in_summary'] == 0)].copy()
    ix_tn = true_negatives.index.values

    false_negatives = y_test[(y_test['in_summary'] != y_test['predictions']) &
                             (y_test['in_summary'] == 1)].copy()
    ix_fn = false_negatives.index.values

    false_positives = y_test[(y_test['in_summary'] != y_test['predictions']) &
                             (y_test['in_summary'] == 0)].copy()
    ix_fp = false_positives.index.values

    sense_spec_dict = {'tp': ix_tp, 'tn': ix_tn, 'fp': ix_fp, 'fn': ix_fn}

    return sense_spec_dict


def get_sense_spec_df(df, sense_spec_dict):

    df['sense_spec'] = df.index.map(sense_spec_dict)

    return df


def main():

    training_files = os.listdir(TRAINING_DATA_ROOT)

    data = pd.read_csv(TRAINING_DATA_ROOT + training_files[0])
    del data['Unnamed: 0']

    X = data.drop(columns=['in_summary'])
    y = data[['in_summary']]

    (X_train,
     X_test,
     y_train,
     y_test) = train_test_split(X, y, test_size=0.25, random_state=33)

    ## Define classifier
    classifier = LogisticRegression(C=1e5, solver='lbfgs')
    #classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
    #classifier = SGDClassifier(loss='hinge', penalty='l2',
    #                           alpha=1e-3, random_state=42,
    #                           max_iter=5, tol=None)
    mdl = basic_classifier_train(classifier, X, y)

    mdl.predict(X_test)


if __name__ == "__main__":
    main()

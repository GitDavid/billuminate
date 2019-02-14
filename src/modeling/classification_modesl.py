import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import HashingVectorizer, TfidfEmbeddingVectorizer, TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


PATH_TO_TRAINING = '../data/training_data/'
PATH_TO_MODELS = '../../models/'
PATH_TO_NLP_MODELS = '../../nlp_models/'


def leglove():
    with open(PATH_TO_NLP_MODELS + 'LeGlove.model', 'rb') as f:
        embeddings = pickle.load(f, encoding='latin-1')
    return embeddings

class MeanEmbeddingVectorizer(object):
    # Example on how to use my own prepocesing in the pipeline
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


def custom_pipelines(MeanEmbeddingVectorizer, embeddings):
    etree_w2v = Pipeline([
        ("word2vec vectorizer", MeanEmbeddingVectorizer(embeddings)),
        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
    etree_w2v = Pipeline([
        ("word2vec vectorizer", TfidfEmbeddingVectorizer(embeddings)),
        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
    return etree_w2v, etree_w2v


def basic_classifier_train(classifier,
                           X_train, y_train,
                           X_test, y_test):

    classifier.fit(X_train, y_train)
    print("Accuracy: {} ".format(classifier.score(X_test, y_test)))
    return classifier


def calculate_metrics(model, X_test, y_test,
                      as_csv=True, save=False,
                      report_name="report.csv"):
    report = classification_report(y_test, model.predict(X_test),
                                   output_dict=True)
    if as_csv:
        report = pd.DataFrame(report).transpose()
    return report


def save_model(model, model_save_path, save_report=True):
    with open(model_save_path, 'wb') as picklefile:
        pickle.dump(model, picklefile)
    if save_report:
        report_name = model_save_path.replace(model_save_path.split(".")[-1],
                                              '_classification_report.csv')
        save_report.to_csv(report_name)


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


def main():

    training_files = os.listdir(PATH_TO_TRAINING)

    data = pd.read_csv(PATH_TO_TRAINING + training_files[0])
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
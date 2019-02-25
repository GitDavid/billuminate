import sys
sys.path.append('../')

from networkx import nx
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
from data_preparation import bill_utils
import numpy as np
from scipy import sparse


def load_trained_tfidf(file_path):

    tfidf_train = sparse.load_npz(file_path)

    return tfidf_train


def load_model(model_save_path):
    with open(model_save_path, 'rb') as training_model:
        model = pickle.load(training_model)
    return model


def title_word_count(x, joint_title):
    return len([wrd for wrd in x.split() if wrd in joint_title])


def char_count(x):
    return len(x)


def word_count(x):
    return len(x.split())


def word_density(x):
    return np.divide(char_count(x), word_count(x) + 1)


def title_word_density(x, joint_title):
    return np.divide(title_word_count(x, joint_title), word_count(x) + 1)


def doc_word_count(x):
    return sum(word_count(x))


def sent_density(x):
    return np.divide(word_count(x), doc_word_count(x) + 1)


# def apply_pagerank(x, vector_embeddings):
def apply_pagerank(vector_embeddings):

    """Performs pagerank on list of embeddings.

    Args:
        vector_embeddings: List of vector embeddings to process

    Returns:
        page_ranks: A list of scoring each vector embeddings
            rank. If the graph isn not able to converge, it will
            just return an equal weighting for each sentence.
    """

    vlen = len(vector_embeddings)
    try:
        sim_mat = cosine_similarity(vector_embeddings)

        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph, max_iter=100)
        page_ranks = [x for k, x in scores.items()]
    except ValueError:
        print('TextRank could not converge')
        page_ranks = [1/vlen]*vlen
    return page_ranks


def apply_ents_torows(x, nlp):

    """Performs named entity recognition on single sentence.
    NOT IN USE right now but will be required to streamline
    generate_feature_space feature_dict.
    """

    ENT_TYPES = ['LAW', 'ORDINAL', 'GPE', 'DATE',
                 'MONEY', 'LAW', 'EVENT', 'PRODUCT', 'NORP']

    for doc in nlp.pipe(x):
        ent_list = [ent.label_ for ent in doc.ents]
        ent_array = [ent_list.count(ENT) for ENT in ENT_TYPES]

    return ent_array


def apply_ents(sentences, nlp, df_index=[], ENT_TYPES=None):

    """Performs named entity recognition on list of sentences.

    Performs named entity recognition on list of sentences, filtered
    on desired entity types and returns a dataframe of the count.

    Args:
        sentences: List of sentences to process
        nlp: nlp model to be used for recognizing named entities.
        df_index: Optional list of indices to pass to entity dataframe
            so that it can be merged easily with bill dataframe
        ENT_TYPES: List of entity types to count.

    Returns:
        df_ent: DataFrame of entities (cols) and their corresponding
            counts in each of the sentences (rows). Also the total
            count of entities in ENT_TYPES.
    """

    if not ENT_TYPES:
        ENT_TYPES = ['LAW', 'ORDINAL', 'GPE', 'DATE',
                     'MONEY', 'LAW', 'EVENT', 'PRODUCT', 'NORP']

    ent_array = np.zeros((sentences.shape[0], len(ENT_TYPES)), dtype=int)
    for ix, sentence in enumerate(sentences):
        for doc in nlp.pipe(sentence):
            ent_list = [ent.label_ for ent in doc.ents]
            ent_array[ix, :] = [ent_list.count(ENT) for ENT in ENT_TYPES]

    if len(df_index) == 0:
        df_index = range(len(sentences.shape[0]))

    df_ent = pd.DataFrame(ent_array,
                          index=df_index,
                          columns=['ent_{}'.format(ENT_TYPE)
                                   for ENT_TYPE in ENT_TYPES])
    df_ent['ent_TOTAL'] = np.sum(ent_array, axis=1)

    return df_ent


def generate_feature_space(bill, feature_list, train=False,
                           word_embeddings=False,
                           nlp_lib=False, tfidf=None):

    """Generates features for a particular bill.

    Generates features for a bill given a list of features provided.

    Args:
        bill: A dictionary representation of a single bill
        feature_list: A list of features to be determined for the
            bill.
        train: Whether features generated are for training purposes or
            for applying model on new data. Default is False.
        word_embeddings: Dictionary with keys 'embeddings' for the trained
            word2vec model to be applied on text data and 'size' for the
            length of the embedding vector. Got rid of 'get_vecs' because
            will assume word_embeddings supplied means to get vecs
        nlp_lib: The nlp pipeline to use.
        tfidf: The trained tf-idf model to apply to text features.

    Returns:
        Tuple (bill_df, list_of features)
        bill_df: A dataframe representation of segmented data including
            columns:
                text, clean_text, tag, tag_ranking, loc_ix, abs_loc, norm_loc
        list_of_features: A list of distinct feature types. Straightforward
            features are separated from sparse features like tf-idf at this
            point. Ex. [feature_space, tfidf] or [feature_space,]
    """

    official_title = bill['official_title'].lower()
    short_title = bill['short_title'].lower()
    joint_title = '{} {}'.format(official_title, short_title)

    bill_df, fvecs = bill_utils.generate_bill_data(
        bill, train=train, word_embeddings=word_embeddings,)
    bill_df['clean_text'] = bill_df['clean_text'].fillna("")

    feature_space = bill_df[['tag_rank', 'abs_loc', 'norm_loc']].copy()

    feature_dict = {
        'title_word_count': [title_word_count, 'clean_text',
                             {'joint_title': joint_title}, False],
        'char_count': [char_count, 'clean_text', {}, False],
        'word_count': [word_count, 'clean_text', {}, False],
        'word_density': [word_density, 'clean_text', {}, False],
        'title_word_DENSITY': [title_word_density, 'clean_text',
                               {'joint_title': joint_title}, False]}

    for feature in feature_list:

        if feature == 'ents':
            sentences = bill_df['clean_text'].values
            df_ent = apply_ents(sentences, nlp_lib, bill_df.index.values)
            feature_space = feature_space.merge(
                df_ent, left_index=True, right_index=True)
            feature_space['ent_DENSITY'] = feature_space['ent_TOTAL'].div(
                feature_space['word_count'].where(
                    feature_space['word_count'] != 0, np.nan))

        elif feature == 'tfidf':
            tfidf_mat = tfidf.transform(bill_df['clean_text'])
            feature_space = feature_space.fillna(0)
            return bill_df, [feature_space, tfidf_mat, ]

        elif feature == 'doc_word_count':
            # 'doc_word_count': [doc_word_count, 'clean_text', {}, False]
            feature_space["doc_word_count"] = feature_space["word_count"].sum()

        elif feature == 'sent_DENSITY':
            # 'sent_DENSITY': [sent_density, 'clean_text', {}, False]
            feature_space['sent_DENSITY'] = feature_space['word_count'].div(
                feature_space['doc_word_count'].where(
                    feature_space['doc_word_count'] != 0, np.nan))

        elif feature == 'page_rank':
            # 'page_rank': [apply_pagerank, '',
            #               {'vector_embeddings':fvecs}, False]
            feature_space[feature] = apply_pagerank(fvecs)

        else:
            func, col, args, _ = feature_dict[feature]
            # if not is_list:
            feature_space[feature] = bill_df[col].apply(func, **args)
            # 'ents': [apply_ents, 'clean_text', [nlp], True]

    feature_space = feature_space.fillna(0)

    return bill_df, [feature_space, ]


def join_features(generated_features_list):

    """Combines multiple feature types.

    Takes in list of generated features and concatenates them.

    Args:
        generated_features_list: A list of distinct feature types to
            be joined. Ex. Normal array and sparse array features.
            Ex. [feature_space, tfidf]

    Returns:
        all_features: An array of all features.
    """

    if any(sparse.issparse(x) for x in generated_features_list):
        all_features = sparse.hstack(generated_features_list)

    else:
        all_features = np.hstack(generated_features_list)

    return all_features

import random
import re
import string

import numpy as np
from gensim.models import KeyedVectors
import nltk
import pandas as pd
#import rouge
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------- #
# ----- TEXT CLEANING ----- #
# ------------------------- #

def _make_lowercase(sentences):
    return [s.lower() for s in sentences]


def _general_text_cleaning(text):

    # deal with numbered lists
    # r"^\d+\.\s"
    # r"\([i]+\)"
    # r"[i]+\."
    # r"\([A-Z]\)"
    # r"\([a-z]\)"
    # r"\([\d]\)"

    text = re.sub(r"\'s", "", text)
    text = re.sub(r" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"e\.g\.", " eg ", text, flags=re.IGNORECASE)

    text = re.sub(r"(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text,
                  flags=re.IGNORECASE)
    text = re.sub(r"(the[\s]+|The[\s]+)?United State(s)?", " America ", text,
                  flags=re.IGNORECASE)

    # remove comma between numbers
    text = re.sub(r"(?<=[0-9])\,(?=[0-9])", "", text)

    text = re.sub(r"\$", " dollar ", text)
    text = re.sub(r"\%", " percent ", text)
    text = re.sub(r"\&", " and ", text)

    # for hyphenated
    text = re.sub(r"[a-zA-Z0-9\-]*-[a-zA-Z0-9\-]*", "".join(text.split("-")),
                  text)

    # the single 's' in this stage is 99% of not clean text, just kill it
    text = re.sub(r" s ", " ", text)

    # reduce extra spaces into single spaces
    text = re.sub(r"[\s]+", " ", text)

    return text


def _remove_punct(sentences):
    # remove punctuations and special characters
    regex = re.compile(r"[^a-zA-Z0-9]")
    return [regex.sub(" ", s) for s in sentences]


def _tokenize_words(txt_string, nlp):
    txt_string = txt_string.replace(',', "")
    doc = nlp(txt_string)
    txt_words = [sent for sent in doc]
    return doc, txt_words


def _tokenize_sentences(txt_string, nlp):
    txt_string = txt_string.replace('\n\n', ". ")
    doc = nlp(txt_string)
    txt_sent = [sent.string.strip() for sent in doc.sents]
    return doc, txt_sent


def tokenize_summ(summary_text, nlp, short_title):
    _, summ_sent = _tokenize_sentences(summary_text, nlp)
    if short_title.lower() in summ_sent[0].lower():
        summ_sent = summ_sent[1:]
    return summ_sent


def _apply_text_cleaning(sent):

    sent_clean = [_general_text_cleaning(s) for s in sent]
    sent_clean = _remove_punct(sent_clean)
    sent_clean = _make_lowercase(sent_clean)

    return sent_clean


def _remove_short_words(sentence):
    # Not working properly -- need to work on this
    sentence = " ".join(re.compile(r"\b\w\w+\b", re.U).findall(sentence))
    return sentence


def _full_text_tostring(txt_extract, omit_tags=['header', 'short-title']):
    # No longer in use -- keeping just in case I am wrong
    sentences = []
    enum_string = ""
    full_string = ""
    for ix, row in txt_extract[~txt_extract['tag'].isin(omit_tags)].iterrows():
        nextix = ix
        enum_string += row['text'] + ' '
        if row['tag'] == 'enum':
            nextix = ix + 1
        if nextix == ix:
            sentences.append(row['text'] + ' ')
            full_string += row['text'] + ' '
    return full_string, enum_string, sentences


def _remove_whitespace(sentence_list):
    # No longer in use -- keeping just in case I am wrong
    white_space = list(string.whitespace)[1:]
    for ix in range(len(sentence_list)):
        for bad_string in white_space:
            if bad_string in sentence_list[ix]:
                sentence_list[ix] = sentence_list[ix].replace(bad_string, "")
    return sentence_list


def _stemming_tokenizer(text):
    # No longer in use -- keeping just in case I am wrong
    stemmer = nltk.stem.PorterStemmer()
    return [stemmer.stem(w) for w in nltk.word_tokenize(text)]


def _remove_custom(sentence_list, type='sec'):
    for ix in range(len(sentence_list)):
        s = sentence_list[ix].lower()
        start_ix = s.find("(sec.")
        if start_ix != -1:
            end_ix = start_ix + 8
            sentence_list[ix] = sentence_list[ix].replace(s[start_ix:end_ix],
                                                          " ")
    return sentence_list


# --------------------------- #
# ----- TEXT TO NUMBERS ----- #
# --------------------------- #

def _load_embeddings(path_to_embedding, encoding=None):

    print(path_to_embedding)
    f = open(path_to_embedding)
    embeddings = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings[word] = coefs
    f.close()

    embedding_size = list(coefs.shape)[0]
    word_embeddings = {'embeddings': embeddings,
                       'size': embedding_size}
    return word_embeddings


def _load_embeddings_other(path_to_embedding, binary=True,
                           encoding='latin-1'):

    print(path_to_embedding)
    wmodel = KeyedVectors.load_word2vec_format(path_to_embedding,
                                               binary=binary,
                                               encoding=encoding)
    embeddings = {}
    for idx, key in enumerate(wmodel.vocab):
        embeddings[key] = wmodel.get_vector(key)

    embedding_size = list(embeddings[key].shape)[0]

    word_embeddings = {'embeddings': embeddings,
                       'size': embedding_size}

    return word_embeddings


def _calc_embedding(sen, word_embeddings):

    embeddings = word_embeddings['embeddings']
    size = word_embeddings['size']

    if not size:
        size = random.choice(list(embeddings.values())).shape

    if len(sen) != 0:
        try:
            vector = sum([embeddings.get(w, np.zeros(size))
                          for w in sen.split()])/(len(sen.split())+0.001)
        except TypeError:
            sen_emb = []
            for w in sen.split():
                try:
                    word = embeddings['dictionary'][w]
                    e = embeddings['word_vectors'][word]
                except ValueError:
                    e = np.zeros((100,)).shape
                sen_emb.append(e)
                vector = sum(sen_emb)/(len(sen.split()) + 0.001)

    else:
        # If no embedding exists. Return array of zeros.
        vector = np.zeros(size)
    return vector


def _calc_embeddings_set(sents, word_embeddings):

    sent_vects = [_calc_embedding(s, word_embeddings)
                  for s in sents]

    return sent_vects


# ----------------------------- #
# ----- TEXT MEASUREMENTS ----- #
# ----------------------------- #

def _create_sim_mat(vecs_1, vecs_2, embedding_size):
    sim_mat = np.zeros([len(vecs_1), len(vecs_2)])
    sim_mat = cosine_similarity(vecs_1, vecs_2)
    return sim_mat


def _create_rouge_mat(sents_1, sents_2, max_n=1, metrics=['rouge-n'],
                      weight_factor=1.2, stemming=True):
    metric = 'rouge-{}'.format(int(max_n))
    evaluator = rouge.Rouge(metrics=metrics,
                            max_n=max_n,
                            limit_length=False,
                            alpha=0.5,
                            weight_factor=weight_factor,
                            stemming=stemming)
    rouge_mat = np.zeros([len(sents_1), len(sents_2)])
    for i in range(len(sents_1)):
        for j in range(len(sents_2)):
            rouge_mat[i][j] = evaluator.get_scores(sents_2[j],
                                                   sents_1[i])[metric]['f']
    return rouge_mat


def _sort_matrix_ix(sim_mat, num_rows=1, axis=0):

    ix_sort = (-sim_mat).argsort(axis=axis)
    ix_match = ix_sort[0:num_rows, :]
    ix_match = np.sort(ix_match.flatten())
    ix_match = np.unique(ix_match)
    return ix_sort, ix_match

# import matplotlib
# import matplotlib.pylab as plt

import random
import re
import string
import xml.etree.ElementTree as ET

import numpy as np

import pandas as pd
# import sqlalchemy_utils
import psycopg2
import spacy
import sqlalchemy
from sklearn import metrics


def _clean_extracted_list(txt_extract,
                          tag_rankings=None):
    
    if not tag_rankings:
         tag_rankings = {'bill':0, 'title': 1, 'section':2, 'subsection':3, 'paragraph':4, 
                'subparagraph':5, 'clause':6, 'subclause':7, 'item':8, 'subitem':9, 'subsubitem':10}

    txt_df = pd.DataFrame(txt_extract)
    txt_df.columns = ['loc_ix', 'tag', 'text']    
    txt_df['tag_rank']  = txt_df['tag'].map(tag_rankings)

    # Drop pagebreak tag bc it causes errors
    txt_df = txt_df[txt_df.tag != 'pagebreak']

    # Drop header section and titles
    ix_min = txt_df[txt_df['tag']=='legis-body'].index.values[0]+1
    txt_df = txt_df.drop(txt_df.index[np.arange(ix_min)])

    # Add enumeration to front of each list item
    num_ixs = txt_df[txt_df['tag']=='enum']['loc_ix'].values
    for ix in num_ixs:
        txt_df.loc[ix+1, 'text'] = txt_df.reindex(range(ix,ix+2))['text'].str.cat(sep=' ')
        txt_df = txt_df.drop(ix)

    txt_df = txt_df[txt_df.tag != 'quoted-block']
    txt_df = txt_df[txt_df.tag != 'after-quoted-block']
    """
    #Should label quoted blocks in future
    ## Concat the quote-blocks
    min_ixs = txt_df[txt_df.tag == 'quoted-block']['loc_ix'].values
    max_ixs = txt_df[txt_df.tag == 'after-quoted-block']['loc_ix'].values

    # Catch quote blocks in quote blocks
    if any(min_ixs[1:] < max_ixs[:-1]):
        for ix in range(len(min_ixs)-1):
            if min_ixs[ix+1] < max_ixs[ix]:
                min_ixs = np.delete(min_ixs, ix+1)
                max_ixs = np.delete(max_ixs, ix)

    for ix_loc in range(len(min_ixs)):
        txt_df.loc[min_ixs[ix_loc], 'text'] = txt_df.reindex(range(min_ixs[ix_loc]+1,max_ixs[ix_loc]+1))['text'].str.cat(sep=' ')
        txt_df = txt_df.drop(np.arange(min_ixs[ix_loc]+1,max_ixs[ix_loc]+1), errors='ignore')
    """
    section_ix = txt_df[txt_df['tag'] == 'section']['loc_ix'].values
    # Collapse section text
    try:
        assert all(txt_df.reindex(section_ix)['tag'] == 'section')
        assert all(txt_df.reindex(section_ix+2)['tag'] == 'header')
        txt_df.loc[section_ix, 'text'] = txt_df.reindex(section_ix+2)['text'].values
        drop_list = np.append(section_ix + 1, section_ix + 2)
        txt_df = txt_df.drop(drop_list, errors='ignore')
    except: 
        diff = 1
        while section_ix.size !=0:
            inds = txt_df.reindex(section_ix+diff).dropna(subset=['loc_ix']).index.values
            if inds.size !=0:
                txt_df.loc[inds - diff, 'text'] = txt_df.reindex(inds)['text'].values
                txt_df = txt_df.drop(inds)
                rm_ix = txt_df.reindex(inds-diff)['loc_ix'].values
                section_ix = np.array(list(filter(lambda x: x not in rm_ix, section_ix)))
            diff += 1

    # Collapse subsection text
    subsection_ix = txt_df[txt_df['tag'] == 'subsection']['loc_ix'].values
    try:
        assert all(txt_df.reindex(subsection_ix)['tag'] == 'subsection')
        assert all(txt_df.reindex(subsection_ix+2)['tag'] == 'header')
        txt_df.loc[subsection_ix, 'text'] = txt_df.reindex(subsection_ix+2)['text'].values
        drop_list = np.append(subsection_ix + 1, subsection_ix + 2)
        txt_df = txt_df.drop(drop_list, errors='ignore')
    except: 
        diff = 1
        while subsection_ix.size !=0:
            inds = txt_df.reindex(subsection_ix+diff).dropna(subset=['loc_ix']).index.values
            if inds.size !=0:
                txt_df.loc[inds - diff, 'text'] = txt_df.reindex(inds)['text'].values
                txt_df = txt_df.drop(inds)
                rm_ix = txt_df.reindex(inds-diff)['loc_ix'].values
                subsection_ix = np.array(list(filter(lambda x: x not in rm_ix, subsection_ix)))
            diff += 1

    # Concat text between ranked tags
    ranked_tags = txt_df.dropna(subset=['tag_rank']).index.values
    ranked_tags = np.append(ranked_tags, max(txt_df.index)+1)
    for ix in range(len(ranked_tags)-1):
        txt_df.loc[ranked_tags[ix], 'text'] = txt_df.reindex(range(ranked_tags[ix],ranked_tags[ix+1]))['text'].str.cat(sep=' ')
        txt_df = txt_df.drop(np.arange(ranked_tags[ix]+1,ranked_tags[ix+1]), errors='ignore')

    # Remove short title if first section
    if 'short title' in txt_df.iloc[0]['text'].lower():
        txt_df = txt_df.drop(txt_df.iloc[0]['loc_ix'])

    return txt_df


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


def _remove_punct(sentences):
    # remove punctuations and special characters
    regex = re.compile(r"[^a-zA-Z0-9]")
    return [regex.sub(" ", s) for s in sentences]


def _make_lowercase(sentences):
    return [s.lower() for s in sentences]


def _general_text_cleaning(text):

    text = re.sub("\'s", "", text)
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)

    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)

    # remove comma between numbers    
    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)

    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)
    
    # for hyphenated
    text = re.sub("[a-zA-Z0-9\-]*-[a-zA-Z0-9\-]*", "".join(text.split("-")) , text)
    
    # the single 's' in this stage is 99% of not clean text, just kill it
    text = re.sub(' s ', " ", text)
    
    # reduce extra spaces into single spaces
    text = re.sub('[\s]+', " ", text)
    
    return text


def _full_text_tostring(txt_extract, omit_tags=['header', 'short-title']):
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


def bill_from_xml(xml_string):
    
    xml_string = _remove_whitespace([xml_string])[0]

    match = re.search(r'<external-xref(.*?)>', xml_string)
    while match:
        start, end = (match.start(), match.end())
        xml_string = xml_string.replace(xml_string[start:end], "")
        match = re.search(r'<external-xref(.*?)>', xml_string)
    
    xml_string = xml_string.replace("</external-xref>", "")
    
    split_xml = xml_string.split("<legis-body")
    xml_root = split_xml[0] + "<legis-body"
    xml_string = split_xml[-1]
    
    # Close text tag before external-xref and term to avoid loss of information
    #xml_string = xml_string.replace("<external-xref", "</text><external-xref")
    #xml_string = xml_string.replace("</external-xref>", "</external-xref><text>")
    #xml_string = xml_string.replace("<term>", "</text><term>")
    #xml_string = xml_string.replace("</term>", "</term><text>")

    # Closing text tag didn't work because they weren't always embedded in <text>
    # Need to go back to this when I understand XML trees better. 
    xml_string = xml_string.replace("<quote>", "")
    xml_string = xml_string.replace("</quote>", "")
    xml_string = xml_string.replace("<term>", "")
    xml_string = xml_string.replace("</term>", "")
    
    xml_string = xml_root + xml_string
    txt_tree = ET.ElementTree(ET.fromstring(xml_string))
    txt_root = txt_tree.getroot()

    txt_extract = [[ix, elem.tag, elem.text] for ix, elem
                   in enumerate(txt_root.iter())]
    
    return txt_extract


def _remove_whitespace(sentence_list):
    white_space = list(string.whitespace)[1:]
    for ix in range(len(sentence_list)):
        for bad_string in white_space:
            if bad_string in sentence_list[ix]:
                sentence_list[ix] = sentence_list[ix].replace(bad_string, "")
    return sentence_list


def _remove_short_words(sentence):
    # Only get words with at least min_chars characters
    sentence = " ".join(re.compile(r"\b\w\w+\b", re.U).findall(sentence))
    return sentence


def _extract_entities(sentences, nlp, bad_ents=['WORK_OF_ART', 'CARDINAL']):
    df_ent = pd.DataFrame()
    for ix, sentence in enumerate(sentences):
        doc = nlp(sentence)
        for ent in doc.ents:

            df_ent = df_ent.append([[ix, ent.text, ent.start_char,
                                     ent.end_char, ent.label_]],
                                   ignore_index=True)

    df_ent.columns = ['sentence', 'text', 'start_char', 'end_char', 'label']
    df_ent = df_ent[(~df_ent.label.isin(bad_ents)) &
                    (~df_ent['text'].apply(str.isspace))]
    return df_ent


def _extract_embeddings(
        path_to_embedding='../nlp_models/glove.6B/glove.6B.300d.txt'):
    f = open(path_to_embedding, encoding='utf-8')
    word_embeddings = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    embedding_size = list(coefs.shape)[0]
    return word_embeddings, embedding_size


def _return_embeddings(full_sent):
    df_ent = _extract_entities(full_sent)
    return df_ent


def _calc_embedding(sen, word_embeddings, embedding_size, not_leglove):
    if embedding_size is None:
        embedding_size = random.choice(list(word_embeddings.values())).shape
    if len(sen) != 0:
        if not_leglove:
            vector = sum([word_embeddings.get(w, np.zeros(embedding_size))
                        for w in sen.split()])/(len(sen.split())+0.001)
        else:
            sen_emb = []
            for w in sen.split():
                try: 
                    e = word_embeddings['word_vectors'][word_embeddings['dictionary'][w]]
                except: 
                    e = np.zeros((100,)).shape
                sen_emb.append(e)
                vector = sum(sen_emb)/(len(sen.split())+0.001)   

    else:
        vector = np.zeros(embedding_size)
    return vector


def _remove_custom(sentence_list, type='sec'):
    for ix in range(len(sentence_list)):
        s = sentence_list[ix]
        start_ix = s.find("(Sec.")
        if start_ix != -1:
            end_ix = start_ix + 8
            sentence_list[ix] = sentence_list[ix].replace(s[start_ix:end_ix],
                                                          " ")
    return sentence_list


def _get_full_text_vectors(full_sent_clean, word_embeddings, embedding_size, not_leglove):
    full_vec = [_calc_embedding(s, word_embeddings, embedding_size, not_leglove)
                for s in full_sent_clean]
    return full_vec


def _get_summary_text_vectors(summ_sent_clean, word_embeddings,
                              embedding_size, not_leglove):
    summ_vec = [_calc_embedding(s, word_embeddings, embedding_size, not_leglove)
                for s in summ_sent_clean]
    return summ_vec


if __name__ == "__main__":
    nlp = spacy.load('en_core_web_lg')

    # Connect to db wiht sqlalchemy
    dbname = 'congressional_bills'
    username = 'melissaferrari'
    engine = sqlalchemy.create_engine('postgres://%s@localhost/%s' %
                                      (username, dbname))
    print(engine.url)

    # Connect to make queries using psycopg2
    con = None
    con = psycopg2.connect(database=dbname, user=username)

    get_data = retrieve_data(engine)
    summary_table, bills_text_table, bill_inner_join, ed_bills = get_data

import numpy as np

import bill_utils
import pandas as pd
import pickle


def feature_generators(txt_df, feature_dict=None, joint_title=None):

    txt_df['clean_text'] = txt_df['clean_text'].fillna("")

    assert 'clean_text' in txt_df.columns
    print(txt_df.columns)
    if not feature_dict:
        feature_dict = {
            'title_word_count': lambda x: len([wrd for wrd in x.split()
                                               if wrd in joint_title]),
            'char_count': len,
            'word_count': lambda x: len(x.split()),
            }
        feature_list = list(feature_dict.keys())

    # for feature_name in feature_dict.keys():
    #     txt_df[feature_name] = txt_df['clean_text'].apply(
    #         feature_dict[feature_name])

    txt_df['title_word_count'] = txt_df['clean_text'].apply(lambda x: len([wrd for wrd in x.split()
                                               if wrd in joint_title]))

    txt_df['char_count'] = txt_df['clean_text'].apply(len)

    txt_df['word_count'] = txt_df['clean_text'].apply( lambda x: len(x.split()))
    txt_df['word_density'] = np.divide(txt_df['char_count'], txt_df.word_count + 1)

    feature_list.append('word_density')
    # if all(x in txt_df.columns for x in ['char_count', 'word_count']):
    #     txt_df['word_density'] = txt_df.apply(
    #         lambda x: np.divide(x.char_count, x.word_count + 1))
    #     feature_list.append('word_density')

    return txt_df, feature_list


def prepare_features(bill):

    official_title = bill['official_title'].lower()
    short_title = bill['short_title'].lower()
    joint_title = official_title + short_title

    X = bill_utils.generate_bill_data(bill,
                                      train=False)

    X, feature_list = feature_generators(X,
                                         joint_title=joint_title)
    print(joint_title)
    # feature_df_cols = ['clean_text', 'tag_rank', 'abs_loc', 'norm_loc']
    # feature_df_cols.extend(feature_list)
    # X = full_txt[feature_df_cols]

    return X

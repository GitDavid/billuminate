import sys
if sys.platform == "linux":
    sys.path.append('/home/ubuntu/repo/billuminate/src/')
    sys.path.append('/media/swimmers3/ferrari_06/repo/billuminate/src/')

elif sys.platform == "darwin":
    sys.path.append('/Users/melissaferrari/Projects/repo/billuminate/src/')

MODEL_ROOT = '../../models/'
NLP_MODEL_ROOT = '../../nlp_models/'
TRAINING_DATA_ROOT = '../../data/training_data/'

import os
import time

import numpy as np

import pandas as pd
import sqlalchemy
import datetime
import pdb

from data_preparation import feature_utils, bill_utils
import tqdm


def aggregate_feature_data(features_df, train_df, bills_info,
                           word_embeddings, embedding_size):

    all_X = pd.DataFrame()
    all_y = pd.DataFrame()

    unique_bills = features_df.bill_id.unique()
    train_df = train_df[train_df.bill_id.isin(unique_bills)].copy()

    for bill_id in tqdm.tqdm(unique_bills):
        try:
            feature_df = features_df[(
                features_df['bill_id'] == bill_id)].copy()
            
            bill = bills_info[bills_info['bill_id'] == bill_id].copy()
            bill = bill_utils._return_correct_version(bill, as_dict=True)

            official_title = bill['official_title'].lower()
            short_title = bill['short_title'].lower()
            joint_title = official_title + short_title
            #print(bill.keys())
            #get_data = bill_utils.generate_bill_data(bill, word_embeddings=None,
             #                                        embedding_size=None,
              #                                       train=False, get_vecs=False)
            #full_txt = get_data
            #pdb.set_trace()
            feat_data = feature_utils.feature_generators(
                feature_df, joint_title=joint_title)

            feature_df, feature_list = feat_data
            #pdb.set_trace()
            embeds_df = train_df[train_df.bill_id == bill_id].copy()
            #y = embeds_df[['bill_id', 'in_summary']]
            y = embeds_df[['bill_id', 'in_summary', 'mean_importance']]
            
            feature_df_cols = ['tag_rank', 'abs_loc', 'norm_loc']
            feature_df_cols.extend(feature_list)
            #pdb.set_trace()
            feature_df = feature_df[feature_df_cols]

            feature_df = feature_df.reset_index(drop=True).merge(
                y.reset_index(drop=True), left_index=True, right_index=True)

            X = feature_df.drop(columns=['in_summary'])

            X = X.set_index('bill_id')
            y = y.set_index('bill_id')

            all_X = all_X.append(X)
            all_y = all_y.append(y)
        except TypeError:
            print('{} failed'.format(bill_id))

    return all_X, all_y


def main():

    date = datetime.datetime.today().strftime('%Y%m%d')

    # Connect to database
    dbname = 'congressional_bills'
    username = 'postgres'
    password = 'password'
    engine = sqlalchemy.create_engine('postgres://%s:%s@localhost/%s' %
                                      (username, password, dbname))
    print(engine.url)

    subject = 'Health'

    all_files = np.sort(os.listdir(TRAINING_DATA_ROOT))
    print(all_files)
#     trainfiles = [f for f in all_files if 'training_labeled' in f]
#     textfiles = [f for f in all_files if 'structuredtext' in f]
#     trainfiles = [f for f in trainfiles if subject in f]
#     textfiles = [f for f in textfiles if subject in f]
#     trainfiles = [f for f in trainfiles if 'leglemno' in f]
#     textfiles = [f for f in textfiles if 'leglemno' in f]    

    trainfiles = ['20190217_training_labeled_glove200_Health.csv']
    textfiles = [ '20190217_structuredtext_glove200_Health.csv']
    assert len(trainfiles) > 0
    print(trainfiles)
    print(textfiles)

    # Get data files
    train_df = pd.DataFrame()
    for file_name in np.sort(trainfiles):
        training_data = pd.read_csv(TRAINING_DATA_ROOT + file_name)
        del training_data['Unnamed: 0']
        train_df = train_df.append(training_data)

    save_name = file_name.split('.csv')[0]
    
    features_df = pd.DataFrame()
    for file_name in np.sort(textfiles):
        features = pd.read_csv(TRAINING_DATA_ROOT + file_name)
        del features['Unnamed: 0']
        features_df = features_df.append(features)

    # Organize feature space
    bills_info = pd.read_sql_table('bills', con=engine)
    df_X, df_y = aggregate_feature_data(features_df, train_df, bills_info, 
                                       word_embeddings=None,
                                                     embedding_size=None)

    # Save features
    save_name_X = '{}_features_X.csv'.format(save_name)
    save_path_X = os.path.join(TRAINING_DATA_ROOT, save_name_X)
    df_X.to_csv(save_path_X)

    save_name_y = '{}_features_y.csv'.format(save_name)
    save_path_y = os.path.join(TRAINING_DATA_ROOT, save_name_y)
    df_y.to_csv(save_path_y)


if __name__ == "__main__":
    start_time = time.time()
    main()
    seconds = time.time() - start_time
    minutes = int(np.divide(seconds, 60))
    print("--- Creating the feature space took {} minutes ---".format(minutes))

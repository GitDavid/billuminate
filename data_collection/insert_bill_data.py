import psycopg2
import pandas as pd
import os
import numpy
import ast


def import_csv(file_path):
    df = pd.read_csv(file_path)
    del df['Unnamed: 0']
    return df


def _SQLconstruct_bill_general_info():

    # Construct SQL query data relatations
    col_names = ['bill_id', 'by_request', 'congress', 'introduced_at',
                 'number', 'official_title', 'popular_title', 'short_title',
                 'status_at', 'subjects_top_term', 'updated_at', 'url']
    cols_string = str(col_names)[1:-1]
    cols_string = cols_string.replace("'", '')

    vals_type = '%s, ' * len(col_names)
    vals_type = vals_type[:-2]

    relational_cols = ['status', 'bioguide_id', 'bill_type']
    related_cols_str = str(relational_cols)[1:-1]
    related_cols_str = related_cols_str.replace("'", '')

    string_mapping = {"bioguide_id": "sponsor"}
    for key in string_mapping.keys():
        related_cols_str = related_cols_str.replace(key,
                                                    string_mapping[key])

    query_keys = related_cols_str.split(', ')

    index = []
    column_name = []
    for ix, val in enumerate(query_keys):
        index.append(ix)
        column_name.append(val)
    table_relate = pd.DataFrame({'column': column_name}, index=index)
    table_relate['relation_column'] = ['name', 'bioguide_id', 'name']
    table_relate['relation_table'] = ['lk_status',
                                      'legislators',
                                      'lk_bill_type']

    query_list = ''
    for ix, row in table_relate.iterrows():
        query_list += '(SELECT ls.id FROM {} ls WHERE {}=%s),'.format(
                        row['relation_table'], row['relation_column'])
    query_list = query_list[:-1]

    sql = "INSERT INTO bills ({}, {}) VALUES ({}, {})".format(cols_string,
                                                              related_cols_str,
                                                              vals_type,
                                                              query_list)

    return sql, col_names, relational_cols


def _SQLconstruct_bill_summaries():

    # Construct SQL query data relatations
    col_names = ['as', 'date', 'text']
    cols_string = str(col_names)[1:-1]
    cols_string = cols_string.replace("'", '')

    vals_type = '%s, ' * len(col_names)
    vals_type = vals_type[:-2]

    relational_cols = ['bill_id', 'source']
    related_cols_str = str(relational_cols)[1:-1]
    related_cols_str = related_cols_str.replace("'", '')

    string_mapping = {"bill_id": "bill_ix"}
    for key in string_mapping.keys():
        related_cols_str = related_cols_str.replace(key,
                                                    string_mapping[key])

    query_keys = related_cols_str.split(', ')

    index = []
    column_name = []
    for ix, val in enumerate(query_keys):
        index.append(ix)
        column_name.append(val)
    table_relate = pd.DataFrame({'column': column_name}, index=index)
    table_relate['relation_column'] = ['bill_id', 'name']
    table_relate['relation_table'] = ['bills',
                                      'lk_summary_source']

    query_list = ''
    for ix, row in table_relate.iterrows():
        query_list += '(SELECT ls.id FROM {} ls WHERE {}=%s),'.format(
                        row['relation_table'], row['relation_column'])
    query_list = query_list[:-1]

    sql = "INSERT INTO summaries ({}, {}) VALUES ({}, {})".format(
                                                            cols_string,
                                                            related_cols_str,
                                                            vals_type,
                                                            query_list)
    sql = sql.replace('(as', '("as"')

    return sql, col_names, relational_cols


def bill_general_info(df):

    sql, col_names, relational_cols = _SQLconstruct_bill_general_info()

    # Get bill sponsor info
    sponsor_dict = df['sponsor'].apply(ast.literal_eval).values.tolist()
    sponsor_df = pd.DataFrame(sponsor_dict,
                              index=df.index)[['bioguide_id']]
    df = df.merge(sponsor_df, left_index=True, right_index=True)
    del df['sponsor']

    # Clean dataframe
    df = df[col_names + relational_cols]

    return df, sql


def bill_summaries(df):

    sql, col_names, relational_cols = _SQLconstruct_bill_summaries()

    df_indices = df.index
    df = df[['summary', 'bill_id']].copy()
    df.loc[df['summary'].isna(),
           'summary'] = "{'as': 'nan'}"

    to_dict = ast.literal_eval
    summary_dict = df['summary'].apply(to_dict).values.tolist()

    summary_df = pd.DataFrame(summary_dict, index=df_indices)
    df = df.merge(summary_df,
                  left_index=True,
                  right_index=True)
    del df['summary']
    df['source'] = 'CRS'

    df['date'] = pd.to_datetime(df['date']).dt.date

    # Clean dataframe
    df = df[col_names + relational_cols]

    return df, sql


def send_to_database(df_method, df_path,
                     hostname='localhost',
                     username='melissaferrari',
                     dbname='congressional_bills'):

    # connect to the PostgreSQL database
    db_conn = psycopg2.connect("host={} dbname={} user={}".format(hostname,
                                                                  dbname,
                                                                  username))
    # create a new cursor
    db_cursor = db_conn.cursor()

    # get data to insert
    df = import_csv(df_path)
    df, sql_query = df_method(df)

    for ix, row in df.iterrows():
        insert_list = list(row)

        for ix in range(len(insert_list)):
            if type(insert_list[ix]) == numpy.int64:
                insert_list[ix] = int(insert_list[ix])
            if type(insert_list[ix]) == numpy.bool_:
                insert_list[ix] = bool(insert_list[ix])

        db_cursor.execute(sql_query, insert_list)

    # commit the changes to the database
    db_conn.commit()

    # close communication with the database
    db_cursor.close()
    db_conn.close()


if __name__ == '__main__':

    # Database info
    hostname = 'localhost'
    dbname = 'congressional_bills'
    username = 'melissaferrari'

    # Datapath
    data_path = '/Users/melissaferrari/Projects/repo/bill-summarization/'
    data_path += 'data_files/bill_details'
    file_name = 'agg_propublica_113hr.csv'
    df_path = os.path.join(data_path, file_name)

    df_methods = [bill_general_info, bill_summaries]
    df_method = df_methods[1]

    print('Applying {} to {}'.format(df_method.__name__,
                                     df_path))
    send_to_database(df_method, df_path)

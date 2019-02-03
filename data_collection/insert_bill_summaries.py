import psycopg2
import pandas as pd
import os
import numpy
import ast


def import_csv(file_path):
    df = pd.read_csv(file_path)
    del df['Unnamed: 0']
    return df


if __name__ == '__main__':

    col_names = ['as', 'date', 'text']

    relational_cols = ['bill_id', 'source']

    data_path = '/Users/melissaferrari/Projects/repo/insight/data_collection'
    file_name = 'agg_propublica_114hr.csv'
    data_agg = import_csv(os.path.join(data_path, file_name))

    data_summary = data_agg[['summary', 'bill_id']].copy()
    data_summary.loc[data_summary['summary'].isna(),
                     'summary'] = "{'as': 'nan'}"

    # data_summary = data_summary.dropna()
    # data_summary.loc[data_summary['summary'].isna(),
    #                'summary'] = "{'as': 'nan'}"

    to_dict = ast.literal_eval
    summary_dict = data_summary['summary'].apply(to_dict).values.tolist()

    summary_df = pd.DataFrame(summary_dict, index=data_agg.index)
    data_summary = data_summary.merge(summary_df,
                                      left_index=True,
                                      right_index=True)
    del data_summary['summary']
    data_summary['source'] = 'CRS'

    data_summary['date'] = pd.to_datetime(data_summary['date']).dt.date

    cols_string = str(col_names)[1:-1]
    cols_string = cols_string.replace("'", '')

    vals_type = '%s, ' * len(col_names)
    vals_type = vals_type[:-2]

    related_cols_str = str(relational_cols)[1:-1]
    related_cols_str = related_cols_str.replace("'", '')

    string_mapping = {"bill_id": "bill"}

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

    data_summary = data_summary[col_names + relational_cols]

    print(sql)

    print(data_summary.columns)
    hostname = 'localhost'
    dbname = 'congressional_bills'
    username = 'melissaferrari'

    # connect to the PostgreSQL database
    db_conn = psycopg2.connect("host={} dbname={} user={}".format(hostname,
                                                                  dbname,
                                                                  username))
    # create a new cursor
    db_cursor = db_conn.cursor()
    data_summary = data_summary.dropna()
    for ix, row in data_summary.iterrows():
        insert_list = list(row)
        # print(insert_list)
        for ix in range(len(insert_list)):
            if type(insert_list[ix]) == numpy.int64:
                insert_list[ix] = int(insert_list[ix])
            if type(insert_list[ix]) == numpy.bool_:
                insert_list[ix] = bool(insert_list[ix])

        db_cursor.execute(sql, insert_list)

    # commit the changes to the database
    db_conn.commit()

    # close communication with the database
    db_cursor.close()

    db_conn.close()

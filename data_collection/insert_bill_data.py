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

    col_names = ['bill_id', 'by_request', 'congress', 'introduced_at',
                 'number', 'official_title', 'popular_title', 'short_title',
                 'status_at', 'subjects_top_term', 'updated_at', 'url']

    relational_cols = ['status', 'bioguide_id', 'bill_type']

    data_path = '/Users/melissaferrari/Projects/repo/insight/data_collection'
    file_name = 'agg_propublica_114hr.csv'
    data_agg = import_csv(os.path.join(data_path, file_name))

    sponsor_dict = data_agg['sponsor'].apply(ast.literal_eval).values.tolist()
    sponsor_df = pd.DataFrame(sponsor_dict,
                              index=data_agg.index)[['bioguide_id']]
    data_agg = data_agg.merge(sponsor_df, left_index=True, right_index=True)
    del data_agg['sponsor']

    data_agg = data_agg[col_names + relational_cols]

    cols_string = str(col_names)[1:-1]
    cols_string = cols_string.replace("'", '')

    vals_type = '%s, ' * len(col_names)
    vals_type = vals_type[:-2]

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

    hostname = 'localhost'
    dbname = 'congressional_bills'
    username = 'melissaferrari'

    # connect to the PostgreSQL database
    db_conn = psycopg2.connect("host={} dbname={} user={}".format(hostname,
                                                                  dbname,
                                                                  username))
    # create a new cursor
    db_cursor = db_conn.cursor()

    for ix, row in data_agg.iterrows():
        insert_list = list(row)

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

import psycopg2
import pandas as pd
import os
import numpy
import ast
import re
from lxml import etree
import tqdm


def import_csv(file_path):
    df = pd.read_csv(file_path)
    del df['Unnamed: 0']
    return df


def _choose_bill(folder_contents):
    return [k for k in folder_contents if (k.startswith('BILLS') &
                                           k.endswith('xml'))][0]


def _get_xml_paths(bill_path_root):

    path_list = list(os.walk(bill_path_root))
    file_paths = [os.path.join(x[0], _choose_bill(x[2])) for x in path_list
                  if any(f.endswith('xml') for f in x[2])]
    print('There are {} files to parse through'.format(len(file_paths)))
    return file_paths


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


def _SQLconstruct_bill_text():
    sql = """
          INSERT INTO bill_text (bill_ix, text) VALUES ((SELECT ls.id FROM
          bills ls WHERE bill_id=%s), %s);
          """
    # print('The SQL query passed to psycopg2 /n{}'.format(sql))
    return sql


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


def bill_text(data_path):

    sql = _SQLconstruct_bill_text()

    xml_paths = _get_xml_paths(data_path)
    for xml_path in xml_paths:
        regex_pattern_00 = re.compile(r"\d\d\d([a-z]+)\d+")
        match_00 = regex_pattern_00.search(xml_path).group()

        regex_pattern_01 = re.compile(r"[a-z]+")
        bill_type = regex_pattern_01.search(match_00).group()
        congress, number = match_00.split(bill_type)
        bill_id = '{}{}-{}'.format(bill_type, number, congress)

        tree = etree.parse(xml_path, parser=etree.XMLParser(recover=True))
        string_tree = etree.tostring(tree).decode()

        data = (bill_id, string_tree,)
        yield data, sql


def send_to_database(db_method, data_path,
                     hostname='localhost',
                     username='melissaferrari',
                     dbname='congressional_bills',
                     from_dataframe=True):

    # connect to the PostgreSQL database
    db_conn = psycopg2.connect("host={} dbname={} user={}".format(hostname,
                                                                  dbname,
                                                                  username))
    # create a new cursor
    db_cursor = db_conn.cursor()

    if from_dataframe:
        # get data to insert
        df = import_csv(data_path)
        df, sql_query = db_method(df)

        for ix, row in tqdm.tqdm(df.iterrows()):
            insert_list = list(row)

            for ix in range(len(insert_list)):
                if type(insert_list[ix]) == numpy.int64:
                    insert_list[ix] = int(insert_list[ix])
                if type(insert_list[ix]) == numpy.bool_:
                    insert_list[ix] = bool(insert_list[ix])

            db_cursor.execute(sql_query, insert_list)

    else:
        for data, sql_query in tqdm.tqdm(bill_text(data_path)):
            db_cursor.execute(sql_query, data)

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
    # data_path = '/Users/melissaferrari/Projects/repo/bill-summarization/'
    # data_path += 'data_files/bill_details'
    # file_name = 'agg_propublica_113hr.csv'
    data_path = '/Users/melissaferrari/Projects/repo/congress/data/'
    file_name = '114'
    data_path = os.path.join(data_path, file_name)

    db_methods = [bill_general_info, bill_summaries, bill_text]
    db_method = db_methods[2]
    print('Applying {} to {}'.format(db_method.__name__,
                                     data_path))

    if db_method in [bill_general_info, bill_summaries]:
        from_dataframe = True
    elif db_method in [bill_text]:
        from_dataframe = False

    send_to_database(db_method, data_path,
                     hostname=hostname,
                     username=username,
                     dbname=dbname,
                     from_dataframe=from_dataframe)

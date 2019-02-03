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


def _SQLconstruct_bill_subects():

    sql = _SQLconstruct_bill_text()
    sql = sql.replace("bill_text", "subjects")
    sql = sql.replace("text)", "subject)")

    return sql


def _SQLconstruct_bill_cosponsors():

    # Construct SQL query data relatations
    col_names = ['original_cosponsor']
    cols_string = str(col_names)[1:-1]
    cols_string = cols_string.replace("'", '')

    vals_type = '%s, ' * len(col_names)
    vals_type = vals_type[:-2]

    relational_cols = ['bill_id', 'bioguide_id']
    related_cols_str = str(relational_cols)[1:-1]
    related_cols_str = related_cols_str.replace("'", '')

    string_mapping = {"bill_id": "bill_ix",
                      "bioguide_id": "legislator_ix"}
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
    table_relate['relation_column'] = ['bill_id', 'bioguide_id']
    table_relate['relation_table'] = ['bills', 'legislators']

    query_list = ''
    for ix, row in table_relate.iterrows():
        query_list += '(SELECT ls.id FROM {} ls WHERE {}=%s),'.format(
                        row['relation_table'], row['relation_column'])
    query_list = query_list[:-1]

    table_name = 'cosponsorship'
    sql = "INSERT INTO {} ({}, {}) VALUES ({}, {})".format(table_name,
                                                           cols_string,
                                                           related_cols_str,
                                                           vals_type,
                                                           query_list)

    return sql, col_names, relational_cols


def _SQLconstruct_bill_related():

    # Construct SQL query data relatations
    col_names = ['identified_by', 'reason']
    cols_string = str(col_names)[1:-1]
    cols_string = cols_string.replace("'", '')

    vals_type = '%s, ' * len(col_names)
    vals_type = vals_type[:-2]

    relational_cols = ['bill_id', 'related_bill_id']
    related_cols_str = str(relational_cols)[1:-1]
    related_cols_str = related_cols_str.replace("'", '')

    string_mapping = {"bill_id": "bill_ix",
                      "related_bill_id": "related_bill_ix"}
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
    table_relate['relation_column'] = ['bill_id', 'bill_id']
    table_relate['relation_table'] = ['bills', 'bills']

    query_list = ''
    for ix, row in table_relate.iterrows():
        query_list += '(SELECT ls.id FROM {} ls WHERE {}=%s),'.format(
                        row['relation_table'], row['relation_column'])
    query_list = query_list[:-1]

    table_name = 'related_bills'
    sql = "INSERT INTO {} ({}, {}) VALUES ({}, {})".format(table_name,
                                                           cols_string,
                                                           related_cols_str,
                                                           vals_type,
                                                           query_list)
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


def bill_subjects(df):

    sql = _SQLconstruct_bill_subects()

    # Get bill subject info
    df = df[['subjects', 'bill_id']].copy()
    df['subjects'] = df['subjects'].apply(ast.literal_eval)

    df = df.set_index(['bill_id'])['subjects'].apply(pd.Series).stack()
    df = df.reset_index()
    del df['level_1']
    df.columns = ['bill_id', 'subject']

    return df, sql


def bill_cosponsors(df):

    sql, col_names, relational_cols = _SQLconstruct_bill_cosponsors()

    # Get bill sponsor info
    df = df[['cosponsors', 'bill_id']].copy()
    df['cosponsors'] = df['cosponsors'].apply(ast.literal_eval)
    df = df.set_index(['bill_id'])['cosponsors'].apply(pd.Series).stack()
    df = df.reset_index()
    del df['level_1']
    df.columns = ['bill_id', 'cosponsor']

    df_indices = df.index

    cosponsor_dict = df['cosponsor'].values.tolist()
    cosponsor_df = pd.DataFrame(cosponsor_dict,
                                index=df_indices)[['bioguide_id',
                                                   'original_cosponsor']]

    df = df.merge(cosponsor_df, left_index=True, right_index=True)
    del df['cosponsor']

    # Clean dataframe
    df = df[col_names + relational_cols]

    return df, sql


def bill_related(df):

    sql, col_names, relational_cols = _SQLconstruct_bill_related()

    # Get related bill info
    df = df[['related_bills', 'bill_id']].copy()
    df['related_bills'] = df['related_bills'].apply(ast.literal_eval)
    df = df.set_index(['bill_id'])['related_bills'].apply(pd.Series).stack()
    df = df.reset_index()
    del df['level_1']
    df.columns = ['bill_id', 'related_bill']

    df_indices = df.index

    related_bills_dict = df['related_bill'].values.tolist()
    related_bills_df = pd.DataFrame(related_bills_dict, index=df_indices)
    del related_bills_df['type']
    related_bills_df.columns = ['related_bill_id', 'identified_by', 'reason']

    df = df.merge(related_bills_df, left_index=True, right_index=True)
    del df['related_bill']

    # Make sure we are only looking at related bills in database
    df['congress'] = df['related_bill_id'].str.split('-').str[1]
    df['congress'] = df['congress'].astype(int)
    df = df[df['congress'].isin([113, 114, 115])]
    del df['congress']

    # Clean dataframe
    df = df[col_names + relational_cols]

    return df, sql


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

        print('There are {} rows to parse'.format(len(df)))
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

    # Select database insert type
    db_methods = {0: bill_general_info,
                  1: bill_summaries,
                  2: bill_text,
                  3: bill_subjects,
                  4: bill_cosponsors,
                  5: bill_related}
    db_method = db_methods[5]

    if db_method in [bill_text]:
        from_dataframe = False
    else:
        from_dataframe = True

    # Datapath
    raw_path = '/Users/melissaferrari/Projects/repo/bill-summarization/'
    raw_path += 'data_files/bill_details'
    """
    raw_path = '/Users/melissaferrari/Projects/repo/congress/data/'
    file_name = '114'
    """

    file_names = ['agg_propublica_113hr.csv', 'agg_propublica_113s.csv',
                  'agg_propublica_114hr.csv', 'agg_propublica_114s.csv',
                  'agg_propublica_115hr.csv', 'agg_propublica_115s.csv']
    for file_name in file_names:
        data_path = os.path.join(raw_path, file_name)

        print('Applying {} to {}'.format(db_method.__name__,
                                         data_path.split('/')[-1]))

        send_to_database(db_method, data_path,
                         hostname=hostname,
                         username=username,
                         dbname=dbname,
                         from_dataframe=from_dataframe)

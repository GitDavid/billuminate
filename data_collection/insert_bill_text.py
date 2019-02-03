import psycopg2
import pandas as pd
import os
import re
from lxml import etree


def import_csv(file_path):
    df = pd.read_csv(file_path)
    del df['Unnamed: 0']
    return df


if __name__ == '__main__':

    hostname = 'localhost'
    dbname = 'congressional_bills'
    username = 'melissaferrari'

    # connect to the PostgreSQL database
    db_conn = psycopg2.connect("host={} dbname={} user={}".format(hostname,
                                                                  dbname,
                                                                  username))
    # create a new cursor
    db_cursor = db_conn.cursor()

    bill_path_root = '/Users/melissaferrari/Projects/repo/congress/data/'
    xml_paths = [x[0] for x in os.walk(bill_path_root) if 'xml' in x[0]]

    for xml_path in xml_paths:
        regex_pattern_00 = re.compile(r"\d\d\d([a-z]+)\d+")
        match_00 = regex_pattern_00.search(xml_path).group()

        regex_pattern_01 = re.compile('[a-z]+')
        bill_type = regex_pattern_01.search(match_00).group()
        congress, number = match_00.split(bill_type)
        bill_id = '{}{}-{}'.format(bill_type, number, congress)

        xml_file = [f for f in os.listdir(xml_path) if f.endswith('xml')][0]
        tree = etree.parse(os.path.join(xml_path, xml_file))
        string_tree = etree.tostring(tree).decode()
        # etree.fromstring(string_tree)

        sql = """
              INSERT INTO bill_text (bill_ix, text) VALUES ((SELECT ls.id FROM
              bills ls WHERE bill_id=%s), %s);
              """
        data = (bill_id, string_tree,)
        db_cursor.execute(sql, data)

    # commit the changes to the database
    db_conn.commit()

    # close communication with the database
    db_cursor.close()

    db_conn.close()

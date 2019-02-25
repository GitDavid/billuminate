import sys
sys.path.append('../')

from data_preparation import text_utils, training_utils
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import re
import os


def get_bill(df, bill_id):
    return df[df.bill_id == bill_id].copy()


def select_random_rows(df, n_rows):
    ixs = np.random.choice(df.index.values, n_rows)
    df = df.reindex(ixs)
    return df


def to_csv_append_mode(df, file_path):
    print(file_path)
    if not os.path.isfile(file_path):
        df.to_csv(file_path)
    else:
        df.to_csv(file_path, mode='a', header=False)


def _return_correct_version(df_bills,
                            as_dict=False,
                            code_order=None):

    # To create a 1-to-1 mapping of bill text and summaries
    # by choosing most recent bill text version

    if not code_order:
        code_order = ['ENR', 'EAS', 'EAH', 'RS', 'ES',
                      'PCS', 'EH', 'RH', 'IS', 'IH']

    num_rows = len(df_bills)
    if num_rows == 0:
        raise Exception('Oh no! I cannot find this bill.')
    elif num_rows > 1:
        code = next(i for i in code_order if i in df_bills['code'].unique())
        df_bills = df_bills[df_bills['code'] == code]

    if as_dict:
        return df_bills.iloc[0].to_dict()
    else:
        return df_bills


def retrieve_data(engine, bill_id=None, bill_title=None, subject=None):

    query = """
            SELECT
            bi.bill_id,
            sm.text AS summary_text,
            bt.text AS full_text,

            bi.subjects_top_term,
            bi.official_title,
            bi.short_title,

            bv.code,
            sm.as as summary_as,
            sm.date as summary_date,
            sm.bill_ix
            FROM summaries sm

            INNER JOIN bill_text bt
            ON sm.bill_ix=bt.bill_ix

            INNER JOIN bill_versions bv
            ON bv.id=bt.bill_version_id

            INNER JOIN bills bi
            ON sm.bill_ix=bi.id
            ;
            """
    if bill_id:
        bill_id_query = """
                        WHERE bi.bill_id='%s';
                        """
        query = query[:query.find(';')] + bill_id_query % bill_id

    if bill_title:
        bill_title_query = """
                           WHERE bi.official_title='%s';
                           """
        query = query[:query.find(';')] + bill_title_query % bill_title

    if subject:
        print('Queries limited to subject: {}'.format(subject))
        subject_query = """
                        INNER JOIN bills
                        ON bills.id=sm.bill_ix
                        WHERE bills.subjects_top_term='%s';
                        """
        query = query[:query.find(';')] + subject_query % subject

    bill_df = pd.read_sql_query(query, engine)

    if bill_df.empty:
        return bill_df

    # The following code uses two methods to return correct bill.
    bill_df = bill_df.groupby('bill_id', group_keys=False
                              ).apply(lambda x:
                                      _return_correct_version(x))

    group_sizes = bill_df.groupby('bill_id').size()
    dummy_ind = group_sizes[group_sizes > 1].index
    duplicates = bill_df[bill_df.bill_id.isin(dummy_ind)]
    if not duplicates.empty:
        dup = duplicates[['full_text', 'code']].reset_index().values
        dup[:, 1] = list(map(lambda x:
                             x.split('bill-stage="')[1].split('"')[0][0],
                             dup[:, 1]))
        dup[:, 2] = list(map(lambda x: x[0], dup[:, 2]))
        bad_ixs = dup[:, 0][dup[:, 1] != dup[:, 2]]
        bill_df = bill_df[~bill_df.index.isin(bad_ixs)]

    print('{} unique bills being analyzed'.format(bill_df.bill_id.nunique()))
    print('{} rows in the table'.format(len(bill_df)))

    return bill_df


def bill_from_xml(xml_str):

    xml_str = text_utils._remove_whitespace([xml_str])[0]

    # Close text tag before external-xref and term to avoid loss of information
    # xml_str = xml_str.replace("<external-xref", "</text><external-xref")
    # xml_str = xml_str.replace("</external-xref>", "</external-xref><text>")
    # xml_str = xml_str.replace("<term>", "</text><term>")
    # xml_str = xml_str.replace("</term>", "</term><text>")

    # Did not work because they weren't always embedded in <text>
    # Just remove <external-xref> tags
    match = re.search(r'<external-xref(.*?)>', xml_str)
    while match:
        start, end = (match.start(), match.end())
        xml_str = xml_str.replace(xml_str[start:end], "")
        match = re.search(r'<external-xref(.*?)>', xml_str)
    xml_str = xml_str.replace("</external-xref>", "")

    # Body of bill starts after <legis-body> tag
    split_xml = xml_str.split("<legis-body")
    xml_root = split_xml[0] + "<legis-body"
    xml_str = split_xml[-1]

    # Need better processing for nested tags - for now just remove
    xml_str = xml_str.replace("<quote>", "")
    xml_str = xml_str.replace("</quote>", "")
    xml_str = xml_str.replace("<term>", "")
    xml_str = xml_str.replace("</term>", "")

    xml_str = xml_root + xml_str
    txt_tree = ET.ElementTree(ET.fromstring(xml_str))
    txt_root = txt_tree.getroot()

    txt_extract = [[ix, elem.tag, elem.text] for ix, elem
                   in enumerate(txt_root.iter())]

    return txt_extract


def _clean_extracted_list(txt_extract, tag_rankings=None):

    # Generate ranking for tags if not provided
    if not tag_rankings:
        ordered_tags = ['bill', 'title', 'section', 'subsection', 'paragraph',
                        'subparagraph', 'clause', 'subclause', 'item',
                        'subitem', 'subsubitem']
        tag_rankings = {key: value for (value, key) in enumerate(ordered_tags)}

    # Create dataframe with extracted text and corresponding tag, location
    txt_df = pd.DataFrame(txt_extract)
    txt_df.columns = ['loc_ix', 'tag', 'text']
    txt_df['tag_rank'] = txt_df['tag'].map(tag_rankings)

    # Drop pagebreak tag bc it causes errors
    txt_df = txt_df[txt_df.tag != 'pagebreak']

    # Drop header section and titles
    ix_min = txt_df[txt_df['tag'] == 'legis-body'].index.values[0]+1
    txt_df = txt_df.drop(txt_df.index[np.arange(ix_min)])

    # Add enumeration to front of each list item
    num_ixs = txt_df[txt_df['tag'] == 'enum']['loc_ix'].values
    for ix in num_ixs:
        txt_df.loc[ix+1, 'text'] = txt_df.reindex(
            range(ix, ix+2))['text'].str.cat(sep=' ')
        txt_df = txt_df.drop(ix)

    # Drop quoted block section FOR NOW
    txt_df = txt_df[txt_df.tag != 'quoted-block']
    txt_df = txt_df[txt_df.tag != 'after-quoted-block']

    # # Attempt to work with quoted blocks did not deal with edge cases of
    # # entire bills as quoted blocks
    # min_ixs = txt_df[txt_df.tag == 'quoted-block']['loc_ix'].values
    # max_ixs = txt_df[txt_df.tag == 'after-quoted-block']['loc_ix'].values

    # # Catch quote blocks in quote blocks
    # if any(min_ixs[1:] < max_ixs[:-1]):
    #     for ix in range(len(min_ixs)-1):
    #         if min_ixs[ix+1] < max_ixs[ix]:
    #             min_ixs = np.delete(min_ixs, ix+1)
    #             max_ixs = np.delete(max_ixs, ix)

    # for ix_loc in range(len(min_ixs)):
    #     txt_df.loc[min_ixs[ix_loc], 'text'] = txt_df.reindex(
    #         range(min_ixs[ix_loc] + 1,
    #               max_ixs[ix_loc]+1))['text'].str.cat(sep=' ')
    #     txt_df = txt_df.drop(
    #         np.arange(min_ixs[ix_loc] + 1, max_ixs[ix_loc]+1),
    #         errors='ignore')

    # FIND SECTION TITLE (because tag is empty)

    # In most cases, the section title will be two rows below the
    # section tag in the header. In cases where it is not we just
    # find the next tag with text bc that is likely the title.
    section_ix = txt_df[txt_df['tag'] == 'section']['loc_ix'].values

    try:
        assert all(txt_df.reindex(section_ix)['tag'] == 'section')
        assert all(txt_df.reindex(section_ix + 2)['tag'] == 'header')
        txt_df.loc[section_ix, 'text'] = txt_df.reindex(
            section_ix + 2)['text'].values
        drop_list = np.append(section_ix + 1, section_ix + 2)
        txt_df = txt_df.drop(drop_list, errors='ignore')

    except AssertionError:
        diff = 1
        while section_ix.size != 0:
            inds = txt_df.reindex(section_ix + diff).dropna(
                subset=['loc_ix']).index.values
            if inds.size != 0:
                txt_df.loc[inds - diff, 'text'] = txt_df.reindex(
                    inds)['text'].values
                txt_df = txt_df.drop(inds)
                rm_ix = txt_df.reindex(inds-diff)['loc_ix'].values
                section_ix = np.array(
                    list(filter(lambda x: x not in rm_ix, section_ix)))
            diff += 1

    # FIND SUBSECTION TITLE (because tag is empty)
    subsection_ix = txt_df[txt_df['tag'] == 'subsection']['loc_ix'].values
    try:
        assert all(txt_df.reindex(subsection_ix)['tag'] == 'subsection')
        assert all(txt_df.reindex(subsection_ix+2)['tag'] == 'header')
        txt_df.loc[subsection_ix, 'text'] = txt_df.reindex(
            subsection_ix + 2)['text'].values
        drop_list = np.append(subsection_ix + 1, subsection_ix + 2)
        txt_df = txt_df.drop(drop_list, errors='ignore')

    except AssertionError:
        diff = 1
        while subsection_ix.size != 0:
            inds = txt_df.reindex(subsection_ix+diff).dropna(
                subset=['loc_ix']).index.values
            if inds.size != 0:
                txt_df.loc[inds - diff, 'text'] = txt_df.reindex(
                    inds)['text'].values
                txt_df = txt_df.drop(inds)
                rm_ix = txt_df.reindex(inds-diff)['loc_ix'].values
                subsection_ix = np.array(
                    list(filter(lambda x: x not in rm_ix, subsection_ix)))
            diff += 1
#     try:
#         rm_upto_inclusive = txt_df[
#           txt_df['tag'] == 'short-title'].index.values[0]
#         txt_df = txt_df.loc[rm_upto_inclusive+1:]
    # Concat text between ranked tags
    # THIS SHOULD BE DONE DIFFERENTLY.
    # MULTIPLE SENTENCES IN SAME ENUMERATION LEVEL SHOULD NOT BE CONCATENATED

    ranked_tags = txt_df.dropna(subset=['tag_rank']).index.values
    ranked_tags = np.append(ranked_tags, max(txt_df.index) + 1)
    for ix in range(len(ranked_tags) - 1):
        txt_df.loc[ranked_tags[ix], 'text'] = txt_df.reindex(
            range(ranked_tags[ix],
                  ranked_tags[ix + 1]))['text'].str.cat(sep=' ')
        txt_df = txt_df.drop(np.arange(ranked_tags[ix] + 1,
                                       ranked_tags[ix + 1]), errors='ignore')

    # Remove short title if first section
    if 'short title' in txt_df.iloc[0]['text'].lower():
        txt_df = txt_df.drop(txt_df.iloc[0]['loc_ix'])

    return txt_df


def get_clean_text(text_string, text_type, short_title=None, nlp=None):

    assert text_type in ['summary', 'full_text']
    if text_type == 'summary':
        assert all(v is not None for v in [short_title, nlp])

        # Tokenize to sentences
        summ_sents = text_utils.tokenize_summ(text_string, nlp, short_title)

        summ_df = pd.DataFrame(summ_sents)

        # Apply generic text cleaning
        summ_sents = text_utils._apply_text_cleaning(summ_sents)
        summ_sents = text_utils._remove_custom(summ_sents, type='sec')

        return summ_df, summ_sents

    if text_type == 'full_text':
        txt_extract = bill_from_xml(text_string)
        txt_df = _clean_extracted_list(txt_extract)

        full_sents = list(txt_df.replace(np.nan, '',
                                         regex=True)['text'].values)

        # Apply generic text cleaning
        full_sents = text_utils._apply_text_cleaning(full_sents)

        return txt_df, full_sents


def _describe_full_text(full_string, bill_id,
                        word_embeddings=None):

    full_txt, fsents = get_clean_text(full_string,
                                      text_type='full_text')

    full_txt['bill_id'] = bill_id
    full_txt['clean_text'] = fsents

    locs = full_txt['loc_ix']
    full_txt['abs_loc'] = (locs - locs.min()).values
    if locs.max() - locs.min() > 0:
        full_txt['norm_loc'] = (np.divide(locs - locs.min(),
                                          locs.max() - locs.min())).values
    else:
        full_txt['norm_loc'] = (np.divide(locs - locs.min(),
                                          1)).values

    if word_embeddings:
        fvecs = text_utils._calc_embeddings_set(fsents,
                                                word_embeddings)
        return full_txt, fvecs

    else:
        return full_txt


def _describe_summ_text(summ_string, bill_id, short_title,
                        word_embeddings, nlp):
    sum_df, ssents = get_clean_text(summ_string,
                                    text_type='summary',
                                    short_title=short_title,
                                    nlp=nlp)

    sum_df['bill_id'] = bill_id
    sum_df['clean_text'] = ssents
    sum_df = sum_df.rename(columns={0: 'text'})
    svecs = text_utils._calc_embeddings_set(ssents,
                                            word_embeddings)
    return sum_df, svecs


def generate_bill_data(bill, word_embeddings=None,
                       nlp=None, train=False):

    short_title = bill['short_title']
    full_string = bill['full_text']
    bill_id = bill['bill_id']

    if not word_embeddings:
        full_txt = _describe_full_text(full_string, bill_id)
        return full_txt

    else:
        full_txt, fvecs = _describe_full_text(full_string, bill_id,
                                              word_embeddings)

    if train:
        assert nlp
        assert word_embeddings

        summ_string = bill['summary_text']

        sum_df, svecs = _describe_summ_text(summ_string, bill_id,
                                            short_title,
                                            word_embeddings,
                                            nlp)

        get_values = training_utils.label_important(
            fvecs, svecs, word_embeddings['size'], max_sim=0.5)
        label_df, sim_mat_mask, mean_importance = get_values
        summ_data = pd.DataFrame(
            svecs, columns=['embed_{:03}'.format(i)
                            for i in range(word_embeddings['size'])])

        summ_data = summ_data.reset_index()
        summ_data = summ_data.rename(columns={'index': 'loc_ix'})

        summ_data['in_summary'] = 2
        summ_data['bill_id'] = bill_id
        summ_data['mean_importance'] = 1.

        label_df = label_df.reset_index()
        label_df = label_df.rename(columns={'index': 'loc_ix'})
        label_df['bill_id'] = bill_id

        assert(len(label_df[label_df['in_summary'] != 1]) +
               len(label_df[label_df['in_summary'] == 1]) ==
               len(label_df))

        label_df = pd.concat([label_df, summ_data], sort=False).copy()

        return label_df, full_txt, sum_df

    else:
        return full_txt, fvecs

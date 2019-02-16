import sys
sys.path.append('/home/ubuntu/repo/billuminate/src/')

import numpy as np

import pandas as pd
from data_preparation import text_utils


def label_important(full_vec, summ_vec, embedding_size,
                    max_sim=None, set_percentile=None, version=None):

    assert any([max_sim, set_percentile])

    sim_mat = text_utils._create_sim_mat(full_vec, summ_vec,
                                         embedding_size=embedding_size)

    if version == 1:
        ix_match = text_utils._sort_matrix_ix(sim_mat)

    if version == 2:
        ix_match = []
        mat_percent = np.percentile(sim_mat, set_percentile)
        if len(summ_vec) < len(full_vec):
            while len(ix_match) < len(summ_vec):
                sim_mat_mask = np.where(sim_mat >= mat_percent, 1, 0)
                ix_match = np.unique(np.argwhere(sim_mat_mask != 0)[:, 0])
                set_percentile -= 5
        else:
            print('summary length {} > {} bill length'.format(len(summ_vec),
                                                              len(full_vec)))
            _, ix_match = text_utils._sort_matrix_ix(sim_mat, 1)

    else:
        if set_percentile:
            max_sim = np.percentile(sim_mat, set_percentile)
            sim_mat_mask = np.where(sim_mat >= max_sim, 1, 0)
            ix_match = np.unique(np.argwhere(sim_mat_mask != 0)[:, 0])

            if len(ix_match) < len(summ_vec):
                print_str = 'matches found {} < {} summary length'
                print(print_str.format(len(ix_match), len(summ_vec)))
            _, ix_match = text_utils._sort_matrix_ix(sim_mat, 1)

    labeled_df = pd.DataFrame(full_vec, columns=['embed_{:03}'.format(i)
                              for i in range(embedding_size)])

    labeled_df['in_summary'] = 0
    labeled_df.loc[ix_match, 'in_summary'] = 1

    return labeled_df, ix_match

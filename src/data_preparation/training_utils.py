import sys
if sys.platform == "linux":
    sys.path.append('/home/ubuntu/repo/billuminate/src/')
    sys.path.append('/media/swimmers3/ferrari_06/repo/billuminate/src/')
elif sys.platform == "darwin":
    sys.path.append('/Users/melissaferrari/Projects/repo/billuminate/src/')


import numpy as np

import pandas as pd
from data_preparation import text_utils


def label_important(full_vec, summ_vec, embedding_size,
                    max_sim=None, set_percentile=None, version=None):

    assert any([max_sim, set_percentile])
    try:
        sim_mat = text_utils._create_sim_mat(full_vec, summ_vec,
                                             embedding_size=embedding_size)
        if set_percentile:
            max_sim = np.percentile(sim_mat, set_percentile)
        
        mean_importance = np.mean(sim_mat, axis=1)
        
        sim_mat_mask = np.where(mean_importance >= max_sim, 1, 0)
        # sim_mat_mask = np.where(sim_mat >= max_sim, 1, 0)
        ix_match = np.count_nonzero(sim_mat_mask)
        #np.unique(np.argwhere(sim_mat_mask != 0)[:, 0])

        if ix_match < len(summ_vec):
            print_str = 'matches found {} < {} summary length'
            print(print_str.format(ix_match, len(summ_vec)))

        labeled_df = pd.DataFrame(full_vec, columns=['embed_{:03}'.format(i)
                                  for i in range(embedding_size)])

        # labeled_df['in_summary'] = 0
        # labeled_df.loc[ix_match, 'in_summary'] = 1

        labeled_df['in_summary'] = sim_mat_mask
        labeled_df['mean_importance'] = mean_importance
        
        return labeled_df, sim_mat_mask, mean_importance#, ix_match
    except AttributeError:
        print(len(full_vec), len(summ_vec))
        pass
#     if version == 1:
#         ix_match = text_utils._sort_matrix_ix(sim_mat)

#     if version == 2:
#         ix_match = []
#         mat_percent = np.percentile(sim_mat, set_percentile)
#         if len(summ_vec) < len(full_vec):
#             while len(ix_match) < len(summ_vec):
#                 sim_mat_mask = np.where(sim_mat >= mat_percent, 1, 0)
#                 ix_match = np.unique(np.argwhere(sim_mat_mask != 0)[:, 0])
#                 set_percentile -= 5
#         else:
#             print('summary length {} > {} bill length'.format(len(summ_vec),
#                                                               len(full_vec)))
#             _, ix_match = text_utils._sort_matrix_ix(sim_mat, 1)

#     else:


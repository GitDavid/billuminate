import numpy as np
import pandas as pd

import sys
sys.path.append('../')
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
        ix_match = np.count_nonzero(sim_mat_mask)

        if ix_match < len(summ_vec):
            print_str = 'matches found {} < {} summary length'
            print(print_str.format(ix_match, len(summ_vec)))

        labeled_df = pd.DataFrame(full_vec, columns=['embed_{:03}'.format(i)
                                  for i in range(embedding_size)])

        labeled_df['in_summary'] = sim_mat_mask
        labeled_df['mean_importance'] = mean_importance

        return labeled_df, sim_mat_mask, mean_importance
    except AttributeError:
        print(len(full_vec), len(summ_vec))
        pass

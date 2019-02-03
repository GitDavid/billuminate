import os
import pandas as pd
import json
import tqdm


def aggregate_propublica_data(congress_session, bill_type):
    raw_path = '/Users/melissaferrari/Projects/repo/congress/data-propublica/'
    bill_path_ext = '{}/{}/'.format(congress_session, bill_type)
    bill_path = raw_path + bill_path_ext
    bill_list = os.listdir(bill_path)

    bill_folder = bill_list[0]
    data_path = os.path.join(bill_path, bill_folder)
    with open(os.path.join(data_path, 'data.json')) as f:
        data = json.load(f)

    df = pd.DataFrame([data])

    for bill_folder in tqdm.tqdm(bill_list[1:]):
        if bill_folder.startswith(bill_type):
            data_path = os.path.join(bill_path, bill_folder)

            try:
                with open(os.path.join(data_path, 'data.json')) as f:
                    data = json.load(f)

                df = df.append([data])
            except Exception:
                pass
        else:
            pass

    return df


if __name__ == "__main__":

    congress_sessions = ['113']
    bill_types = ['hr', 's']
    for congress_session in congress_sessions:
        for bill_type in bill_types[1:]:
            print('congress_session = {}'.format(congress_session))
            print('bill_type = {}'.format(bill_type))
            save_path = '/Users/melissaferrari/Projects/repo/insight/'
            save_path += 'data_collection/'
            save_path += 'agg_propublica_{}{}.csv'.format(congress_session,
                                                          bill_type)
            print(save_path)
            if not os.path.isfile(save_path):
                df = aggregate_propublica_data(congress_session,
                                               bill_type)
                df.to_csv(save_path)
            else:
                print('{} already exists')

import os
import pandas as pd
import json
import tqdm


def get_bill_path(base_path, congress_session, bill_type):
    base_path = '/Users/melissaferrari/Projects/repo/congress/data-propublica/'
    bill_path_ext = '{}/{}/'.format(congress_session, bill_type)
    bill_path = base_path + bill_path_ext
    return bill_path


def get_json(file_path):
    with open(os.path.join(file_path, 'data.json')) as f:
        return json.load(f)


def aggregate_propublica_data(base_path, congress_session, bill_type):

    bill_path = get_bill_path(base_path, congress_session, bill_type)

    bill_folders = os.listdir(bill_path)
    bill_folders = [x for x in bill_folders if not x.startswith('.')]

    # Create initial entry for dataframe
    bill_folder = bill_folders[0]
    data_path = os.path.join(bill_path, bill_folder)

    data = get_json(os.path.join(data_path, 'data.json'))
    df = pd.DataFrame([data])

    # Fill the dataframe
    for bill_folder in tqdm.tqdm(bill_folders[1:]):
        if bill_folder.startswith(bill_type):
            data_path = os.path.join(bill_path, bill_folder)
            try:
                data = get_json(os.path.join(data_path, 'data.json'))
                df = df.append([data])
            except Exception:
                pass
        else:
            pass

    return df


if __name__ == "__main__":

    congress_sessions = ['112']
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

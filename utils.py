import pandas as pd


def keyword_split(df, key):
    df[key] = df[key].str.split("; ", n=20, expand=False)
    return df


def load_dataset(filepath, column_split):
    df = pd.read_csv(filepath[0])
    key_list = pd.read_csv(filepath[1])
    for i in range(0,len(column_split)):
        df = keyword_split(df, column_split[i])
    return df, key_list


def min_max_norm(d):
    d = (d - d.min()) / (d.max() - d.min())
    if d.isnull().any():
        d = 0
    return d


def shuffling(df, freq):
    neg = df[(df['label'] == 0)]
    pos = df[(df['label'] == 1)]
    pos_len = len(pos)
    neg = neg.sample(n=pos_len * freq)
    df = pd.concat([pos, neg])
    df = df.sample(frac=1).reset_index(drop=True)
    return df

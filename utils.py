import pandas as pd


def keyword_split(df, key):
    df[key] = df[key].str.split("; ", n=20, expand=False)
    return df


def load_dataset(filepath):
    df = pd.read_csv(filepath[0])
    key_list = pd.read_csv(filepath[1])
    df = keyword_split(df, 'keyword')
    df = keyword_split(df, 'author_name')
    df = df.drop_duplicates(['title'], keep='first')
    return df, key_list


def min_max_norm(d):
    d = (d - d.min()) / (d.max() - d.min())
    return d
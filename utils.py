import pandas as pd
from pylab import *
import pickle
import os.path


def keyword_split(df, key):
    """
    split sub column elements(separated by ;) and put it in a list
    :param df: dataset in a dataframe
    :param key: column name
    :return: dataset after splitting sub-column values
    """
    df[key] = df[key].str.split("; ", n=30, expand=False)
    return df


def load_dataset(filepath, column_split):
    """
    load dataset and keylist(keyword, integer id) into dataframes
    :param filepath: filepath[0]:dataset, filepath[1]:keyword list
    :param column_split: array of column names where we split the sub column attributes
    :return:
    """
    df = pd.read_csv(filepath[0])
    key_list = pd.read_csv(filepath[1])
    for i in range(0, len(column_split)):
        df = keyword_split(df, column_split[i])
    return df, key_list


def min_max_norm(d):
    """
    min-max normalization
    :param d: array to normalize
    :return: normalized array
    """
    d = (d - d.min()) / (d.max() - d.min())
    if d.isnull().any():
        d = 0
    return d


def shuffling(df, freq):
    """
    shuffling dataframe to make pos:neg ratio 1:freq
    :param df: training dataframe
    :param freq: ratio value
    :return: sampled dataframe
    """
    neg = df[(df['label'] == 0)]
    pos = df[(df['label'] == 1)]
    pos_len = len(pos)
    neg = neg.sample(n=pos_len * freq)
    df = pd.concat([pos, neg])
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def arrayToList(arr):
    """
    convert input type to list of array
    :param arr: input (types can be array / array([]) / array )
    :return:
    """
    if type(arr) == type(array([])):
        return arrayToList(arr.tolist())
    elif type(arr) == type([]):
        return [arrayToList(a) for a in arr]
    else:
        return arr


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


###save data
def save_data(data, data_path, domain, name, time):
    """
    save data to .pkl file
    :param data: data to save
    :param data_path: file saving path
    :param domain: obesity/apnea
    :param name: data name (eg. results, train_data etc)
    :param time: training time information

    """
    try:
        data
        filename = domain + "-" + name + "_" + str(time[1]) + "-" + str(time[2]) + ".pkl"
        filename_path = os.path.join(data_path, filename)
        with open(filename_path, "wb") as f:
            pickle.dump(data, f)
            print(filename_path)
    except NameError:
        print(name + ' not exist')


###load reslults
def load_data(data_path, domain, name, time):
    """
    load data from .pkl file
    :param data_path: path where the file is located
    :param domain: obesity/apnea
    :param name: data name (eg. results, train_data etc)
    :param time: training time information
    :return: loaded data
    """
    filename = domain + "-" + name + "_" + str(time[1]) + "-" + str(time[2]) + ".pkl"
    filename_path = os.path.join(data_path, filename)
    with open(filename_path, "rb") as f:
        data = pickle.load(f)
    print(data_path)
    return data

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=(0,1)))*(x_max-x_min)
    denom = X.max(axis=(0,1)) - X.min(axis=(0,1))
    denom[denom==0] = 1
    return x_min + nom/denom

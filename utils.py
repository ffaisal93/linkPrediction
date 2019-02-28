import pandas as pd
from pylab import *

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


def arrayToList(arr):
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
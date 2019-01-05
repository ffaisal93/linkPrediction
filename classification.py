import pandas as pd
import networkx as nx


def classification_train_data(g_train, g_test):
    train_labels = []
    train_rows = list(nx.non_edges(g_train))
    for edge in train_rows:
        if g_test.has_edge(edge[0], edge[1]):
            train_labels.append(1)
        else:
            train_labels.append(0)
    train_data = pd.DataFrame({'row_name': train_rows, 'label': train_labels})
    return train_data


def classification_test_data(g_test):
    test_data = pd.DataFrame({'row_name': list(set(g_test.edges()))})
    return test_data


def print_attributes(data, type):
    if type == 'train':
        print('positive train:', len(data[(data['label'] == 1)]),
              'negative train:', len(data[(data['label'] == 0)]),
              'train_size:', len(data))
    else:
        print('test_size:', len(data))

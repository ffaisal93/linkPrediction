import pandas as pd
import networkx as nx
import utils as ut


def classification_train_data_static(g_train, g_test):
    train_labels = []
    train_edge = []
    train_rows = list(nx.non_edges(g_train))
    for edge in train_rows:
        if g_test.has_edge(edge[0], edge[1]) or g_test.has_edge(edge[1], edge[0]):
            train_labels.append(1)
            train_edge.append(tuple(sorted(edge)))
        else:
            train_labels.append(0)
            train_edge.append(tuple(sorted(edge)))
    train_data = pd.DataFrame({'row_name': train_edge, 'label': train_labels})
    return train_data


def classification_train_data_dynamic(g_train, g_test, static_edge_set):
    train_labels = []
    train_edge = []
    train_rows = list(nx.non_edges(g_train))
    for edge in train_rows:
        sorted_edge = tuple(sorted(edge))
        if sorted_edge in static_edge_set:
            if g_test.has_edge(edge[0], edge[1]) or g_test.has_edge(edge[1], edge[0]):
                train_labels.append(1)
                train_edge.append(sorted_edge)
            else:
                train_labels.append(0)
                train_edge.append(sorted_edge)
    train_data = pd.DataFrame({'row_name': train_edge, 'label': train_labels})
    return train_data


def classification_test_data_static(g_test):
    test_data = pd.DataFrame({'row_name': list(set(g_test.edges()))})
    return test_data


def non_edge_feature_dataframe(g_train, g_test, g_parent, g_train_static, g_test_static, time, freq=5):
    train_data = {}
    total_pos = set()
    total_neg = set()
    ts = time[1]
    te = time[2]
    it_index = time[4]
    parent_data = classification_train_data_static(g_parent, g_test_static)
    print_attributes(parent_data, 'parent')
    train_data_static = classification_train_data_static(g_train_static, g_test_static)
    print_attributes(train_data_static, 'train')
    train_data_static = ut.shuffling(train_data_static, freq)
    print_attributes(train_data_static, 'train')
    static_edge_set = set(train_data_static['row_name'])
    for t in range(ts, te, it_index):
        train_data[t] = classification_train_data_dynamic(g_train[t], g_test_static, static_edge_set)
        print_attributes(train_data[t], 'train')
        total_pos.update(set(train_data[t][(train_data[t]['label'] == 1)]['row_name']))
        total_neg.update(set(train_data[t][(train_data[t]['label'] == 0)]['row_name']))
    test_data_static = classification_test_data_static(g_test_static)
    print_attributes(test_data_static, 'test')
    print("pos in time series:", len(total_pos), "neg in ts:", len(total_neg),
          "pos-neg ratio:", len(total_pos) / len(total_neg))
    return train_data, train_data_static, parent_data, test_data_static


def print_attributes(data, type):
    print(type)
    if type == 'train' or type == 'parent':
        print('positive train:', len(data[(data['label'] == 1)]),
              'negative train:', len(data[(data['label'] == 0)]),
              'train_size:', len(data))
    else:
        print('test_size:', len(data))

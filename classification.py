import pandas as pd
import networkx as nx
import utils as ut
import numpy as np
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Reshape
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


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
    total_edge = set()
    ts = time[1]
    te = time[2]
    it_index = time[4]
    parent_data = classification_train_data_static(g_parent, g_test_static)
    # print_attributes(parent_data, 'parent')
    train_data_static = classification_train_data_static(g_train_static, g_test_static)
    # print_attributes(train_data_static, 'train')
    train_data_static = ut.shuffling(train_data_static, freq)
    # print_attributes(train_data_static, 'train')
    static_edge_set = set(train_data_static['row_name'])
    for t in range(ts, te, it_index):
        train_data[t] = classification_train_data_dynamic(g_train[t], g_test_static, static_edge_set)
        # print_attributes(train_data[t], 'train')
        total_pos.update(set(train_data[t][(train_data[t]['label'] == 1)]['row_name']))
        total_neg.update(set(train_data[t][(train_data[t]['label'] == 0)]['row_name']))
        total_edge.update(train_data[t]['row_name'])
    test_data_static = classification_test_data_static(g_test_static)
    # print_attributes(test_data_static, 'test')
    print("pos in time series:", len(total_pos), "neg in time series:", len(total_neg),
          "pos-neg ratio:", len(total_pos) / len(total_neg), "total:", len(total_edge))
    return train_data, train_data_static, parent_data, test_data_static, list(total_edge)


def print_attributes(data, type):
    print(type)
    if type == 'train' or type == 'parent':
        print('positive train:', len(data[(data['label'] == 1)]),
              'negative train:', len(data[(data['label'] == 0)]),
              'train_size:', len(data))
    else:
        print('test_size:', len(data))


def reshape_feature_data_for_classification(train_data, edge_list, time):
    ts = time[1]
    te = time[2]
    it_index = time[4]
    time_range = te - ts
    total_sample = len(edge_list)
    feature_length = len(train_data[ts].iloc[1]) - 2
    X = np.zeros([total_sample, time_range, feature_length])
    y = np.zeros(total_sample)
    print("X shape:", X.shape, "y shape:", y.shape)
    for id, edge in enumerate(edge_list):
        for t in range(ts, te, it_index):
            if edge in set(train_data[t]['row_name']):
                sample_row = np.asarray(train_data[t].loc[train_data[t]['row_name'] == edge].values[0])
                X[id][t - ts] = sample_row[2:feature_length + 2]
                y[id] = sample_row[1]
    input_length = X.shape[1]
    input_dim = X.shape[2]
    if len(y.shape) == 1:
        output_dim = 1
    else:
        output_dim = len(y[0])
    data_len_dm = [input_length, input_dim, output_dim]
    return X, y, data_len_dm


def classification_model(X_train, X_test, y_train, y_test, data_len_dm, con, model_name):
    input_length = data_len_dm[0]
    input_dim = data_len_dm[1]
    output_dim = data_len_dm[2]
    batch_size = 64
    model = Sequential()
    model.add(LSTM(10, input_shape=(input_length, input_dim)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(20, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(40, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(output_dim, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        shuffle=True,
                        batch_size=batch_size,
                        epochs=con[0],
                        verbose=0)

    return model


def model_evaluate(model, X_test, y_test, batch_size, model_name):
    score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_score = auc(fpr, tpr)
    model_result = {
        'model name': model_name,
        'test score': score[0],
        'test accuracy': score[1],
        'auc': auc_score,
        'false positive': fpr,
        'true positive': tpr
    }
    return model_result

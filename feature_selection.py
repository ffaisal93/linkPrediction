import pandas as pd
import numpy as np
import networkx as nx
import utils as ut
import community
from documentCentrality import document_centrality


def node_and_article_feature(df, g):
    """
    prepare node features (recursive term centrality based on article, author, affiliation, closeness centrality)
    :param df: dataset in dataframe
    :param g: graph
    :return: prepared node feature dataframe
    """
    article_set = []
    node_set = []
    y_weight = []
    closeness = []
    author_set = []
    affiliation_1_set = []
    affiliation_2_set = []
    country_set = []
    for nd, row in g.node(data=True):
        node_set.append(nd)
        y_weight.append(len(row['year']))
        closeness.append(nx.closeness_centrality(g, nd))
        for art_s in row['art_id']:
            article_set.append(art_s)
        for author_s in row['author']:
            author_set.append(author_s)
        for affiliation_1_s in row['affiliation_1']:
            affiliation_1_set.append(affiliation_1_s)
        for affiliation_2_s in row['affiliation_2']:
            affiliation_2_set.append(affiliation_2_s)
        for country_s in row['country']:
            country_set.append(country_s)
    article_set = set(article_set)
    author_set = set(author_set)
    affiliation_1_set = set(affiliation_1_set)
    affiliation_2_set = set(affiliation_2_set)
    country_set = set(country_set)
    article_index = list(article_set)
    author_index = list(author_set)
    affiliation_1_index = list(affiliation_1_set)
    affiliation_2_index = list(affiliation_2_set)
    country_index = list(country_set)
    node_index = list(node_set)
    td = np.zeros((len(g.nodes()), len(article_set)))
    ta = np.zeros((len(g.nodes()), len(author_set)))
    taf1 = np.zeros((len(g.nodes()), len(affiliation_1_set)))
    taf2 = np.zeros((len(g.nodes()), len(affiliation_2_set)))
    tc = np.zeros((len(g.nodes()), len(country_set)))
    for nd, art in g.node(data='art_id'):
        for art_s in art:
            td[node_index.index(nd)][article_index.index(art_s)] = 1
    for nd, aut in g.node(data='author'):
        for aut_s in aut:
            ta[node_index.index(nd)][author_index.index(aut_s)] = 1
    for nd, af1 in g.node(data='affiliation_1'):
        for af1_s in af1:
            taf1[node_index.index(nd)][affiliation_1_index.index(af1_s)] = 1
    for nd, af2 in g.node(data='affiliation_2'):
        for af2_s in af2:
            taf2[node_index.index(nd)][affiliation_2_index.index(af2_s)] = 1
    for nd, co in g.node(data='country'):
        for co_s in co:
            tc[node_index.index(nd)][country_index.index(co_s)] = 1

    # document & author centrality feature
    car, cnar = document_centrality(td, 20)
    cat, cnat = document_centrality(ta, 20)
    caf1, cnaf1 = document_centrality(taf1, 20)
    caf2, cnaf2 = document_centrality(taf2, 20)
    cou, cnou = document_centrality(tc, 20)
    node_feature = pd.DataFrame({'node_index': node_index, 'y_weight': y_weight,
                                 'term_art': cnar,
                                 'term_aut': cnat,
                                 'term_af1': cnaf1,
                                 'term_af2': cnaf2,
                                 'term_coun': cnou,
                                 'closeness': closeness})
    return node_feature


def build_feature_set(df, kl, g, type):
    """
    middle function to prepare node feature dataframe for train data.
    Experimental aggregated centrality measure for parent graph
    :param df: dataset in dataframe
    :param kl: keyword list
    :param g: graph
    :param type: graph type (eg. parent, train)
    :return: prepared dataframe of node feature for a specific graph
    """
    # node and article feature
    node_feature = node_and_article_feature(df, g)
    if type == "parent":
        node_feature['aut+art'] = node_feature.apply(lambda row: 0.7 * row['term_art'] + row['term_aut'],
                                                     axis=1)
        node_feature['count'] = 0
    return node_feature


# def feature_partition(g, node_feature):
#     # d_c = {}
#     # partition = community.best_partition(g)
#     # g_main_com = g.copy()
#     # for com in set(partition.values()):
#     #     list_nodes = [node for node in partition.keys() if partition[node] == com]
#     #     H = g_main_com.subgraph(list_nodes)
#     #     d_c[com] = nx.degree_centrality(H)
#     # return partition, d_c
#
#     d_c = {}
#     aut = dict(zip(node_feature['node_index'], node_feature['term_aut']))
#     partition = community.best_partition(g, weight='weight')
#     g_main_com = g.copy()
#     for com in set(partition.values()):
#         list_nodes = [node for node in partition.keys() if partition[node] == com]
#         H = g_main_com.subgraph(list_nodes)
#         x = np.array([aut[j] for j in list_nodes])
#         maxv = set(x[np.argsort(x)[-3:]])
#         s = [(j, aut[j]) if ((len(list_nodes) >= 2) and aut[j] in maxv) else (j, 0) for j in list_nodes]
#         d_c[com] = dict(s)
#     return partition, d_c


def feature_y_weight(g, node, year, year_score, max_year_weight):
    """
    calculates the year weight value for a node
    :param g: static train graph containing node information for all train years
    :param node: node id
    :param year: array of years, in which node appears. eg. year=[1993,1995,1996,1999]
    :param year_score: predefined train year score dict.
        eg. {1993:1, 1994:4, 1995:9, 1996:16} for train period (1993-1996)
    :param max_year_weight: highest year score (not in use currently)
    :return: year weight value for a node
    """
    y_weight = 0
    for y in g.nodes[node]['year']:
        if y <= year:
            y_weight = y_weight + year_score[y] * len(g.nodes[node]['year'])
        #    y_weight = y_weight / max_year_weight
        else:
            pass
    return y_weight


def feature_node_type(p1, g):
    """
    given a set of grandparent keywords, calculate set of parents, children and guest keywords.
    :param p1: set of grandparent keywords
    :param g: graph
    :return: set of parents, children and guest keywords.
    """
    p2 = set()
    ch = set()
    for p in p1:
        nb1 = set(nx.all_neighbors(g, p))
        p2.update(nb1)
        for nbs in nb1:
            nb2 = set(nx.all_neighbors(g, nbs))
            ch.update(nb2)
    p2 = p2.difference(p1)
    ch = ch.difference(p1, p2)
    guest = set(g.nodes()).difference(p1, p2, ch)
    return p2, ch, guest


def feature_citation(g_parent, g_train, row, t, ts):
    """
    calculate temporal citation score for a node
    :param g_parent: parent graph (graph for previous year of training)
    :param g_train: dynamic training graph- dict of training graphs
    :param row: dataframe row for a specific node with it's feature values
    :param t: year for which the score is being calculated
    :param ts: start year of training period
    :return: temporal citation score for a node
    """
    cite_now = g_train[t].nodes[row['node_index']]['citation']
    if t == ts:
        g_prev = g_parent
    else:
        g_prev = g_train[t - 1]
    if g_prev.has_node(row['node_index']):
        cite_prev = g_prev.nodes[row['node_index']]['citation']
    else:
        cite_prev = 0
    cite = abs(cite_now - cite_prev)
    if cite == 0:
        cite = 1
    return cite


def dynamic_graph_feature_set(df, key_list, train_data, g_parent, g_train, g_train_static, time):
    """
    calculates node feature dict and train data dict (feature set for non-connected node pairs) for training period
    :param df: dataset in dataframe
    :param key_list: keyword list with integer ids
    :param train_data: train data dict (empty feature set for non-connected node pairs)
    :param g_parent: parent graph
    :param g_train: dynamic training graph- dict of training graphs
    :param g_train_static: static training graph
    :param time: time information
    :return: calculated dict of node features and train data
    """
    ts_train = time[1]
    ts_test = time[2]
    it_index = time[4]
    list_range = time[5]
    node_feature = {}
    year_score = {}
    x1 = range(ts_test - ts_train + 1)
    year_in = np.power(x1, 2)
    max_year_weight = sum(range(1, ts_test - ts_train + 1, 1))
    parent_keys_aut, parent_keys_art, parent_keys_deg = parent_key_from_parent_graph(df, key_list, g_parent, g_train,
                                                                                     time)
    for t in range(ts_train, ts_test, it_index):
        ########### node feature dataframe #############################################################################
        node_feature[t] = build_feature_set(df, key_list, g_train[t], "train")
        node_feature[t]['degree'] = node_feature[t].apply(lambda row:
                                                          len(g_train[t][row['node_index']]), axis=1)
        node_feature[t]['citation'] = node_feature[t].apply(lambda row: feature_citation(g_parent,
                                                                                         g_train, row, t, ts_train),
                                                            axis=1)
        year_score[t] = year_in[t - ts_train + 1]

        ######------calculate set of grapndparents, parents, children, guest--------------------------------------------
        p1_aut = parent_keys_aut.intersection(set(g_train[t].nodes()))
        p2_aut, ch_aut, guest_aut = feature_node_type(p1_aut, g_train[t])
        # print(len(p1),len(p2),len(ch),len(guest))
        # parent_keys = p1.union(p2)
        parent_keys_aut = node_feature[t].sort_values('term_aut', ascending=False)
        parent_keys_aut = parent_keys_aut.reset_index()
        parent_keys_aut = set(parent_keys_aut['node_index'][0:list_range])

        p1_art = parent_keys_art.intersection(set(g_train[t].nodes()))
        p2_art, ch_art, guest_art = feature_node_type(p1_art, g_train[t])
        # parent_keys_art = p1_art.union(p2_art)
        parent_keys_art = node_feature[t].sort_values('term_art', ascending=False)
        parent_keys_art = parent_keys_art.reset_index()
        parent_keys_art = set(parent_keys_art['node_index'][0:list_range])

        p1_deg = parent_keys_deg.intersection(set(g_train[t].nodes()))
        p2_deg, ch_deg, guest_deg = feature_node_type(p1_deg, g_train[t])
        # parent_keys_deg = p1_deg.union(p2_deg)
        parent_keys_deg = node_feature[t].sort_values('degree', ascending=False)
        parent_keys_deg = parent_keys_deg.reset_index()
        parent_keys_deg = set(parent_keys_deg['node_index'][0:list_range])

        #####-----------------------------------------------------------------------------------------------------------
        node_feature[t]['degrees'] = ut.min_max_norm(node_feature[t]['degree'])
        #  partition, d_c = feature_partition(g_train[t], node_feature[t])
        #  node_feature[t]['partition_id'] = node_feature[t].apply(lambda row:
        #                                                          partition[row['node_index']], axis=1)
        #  node_feature[t]['partition_cnt'] = node_feature[t].apply(lambda row:
        #                                                           d_c[partition[row['node_index']]]
        #                                                           [row['node_index']], axis=1)

        node_feature[t]['y_weight'] = node_feature[t].apply(lambda row:
                                                            feature_y_weight(g_train_static,
                                                                             row['node_index'],
                                                                             t,
                                                                             year_score,
                                                                             max_year_weight), axis=1)
        node_feature[t]['node_type_aut'] = node_feature[t].apply(lambda row:
                                                                 20 if row['node_index'] in p1_aut
                                                                 else 5 if row['node_index'] in p2_aut
                                                                 else 3 if row['node_index'] in guest_aut
                                                                 else 1 if row['node_index'] in ch_aut
                                                                 else 0, axis=1)

        node_feature[t]['node_type_art'] = node_feature[t].apply(lambda row:
                                                                 20 if row['node_index'] in p1_art
                                                                 else 5 if row['node_index'] in p2_art
                                                                 else 1 if row['node_index'] in guest_art
                                                                 else 3 if row['node_index'] in ch_art
                                                                 else 0, axis=1)
        node_feature[t]['node_type_deg'] = node_feature[t].apply(lambda row:
                                                                 20 if row['node_index'] in p1_deg
                                                                 else 5 if row['node_index'] in p2_deg
                                                                 else 1 if row['node_index'] in guest_deg
                                                                 else 3 if row['node_index'] in ch_deg
                                                                 else 0, axis=1)
        ######----------------------------------------------------------------------------------------------------------
        #############################  dataframe non-connected node pair features ######################################
        train_data[t] = train_data_frame_dynamic(train_data[t], node_feature[t], g_train[t])
    return node_feature, train_data


def train_data_frame_dynamic(train_data, node_feature, g):
    """
    calculate and prepare dataframe for non-connected node pair normalized features
    :param train_data: train data dataframe (empty feature set for non-connected node pairs)
    :param node_feature: calculated node feature dataframe
    :param g: training graph
    :return: calculated dataframe for non-connected node pair normalized features
    """
    ########## key-value pair for node-feature--------------------------------------------------------------------------
    aut = dict(zip(node_feature['node_index'], node_feature['term_aut']))
    af1 = dict(zip(node_feature['node_index'], node_feature['term_af1']))
    af2 = dict(zip(node_feature['node_index'], node_feature['term_af2']))
    coun = dict(zip(node_feature['node_index'], node_feature['term_coun']))
    art = dict(zip(node_feature['node_index'], node_feature['term_art']))
    closeness = dict(zip(node_feature['node_index'], node_feature['closeness']))
    # part_id = dict(zip(node_feature['node_index'], node_feature['partition_id']))
    year = dict(zip(node_feature['node_index'], node_feature['y_weight']))
    # part_cnt = dict(zip(node_feature['node_index'], node_feature['partition_cnt']))

    ######### non-connected node-pair feature calculate-----------------------------------------------------------------
    train_data['aut'] = train_data.apply(lambda row:
                                         aut[row['row_name'][0]] * aut[row['row_name'][1]], axis=1)
    train_data['art'] = train_data.apply(lambda row:
                                         art[row['row_name'][0]] * art[row['row_name'][1]], axis=1)
    train_data['af1'] = train_data.apply(lambda row:
                                         af1[row['row_name'][0]] * af1[row['row_name'][1]], axis=1)
    train_data['af2'] = train_data.apply(lambda row:
                                         af2[row['row_name'][0]] * af2[row['row_name'][1]], axis=1)
    train_data['coun'] = train_data.apply(lambda row:
                                          coun[row['row_name'][0]] * coun[row['row_name'][1]], axis=1)
    train_data['close'] = train_data.apply(lambda row:
                                           closeness[row['row_name'][0]] + closeness[row['row_name'][1]], axis=1)
    # train_data['part'] = train_data.apply(lambda row:
    #                                       1 if part_id[row['row_name'][0]] == part_id[row['row_name'][1]] else
    #                                       0, axis=1)
    train_data['y_weight'] = train_data.apply(lambda row:
                                              year[row['row_name'][0]] * year[row['row_name'][1]], axis=1)
    # train_data['part_cnt'] = train_data.apply(lambda row:
    #                                           part_cnt[row['row_name'][0]] *
    #                                           part_cnt[row['row_name'][1]]
    #                                           , axis=1)
    train_data['path3'] = train_data.apply(lambda row:
                                           len(list(nx.all_simple_paths
                                                    (g, source=row[0][0], target=row[0][1],
                                                     cutoff=3))), axis=1)
    train_data['cm'] = train_data.apply(lambda row:
                                        len(list(nx.common_neighbors
                                                 (g, row[0][0], row[0][1]))), axis=1)
    # train_data[t]['semantic_sim']=train_data[t].apply(lambda row:
    #                                                     word_vectors.n_similarity(
    #                                                         gr.node_label_find(key_list,row[0][0]).lower().split(),
    #                                                     gr.node_label_find(key_list,row[0][1]).lower().split()) ,axis=1)

    ########## key-value pair for node-feature--------------------------------------------------------------------------
    types_aut = dict(zip(node_feature['node_index'], node_feature['node_type_aut']))
    types_art = dict(zip(node_feature['node_index'], node_feature['node_type_art']))
    types_deg = dict(zip(node_feature['node_index'], node_feature['node_type_deg']))
    citation = dict(zip(node_feature['node_index'], node_feature['citation']))

    ######### non-connected node-pair feature calculate-----------------------------------------------------------------
    train_data['type'] = train_data.apply(lambda row:
                                          types_aut[row['row_name'][0]] * types_aut[row['row_name'][1]], axis=1)

    train_data['typeaut'] = train_data.apply(lambda row:
                                             (types_aut[row['row_name'][0]] *
                                              len(g.nodes[row['row_name'][0]]['author']) +
                                              len(g.nodes[row['row_name'][1]]['author']) *
                                              types_aut[row['row_name'][1]]), axis=1)

    train_data['typeart'] = train_data.apply(lambda row:
                                             (types_art[row['row_name'][0]] *
                                              len(g.nodes[row['row_name'][0]]['art_id']) +
                                              len(g.nodes[row['row_name'][1]]['art_id']) *
                                              types_art[row['row_name'][1]]), axis=1)

    train_data['typenode'] = train_data.apply(lambda row:
                                              (types_deg[row['row_name'][0]] *
                                               len(g[row['row_name'][0]]) +
                                               len(g[row['row_name'][1]]) *
                                               types_deg[row['row_name'][1]]), axis=1)

    train_data['y_weight1'] = train_data.apply(lambda row:
                                               (year[row['row_name'][0]] *
                                                len(g[row['row_name'][0]])
                                                + len(g[row['row_name'][1]]) *
                                                year[row['row_name'][1]]), axis=1)

    train_data['citation'] = train_data.apply(lambda row:
                                              types_aut[row['row_name'][0]] * citation[row['row_name'][0]]
                                              + types_aut[row['row_name'][1]] * citation[row['row_name'][1]], axis=1)
    ####### train data normalization -----------------------------------------------------------------------------------
    train_data['y_weight1'] = ut.min_max_norm(train_data['y_weight1'])
    train_data['typeart'] = ut.min_max_norm(train_data['typeart'])
    train_data['typeaut'] = ut.min_max_norm(train_data['typeaut'])
    train_data['typenode'] = ut.min_max_norm(train_data['typenode'])
    train_data['citation'] = ut.min_max_norm(train_data['citation'])
    ######### calculate topological metrics-----------------------------------------------------------------------------
    resource_allocation = list(nx.resource_allocation_index(g, list(train_data['row_name'])))
    jaccard_coef = list(nx.jaccard_coefficient(g, list(train_data['row_name'])))
    adamic = list(nx.adamic_adar_index(g, list(train_data['row_name'])))
    pref = list(nx.preferential_attachment(g, list(train_data['row_name'])))
    train_data['res_aloc'] = list(zip(*resource_allocation))[2]
    train_data['jac_coef'] = list(zip(*jaccard_coef))[2]
    train_data['adamic'] = list(zip(*adamic))[2]
    train_data['pref'] = list(zip(*pref))[2]

    ####### train data normalization -----------------------------------------------------------------------------------
    train_data['aut'] = ut.min_max_norm(train_data['aut'])
    train_data['art'] = ut.min_max_norm(train_data['art'])
    train_data['af1'] = ut.min_max_norm(train_data['af1'])
    train_data['af2'] = ut.min_max_norm(train_data['af2'])
    train_data['coun'] = ut.min_max_norm(train_data['coun'])
    train_data['close'] = ut.min_max_norm(train_data['close'])
    # train_data['type'] = ut.min_max_norm(train_data['type'])
    train_data['y_weight'] = ut.min_max_norm(train_data['y_weight'])
    # train_data['part_cnt'] = ut.min_max_norm(train_data['part_cnt'])
    train_data['path3'] = ut.min_max_norm(train_data['path3'])
    train_data['cm'] = ut.min_max_norm(train_data['cm'])
    train_data['pref'] = ut.min_max_norm(train_data['pref'])

    return train_data


def parent_key_from_parent_graph(df, key_list, g_parent, g_train, time):
    """
    calculate set of top-k keywords from parent graph based on centrality measures
    :param df: dataset in dataframe
    :param key_list: keyword list with integer ids
    :param g_parent: parent graph (previous year of training period)
    :param g_train: train graph (not in use here currently)
    :param time: time information
    :return: calculated top-k keyword set
    """
    list_range = time[5]
    parent_node_feature = build_feature_set(df, key_list, g_parent, "parent")

    parent_node_feature['degree'] = parent_node_feature.apply(lambda row: len(g_parent[row['node_index']]), axis=1)
    parent_node_feature['degree'] = ut.min_max_norm(parent_node_feature['degree'])

    parent_aut = parent_node_feature.sort_values('term_aut', ascending=False)
    parent_aut = parent_aut.reset_index()
    parent_aut = set(parent_aut['node_index'][0:list_range])

    parent_art = parent_node_feature.sort_values('term_art', ascending=False)
    parent_art = parent_art.reset_index()
    parent_art = set(parent_art['node_index'][0:list_range])

    parent_deg = parent_node_feature.sort_values('degree', ascending=False)
    parent_deg = parent_deg.reset_index()
    parent_deg = set(parent_deg['node_index'][0:list_range])

    return parent_aut, parent_art, parent_deg


def drop_feature_columns(train_data, columns, time):
    """
    drop a feature column from dataframe
    :param train_data: dataframe for non-connected node-pair features
    :param columns: columns to drop
    :param time: time information
    :return: dataframe after dropping columns
    """
    ts = time[1]
    te = time[2]
    it_index = time[4]
    if len(columns) != 0:
        for t in range(ts, te, it_index):
            train_data[t].drop(columns, inplace=True, axis=1)
    return train_data

# def classify_by_feature(train_data, time):
#     ts = time[1]
#     row = train_data[ts].columns.values.tolist()
#     feature_names = row[2:len(row)]
#     all_combinations = itertools.chain(*[itertools.combinations(feature_names, i + 1)
#                                          for i, _ in enumerate(feature_names)])
#     feature_list = list(all_combinations)
#     for feature in feature_list:
#         columns_drop = list(set(feature_names).difference(set(feature)))
#         train_data = drop_feature_columns(train_data, columns_drop, time)

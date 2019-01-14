import pandas as pd
import numpy as np
import networkx as nx
import utils as ut
import community
import itertools
from documentCentrality import document_centrality


def node_and_article_feature(df, g):
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
    # node and article feature
    node_feature = node_and_article_feature(df, g)
    if type == "parent":
        node_feature['aut+art'] = node_feature.apply(lambda row: 0.7 * row['term_art'] + row['term_aut'],
                                                     axis=1)
        node_feature['count'] = 0
    return node_feature


def feature_partition(g):
    d_c = {}
    partition = community.best_partition(g)
    g_main_com = g.copy()
    for com in set(partition.values()):
        list_nodes = [node for node in partition.keys() if partition[node] == com]
        H = g_main_com.subgraph(list_nodes)
        d_c[com] = nx.degree_centrality(H)
    return partition, d_c


def feature_y_weight(g, node, year, year_score, max_year_weight):
    y_weight = 0
    for y in g.nodes[node]['year']:
        if y <= year:
            y_weight = y_weight + year_score[y]
            y_weight = y_weight / max_year_weight
        else:
            pass
    return y_weight


def feature_node_type(p1, g):
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


def dynamic_graph_feature_set(df, key_list, train_data, g_parent, g_train, g_train_static, time):
    ts_train = time[1]
    ts_test = time[2]
    it_index = time[4]
    node_feature = {}
    year_score = {}
    max_year_weight = sum(range(1, ts_test - ts_train + 1, 1))
    parent_keys = parent_key_from_parent_graph(df, key_list, g_parent, g_train, time)
    for t in range(ts_train, ts_test, it_index):
        year_score[t] = t - ts_train + 1
        p1 = parent_keys.intersection(set(g_train[t].nodes()))
        p2, ch, guest = feature_node_type(p1, g_train[t])
        # print(len(p1),len(p2),len(ch),len(guest))
        parent_keys = p1.union(p2)
        partition, d_c = feature_partition(g_train[t])
        node_feature[t] = build_feature_set(df, key_list, g_train[t], "train")
        node_feature[t]['partition_id'] = node_feature[t].apply(lambda row:
                                                                partition[row['node_index']], axis=1)
        node_feature[t]['partition_cnt'] = node_feature[t].apply(lambda row:
                                                                 d_c[partition[row['node_index']]]
                                                                 [row['node_index']], axis=1)
        node_feature[t]['y_weight'] = node_feature[t].apply(lambda row:
                                                            feature_y_weight(g_train_static,
                                                                             row['node_index'],
                                                                             t,
                                                                             year_score,
                                                                             max_year_weight), axis=1)
        node_feature[t]['node_type'] = node_feature[t].apply(lambda row:
                                                             5 if row['node_index'] in p1
                                                             else 3 if row['node_index'] in p2
                                                             else 2 if row['node_index'] in guest
                                                             else 1 if row['node_index'] in ch
                                                             else 0, axis=1)
        train_data[t] = train_data_frame_dynamic(train_data[t], node_feature[t], g_train[t])
    return node_feature, train_data


def train_data_frame_dynamic(train_data, node_feature, g):
    aut = dict(zip(node_feature['node_index'], node_feature['term_aut']))
    af1 = dict(zip(node_feature['node_index'], node_feature['term_af1']))
    af2 = dict(zip(node_feature['node_index'], node_feature['term_af2']))
    coun = dict(zip(node_feature['node_index'], node_feature['term_coun']))
    art = dict(zip(node_feature['node_index'], node_feature['term_art']))
    closeness = dict(zip(node_feature['node_index'], node_feature['closeness']))
    part_id = dict(zip(node_feature['node_index'], node_feature['partition_id']))
    types = dict(zip(node_feature['node_index'], node_feature['node_type']))
    year = dict(zip(node_feature['node_index'], node_feature['y_weight']))
    part_cnt = dict(zip(node_feature['node_index'], node_feature['partition_cnt']))
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
    train_data['part'] = train_data.apply(lambda row:
                                          1 if part_id[row['row_name'][0]] == part_id[row['row_name'][1]] else
                                          0, axis=1)
    train_data['y_weight'] = train_data.apply(lambda row:
                                              year[row['row_name'][0]] + year[row['row_name'][1]], axis=1)
    train_data['type'] = train_data.apply(lambda row:
                                          types[row['row_name'][0]] * types[row['row_name'][1]], axis=1)
    train_data['part_cnt'] = train_data.apply(lambda row:
                                              part_cnt[row['row_name'][0]] *
                                              part_cnt[row['row_name'][1]], axis=1)
    train_data['path3'] = train_data.apply(lambda row:
                                           len(list(nx.all_simple_paths
                                                    (g, source=row[0][0], target=row[0][1],
                                                     cutoff=3))), axis=1)
    # train_data[t]['semantic_sim']=train_data[t].apply(lambda row:
    #                                                     word_vectors.n_similarity(
    #                                                         gr.node_label_find(key_list,row[0][0]).lower().split(),
    #                                                     gr.node_label_find(key_list,row[0][1]).lower().split()) ,axis=1)


    resource_allocation = list(nx.resource_allocation_index(g, list(train_data['row_name'])))
    jaccard_coef = list(nx.jaccard_coefficient(g, list(train_data['row_name'])))
    adamic = list(nx.adamic_adar_index(g, list(train_data['row_name'])))
    pref = list(nx.preferential_attachment(g, list(train_data['row_name'])))
    train_data['res_aloc'] = list(zip(*resource_allocation))[2]
    train_data['jac_coef'] = list(zip(*jaccard_coef))[2]
    train_data['adamic'] = list(zip(*adamic))[2]
    train_data['pref'] = list(zip(*pref))[2]

    # train_data['aut'] = ut.min_max_norm(train_data['aut'])
    # train_data['art'] = ut.min_max_norm(train_data['art'])
    # train_data['af1'] = ut.min_max_norm(train_data['af1'])
    # train_data['af2'] = ut.min_max_norm(train_data['af2'])
    # train_data['coun'] = ut.min_max_norm(train_data['coun'])
    # train_data['close'] = ut.min_max_norm(train_data['close'])
    # train_data['type'] = ut.min_max_norm(train_data['type'])
    # train_data['y_weight'] = ut.min_max_norm(train_data['y_weight'])
    # train_data['part_cnt'] = ut.min_max_norm(train_data['part_cnt'])
    # train_data['path3'] = ut.min_max_norm(train_data['path3'])

    return train_data


def parent_key_from_parent_graph(df, key_list, g_parent, g_train, time, list_range=10):
    ts_train = time[1]
    ts_test = time[2]
    it_index = time[4]
    parent_node_feature = build_feature_set(df, key_list, g_parent, "parent")
    for t in range(ts_train, ts_test, it_index):
        parent_node_feature['count'] = parent_node_feature.apply(lambda row:
                                                                 row['count'] + 1 if
                                                                 row['node_index'] in set(g_train[t].nodes()) else
                                                                 row['count'], axis=1)
    parent_node_feature['aut+art'] = ut.min_max_norm(parent_node_feature['aut+art'])
    parent_node_feature['count'] = ut.min_max_norm(parent_node_feature['count'])
    parent_node_feature['y_weight'] = ut.min_max_norm(parent_node_feature['y_weight'])
    dist_aut_art = np.linalg.norm(parent_node_feature['aut+art'] - parent_node_feature['count'])
    dist_y_weight = np.linalg.norm(parent_node_feature['y_weight'] - parent_node_feature['count'])
    print("dist_aut_art:", dist_aut_art, "dist_y_weight:", dist_y_weight)
    parent_1st = parent_node_feature.sort_values('aut+art', ascending=False)
    parent_1st = parent_1st.reset_index()
    # parent_1st.plot(y=["count", "y_weight", "aut+art"], figsize=(20, 5))
    parent_1st_set = set(parent_1st['node_index'][0:list_range])

    return parent_1st_set


def drop_feature_columns(train_data, columns, time):
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

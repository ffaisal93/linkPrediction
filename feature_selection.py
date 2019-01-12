import pandas as pd
import numpy as np
import networkx as nx
import utils as ut
from documentCentrality import document_centrality


def node_and_article_feature(df, g):
    article_set = []
    node_set = []
    y_weight = []
    closeness = []
    author_set = []
    for nd, row in g.node(data=True):
        node_set.append(nd)
        y_weight.append(len(row['year']))
        closeness.append(nx.closeness_centrality(g, nd))
        for art_s in row['art_id']:
            article_set.append(art_s)
        for author_s in row['author']:
            author_set.append(author_s)
    article_set = set(article_set)
    author_set = set(author_set)
    article_index = list(article_set)
    author_index = list(author_set)
    node_index = list(node_set)
    td = np.zeros((len(g.nodes()), len(article_set)))
    ta = np.zeros((len(g.nodes()), len(author_set)))
    for nd, art in g.node(data='art_id'):
        for art_s in art:
            td[node_index.index(nd)][article_index.index(art_s)] = 1
    for nd, aut in g.node(data='author'):
        for aut_s in aut:
            ta[node_index.index(nd)][author_index.index(aut_s)] = 1

    # document & author centrality feature
    car, cnar = document_centrality(td, 20)
    cat, cnat = document_centrality(ta, 20)
    node_feature = pd.DataFrame({'node_index': node_index, 'y_weight': y_weight,
                                 'term_art': cnar, 'term_aut': cnat, 'closeness': closeness})
    return node_feature


def build_feature_set(df, kl, g, type):
    # node and article feature
    node_feature = node_and_article_feature(df, g)
    if type == "parent":
        node_feature['aut+art'] = node_feature.apply(lambda row: 0.7 * row['term_art'] + row['term_aut'],
                                                     axis=1)
        node_feature['count'] = 0
    return node_feature


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
    #parent_1st.plot(y=["count", "y_weight", "aut+art"], figsize=(20, 5))
    parent_1st_set = set(parent_1st['node_index'][0:list_range])

    return parent_1st_set

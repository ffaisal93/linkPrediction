import pandas as pd
import numpy as np
import networkx as nx
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
    print(len(author_set))
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


def build_feature_set(df, kl, g, t0, t1):
    # node and article feature
    node_feature = node_and_article_feature(df, g)
    return node_feature


weight={}

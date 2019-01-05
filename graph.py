import itertools
import networkx as nx


def node_key_find(kl, label):
    key = kl[kl['keyword'] == label]['id'].iloc[0]
    return key


def node_label_find(kl, key):
    label = kl[kl['id'] == key]['keyword'].iloc[0]
    return label


def nodes_intersection(df, kl, t0, t1, t2):
    prelist = []
    postlist = []
    pre_df = df[(df['art_year'] >= t0) & (df['art_year'] < t1)]
    post_df = df[(df['art_year'] >= t1) & (df['art_year'] < t2)]
    for index, row in pre_df.iterrows():
        for label in row["keyword"]:
            node_id = node_key_find(kl, label)
            prelist.append(node_id)
    for index, row in post_df.iterrows():
        for label in row["keyword"]:
            node_id = node_key_find(kl, label)
            postlist.append(node_id)
    return set(prelist).intersection(set(postlist))


def build_graph(g, df, kl, nodes, t0, t1):
    g_df = df[(df['art_year'] >= t0) & (df['art_year'] < t1)]
    # node insert
    for index, row in g_df.iterrows():
        for label in row["keyword"]:
            node_id = node_key_find(kl, label)
            if node_id in nodes:
                if g.has_node(node_id):
                    g.nodes[node_id]['art_id'].add(row["art_id"])
                    g.nodes[node_id]['year'].add(row["art_year"])
                    g.nodes[node_id]['title'].add(row["title"])
                    g.nodes[node_id]['author'].update(row["author_name"])
                else:
                    art_id = {row["art_id"]}
                    year = {row["art_year"]}
                    title = {row["title"]}
                    author = set(row["author_name"])
                    g.add_node(node_id, art_id=art_id, year=year, title=title, author=author)
    # edge insert
    for index, row in g_df.iterrows():
        edges = list(itertools.combinations(row["keyword"], 2))
        for edge in edges:
            node1 = node_key_find(kl, edge[0])
            node2 = node_key_find(kl, edge[1])
            if (node1 in nodes) and (node2 in nodes) and (node1 != node2):
                if not g.has_edge(node1, node2):
                    g.add_edge(node1, node2, weight=1)
                else:
                    g[node1][node2]['weight'] = g[node1][node2]['weight'] + 1
    return g


def save_graph(g, name):
    nx.write_gpickle(g, name)


def load_graph(name):
    g = nx.read_gpickle(name)
    return g

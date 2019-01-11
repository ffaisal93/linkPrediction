import itertools
import networkx as nx
import os.path


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
                    g.add_node(node_id,
                               art_id=art_id,
                               year=year,
                               title=title,
                               author=author)
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


# graph build and save
def dynamic_train_test_graph_build(df, key_list, graphpath, time):
    ts = time[1]
    te = time[2]
    test_range = time[3]
    it_index = time[4]
    graph_type = ['parent', 'train', 'test']
    # nodes intersection between train and test
    nodes = nodes_intersection(df, key_list, ts, te, te + test_range)
    print(len(nodes))
    # dynamic train graph build
    for t in range(ts, te + test_range, it_index):
        if t < te:
            types = graph_type[1]
        else:
            types = graph_type[2]
        g = nx.Graph()
        graph = build_graph(g, df, key_list, nodes, t, t + 1)
        file = types + "_graph_" + str(t)
        file_name = os.path.join(graphpath, file + ".gpickle")
        save_graph(graph, file_name)
        print(file_name + " saved,", "nodes:",
              len(graph.nodes()), "edges:", len(graph.edges()))
    # static graph build
    for i in range(0, len(graph_type)):
        static_graph_build(df, key_list, graphpath, time, graph_type[i], nodes)


def static_graph_build(df, key_list, graphpath, time, graph_type, nodes):
    if graph_type == 'parent':
        ts = time[0]
        te = time[1]
    elif graph_type == 'train':
        ts = time[1]
        te = time[2]
    elif graph_type == 'test':
        ts = time[2]
        te = time[2] + time[3]
    g = nx.Graph()
    graph = build_graph(g, df, key_list, nodes, ts, te)
    file = graph_type + "_graph_" + str(ts) + "-" + str(te)
    file_name = os.path.join(graphpath, file + ".gpickle")
    save_graph(graph, file_name)
    print(file_name + " saved,", "nodes:", len(graph.nodes()),
          "edges:", len(graph.edges()))


# load all graphs
def graph_load(graphpath, time):
    ts = time[1]
    te = time[2]
    test_range = time[3]
    it_index = time[4]
    g_train = {}
    g_test = {}
    graph_type = ['parent', 'train', 'test']
    for t in range(ts, te + test_range, it_index):
        if t < te:
            file = graph_type[1] + "_graph_" + str(t)
            file_name = os.path.join(graphpath, file + ".gpickle")
            g_train[t] = load_graph(file_name)
            print(file_name + " loaded,", "nodes:", len(g_train[t].nodes()),
                  "edges:", len(g_train[t].edges()))
        else:
            file = graph_type[2] + "_graph_" + str(t)
            file_name = os.path.join(graphpath, file + ".gpickle")
            g_test[t] = load_graph(file_name)
            print(file_name + " loaded,", "nodes:", len(g_test[t].nodes()),
                  "edges:", len(g_test[t].edges()))
    # parent graph load
    ts = time[0]
    te = time[1]
    file = graph_type[0] + "_graph_" + str(ts) + "-" + str(te)
    file_name = os.path.join(graphpath, file + ".gpickle")
    g_parent = load_graph(file_name)
    print(file_name + " loaded,",
          "nodes:", len(g_parent.nodes()),
          "edges:", len(g_parent.edges()))
    # train graph static load
    ts = time[1]
    te = time[2]
    file = graph_type[1] + "_graph_" + str(ts) + "-" + str(te)
    file_name = os.path.join(graphpath, file + ".gpickle")
    g_train_static = load_graph(file_name)
    print(file_name + " loaded,",
          "nodes:", len(g_train_static.nodes()),
          "edges:", len(g_train_static.edges()))
    # test graph static load
    ts = time[2]
    te = time[2] + time[3]
    file = graph_type[2] + "_graph_" + str(ts) + "-" + str(te)
    file_name = os.path.join(graphpath, file + ".gpickle")
    g_test_static = load_graph(file_name)
    print(file_name + " loaded,",
          "nodes:", len(g_test_static.nodes()),
          "edges:", len(g_test_static.edges()))

    return g_train, g_test, g_parent, g_train_static, g_test_static

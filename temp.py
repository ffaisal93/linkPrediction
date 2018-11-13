import networkx as nx
import pandas as pd

df = pd.read_csv('select_top_10_percent____from___select_A.csv')
#Graphtype = nx.Graph()
#G = nx.from_pandas_edgelist(df, edge_attr='art_year', create_using=Graphtype)
print(df.head())
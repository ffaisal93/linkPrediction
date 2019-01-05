from tabulate import tabulate
from documentCentrality import document_centrality


def prm(df):
    print(tabulate(df, headers='keys', tablefmt="fancy_grid"))


if __name__ == '__main__':
    TD = [[1, 2, 3, 4], [2, 3, 4, 5], [1, 2, 7, 8], [3, 5, 1, 2]]
    TD = [[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 0, 0]]
    Cd, Ct = document_centrality(TD, 20)
    print(Cd, Ct)

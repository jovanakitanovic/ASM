import pickle
import networkx as nx
import operator
import matplotlib.pyplot as plot
import functions


def network_assortativity(network):
    """ U prvoj petlji se prolazi kroz svaku granu i uzima se prvi čvor koji se stavlja u y data i drugi čvor koji
    se stavlja u x data tako da se na grafiku vidi kako se čvor povezuje sa svojim susedima"""

    num = 0

    xdata = []
    ydata = []

    for i, j in network.edges:
        # num+=1
        # print(i,j,num)
        ydata.append(network.degree(i))
        xdata.append(network.degree(j))

    node_degree = []
    neighbours_averrage_degree = []

    for node in network.nodes:
        num += 1
        node_degree.append(network.degree(node))
        neighbours_averrage_degree.append((list(nx.average_neighbor_degree(network, nodes=[node]).values())[0]))

    print(len(neighbours_averrage_degree), len(node_degree))

    print(f"ASORTATIVNOST {nx.degree_assortativity_coefficient(network)}")

    plot.pyplot.scatter(neighbours_averrage_degree, node_degree, alpha=0.05)
    plot.pyplot.xlim(0, max(node_degree))
    plot.pyplot.ylim(0, max(neighbours_averrage_degree))
    plot.pyplot.ylabel('node degree')
    plot.pyplot.xlabel('average degree of neighbour')
    plot.pyplot.show()

    plot.pyplot.scatter(xdata, ydata, alpha=0.05)
    plot.pyplot.xlim(0, (max(node_degree) + 50))
    plot.pyplot.ylim(0, (max(node_degree) + 50))
    plot.pyplot.ylabel('node degree')
    plot.pyplot.xlabel('degree of neighbour')
    plot.pyplot.show()


def network_clasterization(network, randomGraphSameSize, randomGraphErdosRenyi):
    sorted_values = sorted(nx.clustering(network).items(), key=operator.itemgetter(1))
    print(sorted_values)

    print("\n")
    print(f"PROSEČAN STEPEN KLASTERISANJA {nx.average_clustering(network)}")
    print(f"GLOBALNI KOEFICIJIENT KLASTERIZACIJE {nx.transitivity(network)}")
    print("\n")

    print(f"PROSEČAN STEPEN KLASTERISANJA ERDOS-RENYI {nx.average_clustering(randomGraphErdosRenyi)}")
    print(f"GLOBALNI KOEFICIJIENT KLASTERIZACIJE ERDOS-RENYI {nx.transitivity(randomGraphErdosRenyi)}")
    print("\n")

    print(f"PROSEČAN STEPEN KLASTERISANJA RANDOM NET {nx.average_clustering(randomGraphSameSize)}")
    print(f"GLOBALNI KOEFICIJIENT KLASTERIZACIJE RANDOM NET {nx.transitivity(randomGraphSameSize)}")
    print("\n")


with open('data/data_cleaned/comments', 'rb') as file:
    data = pickle.load(file)

users = set(data['author'].unique()).union(set(data['parent_author'].unique()))
print('Postoji %d korisnika' % len(users))

G = nx.Graph()
G.add_nodes_from(users)

for _, author, parent_author, _ in data[['author', 'parent_author', 'id']].itertuples():
    if author != parent_author:
        if (author, parent_author) in G.edges:
            G.edges[author, parent_author]['weight'] += 1
        else:
            G.add_edge(author, parent_author, weight=1)

print("Broj cvorova %d" % len(G.nodes))

# n = G.number_of_nodes()
# m = G.number_of_edges()
#
# Gnm = nx.gnm_random_graph(n, m)
#
# p = (2 * float(m)) / (n * (n - 1))
# print(p)
#
# er_mreza = nx.erdos_renyi_graph(n, p)
#
# delta_m = m - er_mreza.number_of_edges()
# print(
#     f"Broj čvorova originalne mreže minus broj čvorova u ER mreži iznosi {delta_m}, što je odstupanje od {abs(float(delta_m)) * 100 / m}%")
#
# network_clasterization(G, Gnm, er_mreza)

# network_assortativity(G)

# functions.power_law_distribution(G, False)

# functions.centrality_analysis(G)

# functions.katz_centrality(G, 1, 1)
# functions.katz_centrality(G, 2, 0.5)
# functions.katz_centrality(G, 0.5, 0.1)

print('Gotovo')

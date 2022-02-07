import pickle
import networkx as nx

with open('data/data_cleaned/comments', 'rb') as file:
    data = pickle.load(file)

print(data)

users = set(data['author'].unique()).union(set(data['parent_author'].unique()))
print('Postoji %d korisnika' % len(users))

# print(users)

G = nx.Graph()
G.add_nodes_from(users)

print(data[['author', 'parent_author', 'id']])

for _, author, parent_author, _ in data[['author', 'parent_author', 'id']].itertuples():
    if author != parent_author:
        if (author, parent_author) in G.edges:
            G.edges[author, parent_author]['weight'] += 1
        else:
            G.add_edge(author, parent_author, weight=1)

print(G.edges.data('weight'))

print("Broj cvorova %d" % len(G.nodes))

output_path = "models/user_net.gml"

nx.write_gml(G, output_path)

print('Gotovo')

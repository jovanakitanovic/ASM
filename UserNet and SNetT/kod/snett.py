import networkx as nx
from functions import network_assortativity
import functions

network = nx.read_gml('models/SNetT.gml')
# network_assortativity(network)
functions.power_law_distribution(network, False)

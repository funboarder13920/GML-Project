import numpy as np
import networkx as nx
import arms
import random
from networkx.algorithms.approximation import *
import functools


def strong_obs_graph(n_nodes, alpha, beta, graph_arms=None):
	"""Generates a strongly connected graph"""
	# alpha influences the number of self edges removed. alpha=1 --> all self-edges will be removed
	# beta influences the number of "peer" edges removed. beta=1 --> only peer edges will be removed
	G = nx.DiGraph()
	m = beta/n_nodes-2

	if graph_arms == None:
		graph_arms = [arms.ArmBernoulli(0.5) for _ in range(n_nodes)]

	# Step 1: generate fully connected graph
	for nodeID in range(n_nodes):
		G.add_node(nodeID, arm=graph_arms[nodeID])
	for node1 in range(n_nodes):
		for node2 in range(n_nodes):
			G.add_edge(node1, node2)
	for nodeID in G.nodes():
		u = random.random()
		if u < alpha:
			G.remove_edge(nodeID, nodeID)
		elif u < alpha + beta:
			k = int((u - alpha)/m)
			# Let's find k+1 elements at random among edges (j,i)
			edges = list(range(n_nodes))
			edges.remove(nodeID)
			edgesToRemove = random.sample(edges, k+1)
			for neighID in edgesToRemove:
				G.remove_edge(neighID, nodeID)
	return G


def weak_nodes(G):
    weak_nodes = []
    for edge in list(G.edges()):
        if edge[1] not in weak_nodes:
            weak_nodes.append(edge[1])
    return weak_nodes


def strong_nodes(G):
	"""Returns a dictionnary of nodes"""
	#1) Find dual nodes (observed by all and themselves)
	#2) Find self-observed only nodes
	#3) Find peer-observed only nodes
	#4) Return everything in a dictionnary
	strong_nodes = {}
	strong_nodes["dual"] = []
	strong_nodes["self"] = []
	strong_nodes["peer"] = []
	node_list = G.nodes()
	edge_list = G.edges()

	for nodeID in G.nodes():
		self_observed = False
		peer_observed = True

		if (nodeID, nodeID) in G.edges():
			self_observed = True

		for neighID in G.nodes():
			if neighID != nodeID and (neighID, nodeID) not in G.edges():
				peer_observed = False

		if peer_observed and self_observed:
			strong_nodes["dual"].append(nodeID)
		elif peer_observed:
			strong_nodes["peer"].append(nodeID)
		elif self_observed:
			strong_nodes["self"].append(nodeID)
	return strong_nodes


def weak_dom_number(G):
    """Computes an approximation of the weak domination number of G"""
    H = G.subgraph(weak_nodes(G))
    weak_dom_set = dominating_set.min_edge_dominating_set(H)
    delta = len(weak_dom_set)
    return delta


def observability_type(G):
    nodes_type = strong_nodes(G)
    _exhausted = object()
    strongly = len(nodes_type["dual"]) + len(nodes_type["peer"]) + len(nodes_type["self"]) == len(G.nodes())
    # weakly = (not strongly) and functools.reduce(lambda u,v: u and v, [next(G.predecessors(node), _exhausted) != _exhausted for node in G.nodes()], True)
    weakly = (not strongly) and np.all([next(iter(G.predecessors(node)), _exhausted) != _exhausted for node in G.nodes()])
    unobservable = not( weakly or strongly)
    obs_type = 0
    if weakly:
        obs_type = 1
    if strongly:
        obs_type = 2
    return obs_type



def remove_edges(G, perturbations, n=None):
    obs_dict = {0:"unobservable", 1:"weakly observable", 2:"strongly observable"}
    H = G.copy()
    if n is None:
        n = len(perturbations)
    count = 1
    for edge in perturbations:
        if edge in H.edges() and count <= n:
            H.remove_edge(edge[0], edge[1])
            count += 1
    print("Edge {0} removed".format(edge))
    obs_type = observability_type(H)
    print("G is now {}".format(obs_dict[obs_type]))
    print("{} edges were removed".format(count-1))
    return H
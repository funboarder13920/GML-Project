import numpy as np
import networkx as nx
import arms
import random
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


def strong_nodes(G):
	"""Returns a dictionnary of nodes"""
	#1) Find dual nodes (observed by all and themselves)
	#2) Find self-observed only nodes
	#3) Find peer-observed only nodes
	#4) Return everything in a dictionnary
	strongNodes = {}
	strongNodes["dual"] = []
	strongNodes["self"] = []
	strongNodes["peer"] = []
	nodeList = G.nodes()
	edgeList = G.edges()

	for nodeID in G.nodes():
		selfObserved = False
		peerObserved = True

		if (nodeID, nodeID) in G.edges():
			selfObserved = True

		for neighID in G.nodes():
			if neighID != nodeID and (neighID, nodeID) not in G.edges():
				peerObserved = False

		if peerObserved and selfObserved:
			strongNodes["dual"].append(nodeID)
		elif peerObserved:
			strongNodes["peer"].append(nodeID)
		elif selfObserved:
			strongNodes["self"].append(nodeID)
	return strongNodes

def observability_type(G):
    nodes_type = strong_nodes(G)
    _exhausted = object()
    strongly = len(nodes_type["dual"]) + len(nodes_type["peer"]) + len(nodes_type["self"]) == len(G.nodes())
    weakly = (not strongly) and functools.reduce(lambda u,v: u and v, [next(G.predecessors(node), _exhausted) != _exhausted for node in G.nodes()], True)
    unobservable = not( weakly or strongly)
    obs_type = 0
    if weakly:
        obs_type = 1
    if strongly:
        obs_type = 2
    return obs_type
import numpy as np
import networkx as nx
import arms
import random


def generateStrongObsGraph(n_nodes, alpha, beta):
	G = nx.DiGraph()
	m = beta/n_nodes-2

	# Step 1: generate fully connected graph
	for nodeID in range(n_nodes):
		G.add_node(nodeID, arm=arms.ArmBernoulli(0.5))
		for neighID in range(nodeID+1):
			G.add_edge(nodeID, neighID)
			if neighID != nodeID:
				G.add_edge(neighID, nodeID)
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
				G.remove_edge(nodeID, neighID)
	return G


def findStrongNodes(G):
	"""Returns a dictionnary of nodes"""
	#1) Find dual nodes (observec by all and themselves)
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
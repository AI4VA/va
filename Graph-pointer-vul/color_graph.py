import json
import networkx as nx 
import csv
import matplotlib.pyplot as plt
import argparse
from networkx.drawing.nx_agraph import graphviz_layout
import subprocess
import os
import numpy as np

edgesType = ["FLOWS_TO", "USE", "DEF", "REACHES", "CONTROLS"]
# edgesType = ["USE", "DEF", "REACHES", "CONTROLS"]


def getStatementNode(nodeCSV):
	statementNodes = list()
	with open(nodeCSV, 'r') as csvfile:
		nodes = csvfile.readlines()
		for n in nodes:
			if len(n.strip().split(' \t')) > 1:
				n, lineNo = n.strip().split(' \t')
				nodeKey, nodeContent = n.strip().split()[0], n.strip().split()[1:]
				statementNodes.append(nodeKey)
				# yield nodeKey
	return statementNodes


def get_non_leaf_nodes(edges):
	non_leaf_nodes = []
	for edge in edges:
		if edge['type'] == 'IS_AST_PARENT':
			non_leaf_nodes.append(edge['start'])
	return non_leaf_nodes


def cumulativeSum(G, activations):
	# TODO: check the node availability in $activations when G is a whole graph
	cumActivations = dict()
	cumSum = dict()
	actValues = np.array([float(activations[node][0]) for node in G.nodes()])
	mean = np.mean(actValues)
	std = np.std(actValues)
	actValues = [float(activations[node][0]) for node in G.nodes()]
	# print(actValues)
	# print(sum(actValues))
	src = [n for n,d in G.in_degree() if d==0]
	for node in G.nodes():
		cumActivations[node] = float(activations[node][0]) 
		cumSum[node] = float(activations[node][0])
	result = cumActivations
	for prop in range(9):
		newResult = dict()
		for node in G.nodes():
			newResult[node] = result[node]
			temp = 0
			for p in G.predecessors(node):
				if p != node:
					edgeData = G.get_edge_data(p, node)
					weight = 0
					for i in range(len(edgeData)):
						weight += edgeData[i]['weight']
					temp += weight * result[p]
			newResult[node] += temp
		result = newResult
		# yield newResult
	# tgt = [n for n,d in G.out_degree() if d == 0]
	'''
	for tgt in G.nodes():
		cumActivations[tgt] = 0
		# for path in nx.all_simple_paths():

		ancestors = nx.algorithms.dag.ancestors(G, tgt)
		print([t for t in G.predecessors(tgt)])
		for node in ancestors:
			standardValue = float(activations[node][0]) - mean
			cumActivations[tgt] += standardValue
		if len(ancestors) > 0:
			cumSum[tgt] = cumActivations[tgt]
		else:
			cumSum[tgt] = np.min(actValues)
	'''
	'''
	successors = nx.bfs_successors(G, src[0])
	for (parent, children) in successors:
		for child in children:
			if cumActivations.get(child):
				cumActivations[child] += cumActivations[parent]
			else:
				cumActivations[child] = cumActivations[parent] + float(activations[child][0])
	'''
	return result
	# return cumActivations
	# return cumSum


def drawGraph(G, colors, labels, flowedges=None, reachedges=None):
	pos=graphviz_layout(G, prog='dot')
	nx.draw(G, pos, node_color=colors, cmap=plt.cm.RdPu) 
	nx.draw_networkx_labels(G, pos, labels=labels)
	if flowedges and reachedges:
		nx.draw_networkx_edges(G, pos, edgelist=flowedges, edge_color='r')
		nx.draw_networkx_edges(G, pos, edgelist=reachedges, edge_color='b')
	plt.show()


def checkLocJl(maxLines, srcName):
	name = srcName.split('.')[0].split("###")
	vulLines = [int(n) for n in name[1:]]
	# return 1 if set(vulLines).issubset(set(maxLines)) else 0
	match = 0
	for v in vulLines:
		if v in maxLines:
			match = 1
	return match


def checkLocSb(maxLines, srcFile):
	vulLines = list()
	with open(srcFile, 'r') as sf:
		srcLines = sf.readlines()
		for idx, s in enumerate(srcLines):
			if "_UNSAFE" in s:
				vulLines.append(idx + 1)

	return 1 if set(vulLines).issubset(set(maxLines)) else 0  # predictions are subset of goldens
	# return 1 if set(maxLines).issubset(set(vulLines)) else 0  # goldens are subset of predictions


def topNKeys(d, N):
	sortD = sorted(d.items(), key=lambda x: x[1], reverse=True)
	sortKeys = list()
	for e in sortD[:N]:
		sortKeys.append(e[0])
	return sortKeys


def buildGraph(edgeCSV, nodeCSV, actJson, lineJson, srcFile, subGraph=None, draw=False, N=1):
	dirName = os.path.dirname(edgeCSV)
	nodesCode = dict() # Concrete code of each node
	G = nx.MultiDiGraph()
	edges = list() # (strat, end)
	flowedge = list()
	reachedge = list()
	edgesFull = list() # (start, type, end)

	with open(lineJson, 'r') as lj:
		lineNum = json.load(lj)

	with open(edgeCSV, 'r') as csvfile:
		edgeReader = csv.DictReader(csvfile, delimiter='\t')
		for edge in edgeReader:
			# edgesFull.append(edge)
			if edge["type"] in edgesType:
				if edge["type"] == 'FLOWS_TO':
					edgeTuple = (edge["start"], edge["end"], 0.5)
					# Just for subgraph
					if edge["start"] in subGraph and edge["end"] in subGraph:
						flowedge.append((edge["start"], edge["end"]))
				elif edge["type"] == 'REACHES':
					edgeTuple = (edge["start"], edge["end"], 2)
					if edge["start"] in subGraph and edge["end"] in subGraph:
						reachedge.append((edge["start"], edge["end"]))
				#edgeTuple = (edge["start"], edge["end"])
				edges.append(edgeTuple)
	# G.add_edges_from(edges)
	G.add_weighted_edges_from(edges)
	with open(nodeCSV, 'r') as csvfile:
		nodeReader = csv.DictReader(csvfile, delimiter='\t')
		for node in nodeReader:
			nodesCode[node["key"]] = node["code"]
	with open(actJson, 'r') as jf:
		nodesInfo = json.load(jf)
	
	
	if subGraph is not None:
		subG = G.subgraph(subGraph)
		cumActivations = cumulativeSum(subG, nodesInfo)
		maxNodes = topNKeys(cumActivations, N)
		# maxNode = max(cumActivations, key=cumActivations.get)
		maxLines = [int(lineNum[node]) for node in maxNodes]
		# ck = checkLocSb(maxLines, srcFile)
		ck = checkLocJl(maxLines, os.path.basename(srcFile))
		# if ck:
			# draw = True

		if draw:
			subActivations = list()
			for idx, v in enumerate(subG.nodes()):
				subG.nodes[v]["code"] = nodesCode[v]
				subActivations.append(cumActivations[v])
				# subActivations.append(ca[v])
				'''
				if nodesInfo.get(v):
					subActivations.append(float(nodesInfo[v][0]))
				else:
					subActivations.append(min(actValues))
				'''
			subLabels = nx.get_node_attributes(subG, 'code')
			drawGraph(subG, subActivations, subLabels, flowedges=flowedge, reachedges=reachedge)
		print(dirName)
		return ck
	else:
		# TODO: To be concretized
		actValues = [float(value[0]) for value in nodesInfo.values()]
		activations = list()
		for idx, v in enumerate(G.nodes()):
			G.nodes[v]["code"] = nodesCode[v]
			if nodesInfo.get(v):
				activations.append(float(nodesInfo[v][0]))
			else:
				activations.append(min(actValues))

		nodeLabels = nx.get_node_attributes(G,'code')
		drawGraph(G, activations, nodeLabels)
	
	# plt.title(dirName)
	# plt.savefig(fname=dirName + ".png", dpi=300)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--single_dir', help="specific directory that we care")
	parser.add_argument('--batch_dir', help="batch processing the directory")
	args = parser.parse_args()
	if args.single_dir:
		srcFile = os.path.join(args.single_dir, os.path.basename(args.single_dir))
		edgeCSV = os.path.join(args.single_dir, "edges.csv")
		nodeCSV = os.path.join(args.single_dir, "nodes.csv")
		lineJson = os.path.join(args.single_dir, "node_lineno.json")
		nodeLineCSV = os.path.join(args.single_dir,
								   "nodes_normalized_with_line.csv")  # with line number and corresponding vulnerability
		statementNodes = getStatementNode(nodeLineCSV)
		jsFile = os.path.join(args.single_dir, "activation.json")
		buildGraph(edgeCSV, nodeCSV, jsFile, lineJson, srcFile, statementNodes, draw=True)
	else:
		cmd = ''
		cmd += "find " + args.batch_dir + " -name *activation.json;"
		result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=True)
		result = result.stdout.decode('utf-8')
		result = result.split("\n")
		actjsonResults = list(filter(None, result))
		correct = []
		for jsFile in actjsonResults:
			csvDir = os.path.dirname(jsFile)
			srcFile = os.path.join(csvDir, os.path.basename(csvDir))
			if ".cpp" in srcFile:
				continue
			edgeCSV = os.path.join(csvDir, "edges.csv")
			nodeCSV = os.path.join(csvDir, "nodes.csv")
			lineJson = os.path.join(csvDir, "node_lineno.json")
			nodeLineCSV = os.path.join(csvDir, "nodes_normalized_with_line.csv") # with line number and corresponding vulnerability
			statementNodes = getStatementNode(nodeLineCSV)
			c = buildGraph(edgeCSV, nodeCSV, jsFile, lineJson, srcFile, statementNodes, draw=False)
			correct.append(c)
		l = len(correct)
		correct = np.array(correct)
		print ("Correct prediction ratio: ", np.count_nonzero(correct) / l)
			

if __name__ == '__main__':
	main()


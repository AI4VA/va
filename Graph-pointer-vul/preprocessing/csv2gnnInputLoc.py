import gensim
import os
import argparse
import subprocess
import numpy as np
from gensim.models import Word2Vec
import csv
import json
import random

class StreamArray(list):
	"""
	Converts a generator into a list object that can be json serialisable
	while still retaining the iterative nature of a generator.

	IE. It converts it to a list without having to exhaust the generator
	and keep it's contents in memory.
	"""
	def __init__(self, generator):
		self.generator = generator
		self._len = 1

	def __iter__(self):
		self._len = 0
		for item in self.generator:
			yield item
			self._len += 1

	def __len__(self):
		"""
		Json parser looks for a this method to confirm whether or not it can
		be parsed
		"""
		return self._len


# We currently consider 12 types of edges mentioned in ICST paper
edgeType = {'IS_AST_PARENT': 1,
			'IS_CLASS_OF': 2,
			'FLOWS_TO': 3,
			'DEF': 4,
			'USE': 5,
			'REACHES': 6,
			'CONTROLS': 7,
			'DECLARES': 8,
			'DOM': 9,
			'POST_DOM': 10,
			'IS_FUNCTION_OF_AST': 11,
			'IS_FUNCTION_OF_CFG': 12}


def checkVul_sababi(cFile):
	with open(cFile, 'r') as f:
		fileString = f.read()
		'''
		# Only Cond_unsafe:
		if "BUFWRITE_COND_UNSAFE" in fileString and "BUFWRITE_COND_SAFE" not in fileString \
				and "BUFWRITE_TAUT_UNSAFE" not in fileString and "BUFWRITE_TAUT_SAFE" not in fileString:
			return 1
		elif "BUFWRITE_COND_SAFE" in fileString and "BUFWRITE_COND_UNSAFE" not in fileString \
				and "BUFWRITE_TAUT_UNSAFE" not in fileString and "BUFWRITE_TAUT_SAFE" not in fileString:
			return 0
		else:
			return 2
		'''
		# return (1 if "BUFWRITE_COND_UNSAFE" in fileString or "BUFWRITE_TAUT_UNSAFE" in fileString else 0)
		return (1 if "_UNSAFE" in fileString else 0)


def checkVul_juliet(cFile):
	return 1 if "bad" in os.path.basename(cFile) else 0


def check_line_target_juliet(cFile):
	target = -1
	if "###" in cFile:
		with open(cFile, 'r') as sf:
			l = sf.readlines()
			pLine = len(l)
		vulLines = cFile.split('.')[0].split('###')[1:]
		for vl in vulLines:
			if int(vl) > 0 and int(vl) <= pLine:
				target = int(vl)
				break
	else:
		target = 0
	return target

def check_line_target_devign(cFile, lineNodes):
	target = -1
	if "###" in cFile:
		vulLines = [int(l) for l in cFile.split('.')[0].split('###')[1:]]
		vulLoc = sorted(vulLines)[0]
		if lineNodes.get(vulLoc) is not None:
			target = lineNodes[vulLoc]
		else:
			avaiLines = list(lineNodes.keys())
			smaller = [i for i in avaiLines if i < vulLoc]
			if len(smaller) == 0:
				try:
					statementLoc = min(avaiLines)
				except:
					print("Bad resources:" + cFile)
					return -1
			else:
				statementLoc = max(smaller)
			target = lineNodes[statementLoc]
	elif "&&&" in cFile:
		vulLines = [int(l) for l in cFile.split('.')[0].split('&&&')[1:]]
		vulLoc = sorted(vulLines)[0]
		if lineNodes.get(vulLoc) is not None:
			target = lineNodes[vulLoc]
		else:
			avaiLines = list(lineNodes.keys())
			bigger = [i for i in avaiLines if i > vulLoc]
			if len(bigger) == 0:
				try:
					statementLoc = max(avaiLines)
				except:
					print("Bad resources:" + cFile)
					return -1
			else:
				statementLoc = min(bigger)
			target = lineNodes[statementLoc]
	elif "good" in cFile:
		target = 0
	return target


def get_non_leaf_nodes(edges):
	non_leaf_nodes = []
	# {child: parent}
	parent_map = {}
	for edge in edges:
		if edge['type'] == 'IS_AST_PARENT':
			non_leaf_nodes.append(edge['start'])
			if parent_map.get(edge['end']) is not None:
				print("Warning!")
			parent_map[edge['end']] = edge['start']
	return non_leaf_nodes, parent_map


def get_concat_vec(tokens, w2v, max_num_tokens=4):  #max num of tokens to concat
	try:
		if len(tokens) > max_num_tokens:
			# print(tokens)
			vectors = [w2v[token] for token in tokens[0:max_num_tokens-1]]  #lose information, leave last token dimension empty
			# vectors = [get_w2v(token, w2v) for token in tokens[0:max_num_tokens-1]]  #lose information, leave last token dimension empty
		else:
			vectors = [w2v[token] for token in tokens[0:max_num_tokens]]
			# vectors = [get_w2v(token, w2v) for token in tokens[0:max_num_tokens]]
	except:
		print (tokens)
		import pdb
		pdb.set_trace()
	return np.concatenate(vectors)


def pad_node_vector(v, pad_len):
	pad_len = max(0, pad_len - len(v))
	v = np.pad(v,(0,pad_len),'constant')    #v will become a 128 wide vector padded with 0s if required
	return v

def inputGenerationCls(sourceFile, nodeCSV, edgeCSV, wv, lineIdx=None, single=False):
	gInput = dict()
	gInput["targets"] = list()
	gInput["graph"] = list()
	gInput["node_features"] = list()

	with open(nodeCSV, 'r') as nc:
		nodes = nc.readlines()
		nodeMap = dict()
		with open(edgeCSV, 'r') as ec:
			reader = csv.DictReader(ec, delimiter='\t')
			edges = list()
			for e in reader:
				edges.append(e)
			nonLeafNodes, parent_map = get_non_leaf_nodes(edges)
			nonLeafNodes = set(nonLeafNodes)

		gInput["node_features"].append(np.zeros(256).tolist())  # Add a dummy node for non-vulnerables
		gInput["node_features"].append(np.ones(256).tolist())  # Add a dummy node for vulnerables
		# lineNodes = dict()
		for idx, n in enumerate(nodes):
			if len(n.strip().split(' \t')) > 1:
				n, lineNo= n.strip().split(' \t')
				# lineNodes[int(lineNo)] = idx + 1

			nodeKey, nodeContent = n.strip().split()[0], n.strip().split()[1:]
			if nodeKey in nonLeafNodes:
				fNrp = get_concat_vec(nodeContent, wv, 8)
			else:
				fNrp = get_concat_vec(nodeContent, wv, 8)
			fRp = pad_node_vector(fNrp, 256)
			# fRp = fNrp
			gInput["node_features"].append(fRp.tolist())
			nodeMap[nodeKey] = idx + 2


		targets = list()
		# target = check_line_target_juliet(sourceFile)
		target = checkVul_juliet(sourceFile)

		if target == -1:
			print("ALERT: Wrong targets!!!")
			pass
		'''
		elif target == 0:
			targets.append(target)
		else:
			if lineNodes.get(target) is not None:
				nodeTarget = lineNodes[target]
			else:
				return None
			targets.append(nodeTarget)
		'''
		targets.append(target)
		gInput["targets"].append(targets)

		for e in edges:
			start, end, eType = e["start"], e["end"], e["type"]
			if eType == "IS_FILE_OF":
				# We ignore this for now
				continue
			else:
				# Here is a small bug: when the result of get() is 0, it will not count. but keep it for now.
				if nodeMap.get(start) and nodeMap.get(end):
					edge = [nodeMap[start], edgeType[eType], nodeMap[end]]
					gInput["graph"].append(edge)
				else:
					continue

	return gInput

def processLineNodes(lineNodes, parentMap, nodeMap):
	newLineNodes = dict()
	newParentMap = dict()
	for j in parentMap.items():
		newParentMap[nodeMap[j[0]]] = nodeMap[j[1]]
	for i in lineNodes.items():
		if len(i[1]) == 1:
			newLineNodes[i[0]] = i[1][0]
		else:
			# check whether all children have the same parent
			parent = list()
			for child in i[1]:
				if newParentMap.get(child) is None:
					break
				else:
					parent.append(newParentMap[child])
			# parent = [newParentMap[child] for child in i[1]]
			if len(set(parent)) == 1:
				newLineNodes[i[0]] = parent[0]
			else:
				newLineNodes[i[0]] = sorted(i[1])[0]

	return newLineNodes


def inputGeneration(sourceFile, nodeCSV, edgeCSV, wv, lineIdx=None, single=False):
	gInput = dict()
	gInput["targets"] = list()
	gInput["graph"] = list()
	gInput["node_features"] = list()
	# gInput["node_lines"] = list()
	# gInput["line_targets"] = list()
	# gInput["node_targets"] = list()

	with open(nodeCSV, 'r') as nc:
		nodes = nc.readlines()
		nodeMap = dict()
		with open(edgeCSV, 'r') as ec:
			reader = csv.DictReader(ec, delimiter='\t')
			edges = list()
			for e in reader:
				edges.append(e)
			# nonLeafNodes = set(get_non_leaf_nodes(edges))
			nonLeafNodes, parent_map = get_non_leaf_nodes(edges)
			nonLeafNodes = set(nonLeafNodes)
		# if lineIdx is None:
			# lineIdx = 0
		# vul = -1
		gInput["node_features"].append(np.zeros(256).tolist())  # Add a dummy node for non-vulnerables
		lineNodes = dict()
		for idx, n in enumerate(nodes):
			# node = {'nodeKey': (idx, nodeRpt)}
			# node = set()
			if len(n.strip().split(' \t')) > 1:
				n, lineNo= n.strip().split(' \t')
				if lineNodes.get(int(lineNo)) is None:
					lineNodes[int(lineNo)] = list()
				lineNodes[int(lineNo)].append(idx + 1)

			nodeKey, nodeContent = n.strip().split()[0], n.strip().split()[1:]
			# gInput["node_lines"].append(lineIdx)
			# gInput["node_targets"].append(vul)
			# nrp = np.zeros(32)
			if nodeKey in nonLeafNodes:
				# fNrp = np.add(nrp, wv[nodeContent[0]])
				fNrp = get_concat_vec(nodeContent, wv, 8)
			else:
				fNrp = get_concat_vec(nodeContent, wv, 8)
			fRp = pad_node_vector(fNrp, 256)
			# fRp = fNrp
			gInput["node_features"].append(fRp.tolist())
			nodeMap[nodeKey] = idx + 1

		lineNodes = processLineNodes(lineNodes, parent_map, nodeMap)
		targets = list()
		# target = check_line_target_juliet(sourceFile)
		target = check_line_target_devign(sourceFile, lineNodes)
		if target == -1:
			print("ALERT: Wrong targets!!!")
			# return None
			# pass
		'''
		elif target == 0:
			targets.append(target)
		else:
			try:
				nodeTarget = lineNodes[target]
			except:
				return None
			targets.append(nodeTarget)
		'''
		targets.append(target)
		gInput["targets"].append(targets)
		'''
		if single:
			lineFile = os.path.join(os.path.dirname(nodeCSV), "node_lineno.json")
			with open(lineFile, 'w') as lf:
				json.dump(nodeLines, lf)
		'''
		for e in edges:
			start, end, eType = e["start"], e["end"], e["type"]
			if eType == "IS_FILE_OF":
				# We ignore this for now
				continue
			else:
				# Here is a small bug, but keep it for now.
				if nodeMap.get(start) and nodeMap.get(end):
					edge = [nodeMap[start], edgeType[eType], nodeMap[end]]
					gInput["graph"].append(edge)
				else:
					continue

	return gInput


def inputGenerator(srcDir, fileList, model, writeStream=None, lineNo=None, cls=False):
	good = 0
	bad = 0
	for nodeCSV in fileList:
		nodeDir = os.path.dirname(nodeCSV)
		edgeCSV = os.path.join(nodeDir, "edges.csv")
		with open(edgeCSV, 'r') as e:
			edges = e.readlines()
			if len(edges) <= 40:  # filter out the samples too small
				continue
		with open(nodeCSV, 'r') as n:
			nodes = n.readlines()
			if len(nodes) > 600:  # filter out the samples too large, to speed up the rnn process
				continue
		# The way to find source file is specific to Joern
		sourceFile = os.path.join(srcDir, nodeDir.split('/')[-1])

		if sourceFile.endswith(".cpp"):
			continue
		# two dummy nodes for classification
		if cls:
			graph_input = inputGenerationCls(sourceFile, nodeCSV, edgeCSV, model)
		else:
			graph_input = inputGeneration(sourceFile, nodeCSV, edgeCSV, model)
		# target = check_line_target_juliet(sourceFile)
		# target = checkVul_juliet(sourceFile)
		target = graph_input["targets"][0][0]
		if target == -1:
			continue  # filter out the wrong line numbers
		elif target == 0:
			good += 1
		else:
			if graph_input is not None:
				bad += 1

		if graph_input is not None:
			yield graph_input
			if writeStream is not None:
				writeStream.write(nodeCSV + '\n')

	print ("This time: good samples are " + str(good) + ", bad samples are " + str(bad))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--good_path', help='txt file contains paths of good samples')
	parser.add_argument('--bad_path', help='txt file contains paths of bad samples')
	parser.add_argument('--csv', help='normalized csv files to process')
	parser.add_argument('--src', help='source c files to process')
	parser.add_argument('--single', action='store_true', help='single .c file to single .json')
	parser.add_argument('--cls', action='store_true', help='classification task')
	parser.add_argument('--w2v', help='pre-trained word2vec')
	args = parser.parse_args()

	model = Word2Vec.load(args.w2v)
	splitTag = True
	if args.cls:
		cls = True
	else:
		cls = False
	if args.good_path and args.bad_path:
		with open(args.good_path, 'r') as gp:
			paths = gp.readlines()
			paths = [p.strip() for p in paths]
			with open(args.csv + "_test_good_GGNNinput.json", 'w') as gi:
				gInputGood = inputGenerator(args.src, paths, model, cls=cls)
				stream_array = StreamArray(gInputGood)
				for chunk in json.JSONEncoder().iterencode(stream_array):
					gi.write(chunk)
		with open(args.bad_path, 'r') as bp:
			paths = bp.readlines()
			paths = [p.strip() for p in paths]
			with open(args.csv + "_test_bad_GGNNinput.json", 'w') as gi:
				gInputGood = inputGenerator(args.src, paths, model, cls=cls)
				stream_array = StreamArray(gInputGood)
				for chunk in json.JSONEncoder().iterencode(stream_array):
					gi.write(chunk)
	else:
		cmd = ""
		# cmd += "cd " + args.csv + ";"
		# cmd += "find " + args.csv + " -name *nodes_normalized.csv;"
		cmd += "find " + args.csv + " -name *nodes_normalized_with_line.csv;"
		result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=True)
		result = result.stdout.decode('utf-8')
		result = result.split("\n")
		nodesResults = list(filter(None, result))

		if args.single:
			cnt = 0
			for nodeCSV in nodesResults:
				nodeDir = os.path.dirname(nodeCSV)
				edgeCSV = os.path.join(nodeDir, "edges.csv")
				# The way to find source file is specific to Joern
				sourceFile = os.path.join(args.src, nodeDir.split('/')[-1])
				gInput = inputGeneration(sourceFile, nodeCSV, edgeCSV, model, single=True)
				# gInputList.append(gInput)

				singleJson = os.path.join(nodeDir, 'GGNNinput_infer.json')
				with open(singleJson, 'w') as gi:
					json.dump([gInput], gi)
				cnt += 1
				print("Generated samples: ", cnt)
		elif splitTag:
			lineIdx = 0
			with open(args.csv + "_valid_list.txt", 'w') as vi:
				with open(args.csv + "_valid_GGNNinput.json", 'w') as gi:
					gInputValid = inputGenerator(args.src, nodesResults[:int(len(nodesResults) * 0.05)], model, writeStream=vi, cls=cls)
					stream_array = StreamArray(gInputValid)
					for chunk in json.JSONEncoder().iterencode(stream_array):
						gi.write(chunk)
					# json.dump(gInputList, gi)

			lineIdx = 0
			with open(args.csv + "_test_list.txt", 'w') as ti:
				with open(args.csv + "_test_GGNNinput.json", 'w') as gi:
					gInputTest = inputGenerator(args.src,
												nodesResults[int(len(nodesResults) * 0.05):int(len(nodesResults) * 0.1)], model, writeStream=ti, cls=cls)
					stream_array = StreamArray(gInputTest)
					for chunk in json.JSONEncoder().iterencode(stream_array):
						gi.write(chunk)

			lineIdx = 0
			with open(args.csv + "_train_list.txt", 'w') as ti:
				with open(args.csv + "_train_GGNNinput.json", 'w') as gi:
					gInputTrain = inputGenerator(args.src,
												nodesResults[int(len(nodesResults) * 0.1):], model, writeStream=ti, cls=cls)
					stream_array = StreamArray(gInputTrain)
					for chunk in json.JSONEncoder().iterencode(stream_array):
						gi.write(chunk)
		else:
			with open(args.csv + "_GGNNinput.json", 'w') as gi:
				gInputValid = inputGenerator(args.src, nodesResults, model, cls=cls)
				stream_array = StreamArray(gInputValid)
				for chunk in json.JSONEncoder().iterencode(stream_array):
					gi.write(chunk)

if __name__ == '__main__':
	main()

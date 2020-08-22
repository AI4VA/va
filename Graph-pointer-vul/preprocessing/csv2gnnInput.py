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


def get_non_leaf_nodes(edges):
	non_leaf_nodes = []
	for edge in edges:
		if edge['type'] == 'IS_AST_PARENT':
			non_leaf_nodes.append(edge['start'])
	return non_leaf_nodes


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


def inputGeneration(sourceFile, nodeCSV, edgeCSV, wv, lineIdx=None, single=False):
	gInput = dict()
	gInput["targets"] = list()
	gInput["graph"] = list()
	gInput["node_features"] = list()
	# gInput["node_lines"] = list()
	# gInput["line_targets"] = list()
	# gInput["node_targets"] = list()

	if single:
		nodeLines = dict()

	targets = list()
	# targets.append(checkVul_sababi(sourceFile))
	targets.append(checkVul_juliet(sourceFile))
	gInput["targets"].append(targets)
	with open(nodeCSV, 'r') as nc:
		nodes = nc.readlines()
		nodeMap = dict()
		with open(edgeCSV, 'r') as ec:
			reader = csv.DictReader(ec, delimiter='\t')
			edges = list()
			for e in reader:
				edges.append(e)
			nonLeafNodes = set(get_non_leaf_nodes(edges))
		# if lineIdx is None:
			# lineIdx = 0
		# vul = -1
		for idx, n in enumerate(nodes):
			# node = {'nodeKey': (idx, nodeRpt)}
			# node = set()
			if len(n.strip().split(' \t')) > 1:
				n, lineNo= n.strip().split(' \t')
				# n, lineNo, vuln = n.strip().split(' \t')
				lineIdx = lineNo
				# gInput["line_targets"].append(int(vuln))
				# lineIdx += 1
				# vul = int(vuln)
			nodeKey, nodeContent = n.strip().split()[0], n.strip().split()[1:]

			if single:
				nodeLines[nodeKey] = lineIdx
			# gInput["node_lines"].append(lineIdx)
			# gInput["node_targets"].append(vul)
			# nrp = np.zeros(32)
			if nodeKey in nonLeafNodes:
				# fNrp = np.add(nrp, wv[nodeContent[0]])
				fNrp = get_concat_vec(nodeContent, wv, 12)
			else:
				'''
			ntrp = wv[nodeContent[0]]
				if len(nodeContent[1:]) > 0:
					for token in nodeContent[1:]:
						nrp = np.add(nrp, wv[token])
					fNrp = np.divide(nrp, len(nodeContent[1:]))
				else:
					fNrp = nrp
			fRp = np.concatenate([ntrp, fNrp])
				'''
				fNrp = get_concat_vec(nodeContent, wv, 12)
			fRp = pad_node_vector(fNrp, 384)
			# fRp = fNrp
			gInput["node_features"].append(fRp.tolist())
			nodeMap[nodeKey] = idx

		if single:
			lineFile = os.path.join(os.path.dirname(nodeCSV), "node_lineno.json")
			with open(lineFile, 'w') as lf:
				json.dump(nodeLines, lf)
			
		for e in edges:
			start, end, eType = e["start"], e["end"], e["type"]
			if eType == "IS_FILE_OF":
				# We ignore this for now
				continue
			else:
				if nodeMap.get(start) and nodeMap.get(end):
					edge = [nodeMap[start], edgeType[eType], nodeMap[end]]
					gInput["graph"].append(edge)
				else:
					continue

	return gInput


def inputGenerator(srcDir, fileList, model, lineNo=None):
	good = 0
	bad = 0
	for nodeCSV in fileList:
		nodeDir = os.path.dirname(nodeCSV)
		edgeCSV = os.path.join(nodeDir, "edges.csv")
		with open(edgeCSV, 'r') as e:
			edges = e.readlines()
			if len(edges) <= 40:
				continue
		# The way to find source file is specific to Joern
		sourceFile = os.path.join(srcDir, nodeDir.split('/')[-1])

		if sourceFile.endswith(".cpp"):
			continue

		# lineIdx += len(gInput["line_targets"])
		if checkVul_juliet(sourceFile):
		# if checkVul_sababi(sourceFile):
			bad += 1
		else:
			good += 1
		yield inputGeneration(sourceFile, nodeCSV, edgeCSV, model)
	print ("This time: good samples are " + str(good) + ", bad samples are " + str(bad))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv', help='normalized csv files to process')
	parser.add_argument('--src', help='source c files to process')
	parser.add_argument('--single', action='store_true', help='single .c file to single .json')
	parser.add_argument('--lines', action='store_true', help='line-level information included')
	parser.add_argument('--w2v', help='pre-trained word2vec')
	args = parser.parse_args()

	model = Word2Vec.load(args.w2v)

	cmd = ""
	# cmd += "cd " + args.csv + ";"
	# cmd += "find " + args.csv + " -name *nodes_normalized.csv;"
	if args.lines:
		cmd += "find " + args.csv + " -name *nodes_normalized_with_line.csv;"
		result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=True)
		result = result.stdout.decode('utf-8')
		result = result.split("\n")
		nodesResults = list(filter(None, result))
	else:
		cmd += "find " + args.csv + " -name *nodes_normalized.csv;"
		result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=True)
		result = result.stdout.decode('utf-8')
		result = result.split("\n")
		nodesResults = list(filter(None, result))
		random.shuffle(nodesResults)

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
	else:
		lineIdx = 0
		with open(args.csv + "_valid_list.txt", 'w') as vi:
			vi.write("\n".join(nodesResults[:int(len(nodesResults) * 0.1)]))
		with open(args.csv + "_valid_GGNNinput.json", 'w') as gi:
			gInputValid = inputGenerator(args.src, nodesResults[:int(len(nodesResults) * 0.1)], model)
			stream_array = StreamArray(gInputValid)
			for chunk in json.JSONEncoder().iterencode(stream_array):
				gi.write(chunk)
				# json.dump(gInputList, gi)

		lineIdx = 0
		with open(args.csv + "_test_list.txt", 'w') as ti:
			ti.write("\n".join(nodesResults[int(len(nodesResults) * 0.1):int(len(nodesResults) * 0.2)]))
		with open(args.csv + "_test_GGNNinput.json", 'w') as gi:
			gInputTest = inputGenerator(args.src,
										nodesResults[int(len(nodesResults) * 0.1):int(len(nodesResults) * 0.2)], model)
			stream_array = StreamArray(gInputTest)
			for chunk in json.JSONEncoder().iterencode(stream_array):
				gi.write(chunk)

		lineIdx = 0
		with open(args.csv + "_train_list.txt", 'w') as ti:
			ti.write("\n".join(nodesResults[int(len(nodesResults) * 0.2):]))
		with open(args.csv + "_train_GGNNinput.json", 'w') as gi:
			gInputTrain = inputGenerator(args.src,
										nodesResults[int(len(nodesResults) * 0.2):], model)
			stream_array = StreamArray(gInputTrain)
			for chunk in json.JSONEncoder().iterencode(stream_array):
				gi.write(chunk)


if __name__ == '__main__':
	main()

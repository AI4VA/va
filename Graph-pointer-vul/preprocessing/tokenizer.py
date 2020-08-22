import os
import subprocess
import itertools
import random
import re
import argparse

documents=[]
last_func=''
num_funcs=0

#functions where:
# 1. joern fails to parse explicitly (gathered via Error msg)
# 2. joern fais to parse implicitly (gathered via ['ClassDef', 'UNKNOWN'] from this script)

#only during verification, still valid for word2vec
ignore_list = []

def read_skip_list():
   fd = open('skip_list','r')
   #fd = open('skip_list_openssl','r')
   skip_list = fd.read().splitlines()
   return skip_list


keywords = ['auto', 'break', 'case', 'const', 'continue', 'default',
			 'do', 'else', 'enum', 'extern', 'for', 'goto', 'if',
			 'register', 'return', 'signed', 'sizeof', 'static', 'switch',
			 'typedef', 'void', 'volatile', 'while', 'EOF', 'NULL',
			 'null', 'struct', 'union']

cpp_keywords = ['asm', 'bool', 'catch', 'class', 'const_cast', 'delete',
				'dynamic_cast', 'explicit', 'export', 'friend', 'false',
				'inline', 'mutable', 'namespace', 'new', 'operator', 'private',
				'protected', 'public', 'reinterpret_cast', 'static_cast', 
				'template', 'this', 'throw', 'true', 'try', 'typeid', 'typename',
				'using', 'virtual', 'wchar_t', 'uint32_t', 'size_t', 'int64_t',
				'int32_t', 'uint64_t']

types = ['char', 'double', 'float', 'int', 'long', 'short', 'unsigned', 'u64']

operators = ['+','-', '*', '/', '%', '++', '--', '=', '==', '!=', '<', '>', 
			'<=', '>=', '&&', '||', '!', '&', '|', '^', '~', '<<', '>>',
			'+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '^=', '!=', '?=',
			'->', '.', '(', ')', '[', ']', ',', '{', '}', ':', ';', '::', '|=',
			'?', '...'
			]

directives = ['#include', '#define', '#undef', '#', '##',
			 '#ifdef', '#ifndef', '#if', '#else', '#elif', '#endif'
			 ]

libfuncs = ['printf', 'scanf', 'cin', 'cout', 'clrscr', 'getch', 'strlen',
			'gets', 'fgets', 'getchar', 'main', 'malloc', 'calloc', 'realloc', 
			'free', 'abs', 'div', 'abort', 'exit', 'system', 'atoi', 'atol',
			'atof', 'strtod', 'strtol', 'getchar', 'putchar', 'gets', 'puts', 
			'getc', 'ungetc', 'putc', 'getenv', 'setenv', 'putenv', 'perror', 
			'ferror', 'rand', 'sleep', 'time', 'clock', 'sprintf', 'fprintf',
			'fscanf', 'strcpy', 'strncpy', 'memset', 'memcpy', 'mmap',
			'copy_to_user', 'copy_from_user', 'snprintf'
		   ]

joern_keywords = ['CFG-ENTRY','CFG-EXIT', 'CFG-ERROR']

reserved = keywords + cpp_keywords + types + operators + directives + libfuncs + joern_keywords

def is_string(item):
	if item.startswith('"') and item.endswith('"'):
		return True
	return False

def is_char(item):
	if item.startswith('\'') and item.endswith('\''):
		return True
	return False

def is_reserved(item):
	if item in reserved:
		return True
	return False    

def is_identifier(item):
	return item.isidentifier()

def is_other_number(item):
	regex1 = '^0[xX]+[0-9a-fA-F]+[uUlL]*$'
	regex2 = '^(-)?[0-9]*(\.)?[0-9]*[fF]?$'
	regex3 = '^[0-9]+[uUlL]+$'
	regex4 = '^[0-9]+[Ee]+[+-]?[0-9]+[uUlL]+$'
	regex5 = '^(-)?[0-9]*(\.)?[0-9]*[Ee]+[+-]?[0-9]+[fF]?$'
	if re.match("|".join([regex1, regex2, regex3, regex4, regex5]), item):
		return True
	else:
		return False

def is_int(item):
	try:
		int(item)
		return True
	except:
		return False

def num_arr(item):
	#convert '123' to ['1','2','3']
	#assuming is_int has been called already
	return [i for i in item]


def partition_unknown_token(token, separator):
	tokens = []
	for t in token.partition(separator):
		if t:
			tokens = tokens + normalize_token(t)   #assuming handle_unknown is non-recursive
	return tokens

def handle_unknown(token):
	# taking care of some joern tokenization issues
	# eg ~ID
	if token.startswith('~'):
		return partition_unknown_token(token,'~')
	
	if token.startswith('('):
		return partition_unknown_token(token,'(')

	if token.endswith(')'):
		return partition_unknown_token(token,')')

	weird_joern_tokens = ['->*', '((', '))']
	if token in weird_joern_tokens: 
		return [token]
		
	return ['UNKNOWN']

def normalize_token(token):
	if is_char(token):
		return ['CHAR']
	elif is_string(token):
		return ['STRING']
	elif is_reserved(token):
		return [token]
	elif is_int(token):
		return num_arr(token)
	elif is_other_number(token):
		return ['NUM']
	elif is_identifier(token):    
		# return ['ID']
		return [token]
	else:
		return handle_unknown(token)


def handle_function_def_joern(tokens):
	# handle special case if FunctionDef '(xxx' and 'xxx)'    
	all_tokens = []
	for token in tokens.strip('"').split():
		if token.startswith('('):
			for sub_token in token.partition('('):
				if sub_token:
					all_tokens.append(sub_token)
		elif token.endswith(')'):
			for sub_token in token.partition(')'):
				if sub_token:
					all_tokens.append(sub_token)
		else:
			all_tokens.append(token)
	return all_tokens


def count_leading_quotes(node):
	if node[0] != '"':
		return 0

	i = 0
	while i < len(node) and node[i] == '"':
		i = i + 1
	
	if i > 6:
		raise ValueError("weird quote count")

	return i


def find_string(node):
	#parse potential strings of kind ""...."" xyz abc
	leading_quotes = 2
	trailing_quotes = 0
	i = 2
	while i < len(node) and trailing_quotes != leading_quotes:
		if node[i] == '"':
			trailing_quotes = trailing_quotes + 1  
		i = i + 1

	
	if trailing_quotes == leading_quotes:
		return i
	if node.split()[0] == '""':
		return 2
	#if len(node) == leading_quotes or trailing_quotes == leading_quotes:
	#    return i
	raise ValueError("quotes mismatch")
		

def _parse_node(node):
	if len(node) == 0:
		return []

	num_quotes = count_leading_quotes(node)

	if num_quotes == 0:
		tokens = node.split(maxsplit=1)
		if len(tokens) > 1:
			return [tokens[0]] + _parse_node(tokens[1])
		else:
			return [tokens[0]]
	
	if num_quotes == 2 or num_quotes==4:
		idx = find_string(node)
		return ['"STRING"'] + _parse_node(node[idx:])
	
	if node[-1] != '"':
		raise ValueError("weird joern string")
	return _parse_node(node[1:-1])
   

def check_line_no(token):
	if ':' in token:
		for c in token:
			if c.isalpha():
				return False
		return True
	else:
		return False


def parse_node_with_lines(node, loc, src_lines):
	all_tokens = []
	tokens = node.split(maxsplit=1)  # handling quoted code from joern sometimes

	joern_node_type = tokens[0]
	yield joern_node_type

	if len(tokens) > 1:
		if tokens[0] == 'FunctionDef':
			all_tokens = handle_function_def_joern(tokens[1])
		else:
			try:
				all_tokens = _parse_node(tokens[1])
			except ValueError as err:
				print(err.args)
			except RecursionError as err:
				print(f"Caught RecursionError for func: {last_func}")
				print(err.args)

	# TODO special 'FUNC' identifier for function calls--joern type 'Callee'?

	if len(all_tokens) > 0:
		# if check_line_no(all_tokens[-1]):  # if the last element is location, then extract the line number
			# line_no = all_tokens[-1].split(':')[0]
		if loc is not None:
			line_no = loc.split(':')[0]
			if len(all_tokens) > 0:
				for token in all_tokens:
					normalized_tokens = normalize_token(token)
					for t in normalized_tokens:
						yield t
			else:
				yield ""
			yield "\t" + line_no
			'''
			if "UNSAFE" in src_lines[int(line_no) - 1]:
				yield '\t1'
			else:
				yield '\t0'
			'''
		else:
			for token in all_tokens:
				normalized_tokens = normalize_token(token)
				for t in normalized_tokens:
					yield t

def parse_node(node):
	all_tokens = []
	tokens = node.split(maxsplit=1) #handling quoted code from joern sometimes

	# tokens = node.split()
   
	joern_node_type = tokens[0]
	yield joern_node_type
   
	if len(tokens) > 1:
		if tokens[0] == 'FunctionDef':
			all_tokens = handle_function_def_joern(tokens[1])
		# elif tokens[0] == 'CFGEntryNode':
			# all_tokens = ['CFG-ENTRY']
		# elif tokens[0] == 'CFGExitNode':
			# all_tokens = ['CFG-EXIT']
		# elif tokens[0] == 'CFGErrorNode':
			# all_tokens = ['CFG-ERROR']
		else:
			try:
				all_tokens = _parse_node(tokens[1])
			except ValueError as err:
				print(err.args)
			except RecursionError as err:
				print(f"Caught RecursionError for func: {last_func}")
				print(err.args)

	#TODO special 'FUNC' identifier for function calls--joern type 'Callee'?            

	if len(all_tokens) > 0:
		'''
		if check_line_no(all_tokens[-1]): # if the last element is location, then extract the line number
			line_no = all_tokens[-1].split(':')[0]
			if len(all_tokens[:-1]) > 0:
				for token in all_tokens[:-1]:
					normalized_tokens = normalize_token(token)
					for t in normalized_tokens:
						yield t
			else:
				yield "NULL_STRING"
			yield line_no
		else:
		'''
		for token in all_tokens:
				normalized_tokens = normalize_token(token)
				for t in normalized_tokens:
					yield t


def parse_nodes_with_lines(nodes, src_lines):
	for node in itertools.islice(nodes, 2, None):  # drop first two lines which is legend and filename
		node_parts = node.split('\t')  # when getting line numbers, split the token into more parts
		node_id = node_parts[0]
		if len(node_parts) > 2:
			node_body = "\t".join([node_parts[1], node_parts[2]])
		else:
			node_body = node_parts[1]
		if len(node_parts) > 3:
			node_loc = node_parts[3]
		else:
			node_loc = None
		'''
		if node_body.split('\t')[0] == 'CFGEntryNode' or node_body.split('\t')[0] == 'CFGExitNode' or \
				node_body.split('\t')[0] == 'CFGErrorNode' or node_body.split('\t')[0] == 'Symbol' or \
				node_body.split('\t')[0] == 'ParameterList':
			continue
		'''
		node_tokens = parse_node_with_lines(node_body, node_loc, src_lines)
		yield (node_id, list(node_tokens))  # ['aa','bb','cc']


def parse_nodes(nodes):
	for node in itertools.islice(nodes,2,None): #drop first two lines which is legend and filename
		node_parts = node.split(maxsplit=1)
		node_id = node_parts[0]
		node_body = node_parts[1]
		node_tokens = parse_node(node_body)
		yield (node_id, list(node_tokens))  #['aa','bb','cc']
	

def read_node_file_with_lineNo(node_file):
	proc = subprocess.Popen(
		['cut', '-f', '2,3,4,5', node_file],
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE)
	while True:
		line = proc.stdout.readline()
		if not line:
			break
		yield line.decode().strip() #\n is preserved inside printf


def read_node_file(node_file):
	proc = subprocess.Popen(
		['cut', '-f', '2,3,4', node_file],
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE)
	while True:
		line = proc.stdout.readline()
		if not line:
			break
		yield line.decode().strip() #\n is preserved inside printf


def read_node_files_with_lines(joern_dir, func_dir, src_dir):
	global last_func
	global num_funcs
	skip_list = read_skip_list()
	cnt = 0
	for func in os.listdir(joern_dir+'/'+func_dir):
		num_funcs = num_funcs + 1
		#func = str(random.randrange(1,100000))+'.c'
		if func_dir+'/'+func in skip_list:
			continue
		#print(func)
		last_func = func
		#node_file = joern_dir+'/'+func_dir+'/'+func+'/'+func+'/'+func+'/nodes.csv'
		node_file = joern_dir + '/' + func_dir + '/' + func + '/nodes.csv'
		src_file = joern_dir + '/' + src_dir + '/' + func
		if not os.path.exists(node_file):
			print("Could not find file:", node_file, "; continuing")
			continue
		if not os.path.exists(src_file):
			print("Could not find file:", src_file, "; continuing")
			continue
		# nodes = read_node_file(node_file)
		nodes = read_node_file_with_lineNo(node_file)
		# parsed_nodes = parse_nodes(nodes)
		with open(src_file, 'r') as sf:
			src_lines = sf.readlines()
		parsed_nodes = parse_nodes_with_lines(nodes, src_lines)
		yield from write_node_file_with_lines(func, parsed_nodes, joern_dir, func_dir)
		cnt += 1
		print ("Tokenized samples: ", cnt)
		#for (_,node_tokens) in parsed_nodes:   #not returning node_id
		#    yield node_tokens


def read_node_files(joern_dir, func_dir):
	global last_func
	global num_funcs
	skip_list = read_skip_list()
	for func in os.listdir(joern_dir+'/'+func_dir):
		num_funcs = num_funcs + 1
		#func = str(random.randrange(1,100000))+'.c'
		if func_dir+'/'+func in skip_list:
			continue
		#print(func)
		last_func = func
		#node_file = joern_dir+'/'+func_dir+'/'+func+'/'+func+'/'+func+'/nodes.csv'
		node_file = joern_dir + '/' + func_dir + '/' + func + '/nodes.csv'
		if not os.path.exists(node_file):
			print("Could not find file:", node_file, "; continuing")
			continue
		nodes = read_node_file(node_file)
		# nodes = read_node_file_with_lineNo(node_file)
		parsed_nodes = parse_nodes(nodes)
		yield from write_node_file(func, parsed_nodes, joern_dir, func_dir)
		#for (_,node_tokens) in parsed_nodes:   #not returning node_id
		#    yield node_tokens
	  

def write_node_file_with_lines(func, parsed_nodes, joern_dir, func_dir):
	node_file = joern_dir + '/' + func_dir + '/' + func + '/nodes_normalized_with_line.csv'
	# if os.path.isfile(node_file):
		# input("File " +  node_file + " already exists. Press key to overwrite, Ctrl+C to abort")
	fd = open(node_file,'w')
	for node in parsed_nodes:   #node == (node_id,[node_tokens])
		node_id = node[0]
		node_tokens = node[1]
		fd.write(node_id+" ")
		for token in node_tokens:
			fd.write(token+" ")
		fd.write("\n")
		yield node_tokens
	fd.close()


def write_node_file(func, parsed_nodes, joern_dir, func_dir):
	node_file = joern_dir+'/'+func_dir+ '/' + func +'/nodes_normalized.csv'
	# node_file = joern_dir + '/' + func_dir + '/' + func + '/nodes_normalized_with_line.csv'
	# node_file = joern_dir + '/' + func_dir + '/' + func + '/nodes.csv'
	# if os.path.isfile(node_file):
		# input("File " +  node_file + " already exists. Press key to overwrite, Ctrl+C to abort")
	fd = open(node_file,'w')
	for node in parsed_nodes:   #node == (node_id,[node_tokens])
		node_id = node[0]
		node_tokens = node[1]
		fd.write(node_id+" ")
		for token in node_tokens:
			fd.write(token+" ")
		fd.write("\n")
		yield node_tokens
	fd.close()    
		
def collect_normalized_node_files(filename,joern_dir, func_dir, line=False): #ssuming normalized files were already created
	skip_list = read_skip_list()
	wfd = open(filename, 'w')
	for func in os.listdir(joern_dir+'/'+func_dir):
		if func_dir+'/'+func in skip_list:
			continue
		# normalized_node_file = joern_dir+'/'+func_dir+'/'+func+'/'+func+'/'+func+'/nodes_normalized.csv'
		if line:
			normalized_node_file = joern_dir + '/' + func_dir + '/' + func + '/nodes_normalized_with_line.csv'
		else:
			normalized_node_file = joern_dir + '/' + func_dir + '/' + func + '/nodes_normalized.csv'
		if not os.path.exists(normalized_node_file):
			print("Could not find file:", normalized_node_file, "; continuing")
			continue
		rfd = open(normalized_node_file,'r')
		for line in rfd.readlines():
			node_tokens = line.split(maxsplit=1)[1]
			wfd.write(node_tokens)
		rfd.close()
	wfd.close()

def print_documents(documents):
	for doc in documents:
		print(list(doc))

def write_documents_to_file(filename, documents):
	if os.path.isfile(filename):
		input("File already exists. Press key to overwrite, Ctrl+C to abort")
	fd = open(filename,'w')
	for stmt in documents:
		for token in stmt:
			fd.write(token+" ")
		fd.write("\n")
	fd.close()    

def print_one_doc(joern_dir, func_dir, func=''):
	if not func:
		func = last_func
	# node_file = joern_dir+'/'+func_dir+'/'+func+'/'+func+'/'+func+'/nodes.csv'
	node_file = joern_dir + '/' + func_dir + '/' + func + '/nodes.csv'
	nodes = read_node_file(node_file)
	doc = parse_nodes(nodes)
	print(node_file)

	i = 0
	for _doc in doc:
		print(i, list(_doc))
		i = i + 1
	
def verify_documents(joern_dir, func_dir, documents):
	for doc in documents:
		for token in doc:
			if token == 'UNKNOWN':
				if func_dir+'/'+last_func in ignore_list:
					continue
				print('num_funcs:', num_funcs,'UNKNOWN token found in file:', last_func)
				#print_one_doc(joern_dir, func_dir)
				break
				#exit()

def verify_documents_UNKNOWN(joern_dir, func_dir, documents):
	for stmt in documents:
		tokens = list(stmt)
		if 'UNKNOWN' in tokens:
			if 'STRING' not in tokens: 
				func = last_func
				# node_file = joern_dir+'/'+func_dir+'/'+func+'/'+func+'/'+func+'/nodes.csv'
				node_file = joern_dir + '/' + func_dir + '/' + func + '/nodes.csv'
				print(node_file)
				print(tokens)

def verify_selected_documents_UNKNOWN(joern_dir, func_dir, selected_funcs_file):
	fd = open(selected_funcs_file,'r') #created using 'grep UNK word2vec_out | awk '{print $NF}'| uniq > word2vec_out_UNK'
	for func in fd.readlines():
		func = func.strip('\n')
		# node_file = joern_dir+'/'+func_dir+'/'+func+'/'+func+'/'+func+'/nodes.csv'
		node_file = joern_dir + '/' + func_dir + '/' + func + '/nodes.csv'
		nodes = read_node_file(node_file)
		doc = parse_nodes(nodes)
		i = 0
		for stmt in doc:
			tokens = list(stmt)
			if 'UNKNOWN' in tokens:
				if 'STRING' not in tokens: 
					print(node_file)
					print(i, tokens)
			i = i + 1

def verify_selected_documents_QUOTES_MISMATCH(joern_dir, func_dir, selected_funcs_file, documents):
	fd = open(selected_funcs_file,'r') 
	for func in fd.readlines():
		func = func.strip('\n')
		print_one_doc(joern_dir, func_dir, func)
		#input("Press Enter to continue...")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--joern_dir', help='Joern files directory')
	parser.add_argument('--func_dir', help='Function files directory')
	parser.add_argument('--src_dir', help='Source files directory')
	parser.add_argument('--w2v_input', help='input file for w2v training')
	parser.add_argument('--line', action='store_true', help='line-level information included')
	args = parser.parse_args()
	# joern_dir='/home/robin/Documents/research/GGNN-bug-detection/Juliet_dataset/'
	joern_dir = args.joern_dir
	# func_dir='functionFiles_all_parsed'
	func_dir = args.func_dir
	# src_dir = 'functionFiles_all'
	src_dir = args.src_dir
	# word2vec_input_filename = 'word2vec_input_sentences_testset'
	word2vec_input_filename = args.w2v_input
		
	#joern_dir='/mnt/m1/joern_openssl'
	#func_dir='functions_openssl'

	#print_one_doc(joern_dir,func_dir,"34122.c")
	#exit()
	if args.line:
		documents = read_node_files_with_lines(joern_dir, func_dir, src_dir)
	else:
		documents = read_node_files(joern_dir, func_dir)

	# verify_documents_UNKNOWN(joern_dir, func_dir, documents)
	#verify_documents(joern_dir, func_dir, documents)
	#verify_selected_documents_UNKNOWN(joern_dir, func_dir, 'word2vec_out_UNK')
	#verify_selected_documents_QUOTES_MISMATCH(joern_dir, func_dir, 'word2vec_out_quotes_mismatch', documents)
	write_documents_to_file(word2vec_input_filename, documents)
	
	collect_normalized_node_files(word2vec_input_filename, joern_dir, func_dir, line=args.line) #ssuming normalized files were already created

if __name__ == "__main__":
	main()
#documents is actually not the right word to use, its actually a list of sentenses
#eg documents = [['first', 'sentence'], ['second', 'sentence']]



#for each file
	#read node list
	#IF NODE-TYPE IS NOT PART OF WORD2VEC VOCAB
		#for each node of type Statement 
			#statement = code string
	#ELSE
		#for each node
			#statement = node type + code string
			#tokenize statement
			#for each token
				#drop '"'
				#if token not in vocab(keyword, numbers, lib_func, etc)
					#convert token to str-const 'ID', 'STR', 'CHAR'
			#emit statement to vocab file for word2vec    
#run word2vec over vocab == stmt vocab file + node_types and get embedings for each token in vocab





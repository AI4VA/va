import csv
import argparse
import os
import subprocess


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
				'using', 'virtual', 'wchar_t', 'uint32_t']

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

reserved = set(keywords + cpp_keywords + types + operators + directives + libfuncs + joern_keywords)


def normalize(nodeCode, identifiers):
	codeString = list()
	if nodeCode == '':
		return nodeCode
	else:
		tokens = nodeCode.split()
		for t in tokens:
			if isChar(t):
				codeString.append('CHAR')
			elif isString(t):
				codeString.append('STRING')
			elif isReserved(t):
				codeString.append(t)
			elif isInt(t):
				#Different with Sahil's code: not split int
				codeString.append(t)
			elif isOtherNum(t):
				#Different with Sahil's code: not abstract numbers
				codeString.append(t)
			elif isIdentifier(t, identifiers):    
				codeString.append('ID')
			else:
				codeString.append(handleUnknown(t))
		codeStr = ' '.join(codeString)
		return codeStr


def isChar(token):
	return (True if token.startswith("'") else False)


def isString(token):
	return (True if token.startswith('"') else False)


def isReserved(token):
	return (True if token in reserved else False)


def isInt(token):
	try: 
		int(token)
		return True
	except ValueError:
		return False


def isOtherNum(token):
	try:
		float(token)
		return True
	except ValueError:
		return False
 
	try:
		import unicodedata
		unicodedata.numeric(token)
		return True
	except (TypeError, ValueError):
		return False



def isIdentifier(token, identifiers):
	return (True if token in identifiers else False)


def handleUnknown(token): 
	splitString = ' '.join([c for c in token])
	return splitString


def normalizeFile(csvFile, cDir, allNodeFile):
	identifiers = list()
	with open(csvFile, 'r') as cf:
		lines = list()
		csvReader = csv.DictReader(cf, delimiter='\t')
		for line in csvReader:
		#Scan the nodes to extract all identifiers
			lines.append(line)
			if line['type'] == 'Identifier':
			# We DO NOT care the different between variables and function names for now
				identifiers.append(line['code'])

		identifiers = set(identifiers)
		normCsv = os.path.join(cDir, 'nodes_normalized.csv')
		with open(normCsv, 'w') as wf:
			for line in lines:
				#Handle Joern keywords differently
				if line['type'] == 'CFGEntryNode':
					lcode = 'CFG-ENTRY'
				elif line['type'] == 'CFGExitNode':
					lcode = 'CFG-EXIT'
				elif line['type'] == 'CFGErrorNode':
					lcode = 'CFG-ERROR'
				else: 
					lcode = line['code']
				#keep same with Sahil's output
				if line['type'] != 'File':
					wf.write(' '.join([line['key'], line['type'], normalize(lcode, identifiers)]))
					allNodeFile.write(' '.join([line['key'], line['type'], normalize(lcode, identifiers)]))
					wf.write('\n')
					allNodeFile.write('\n')


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('repo', help='Node.csv to process')
	args = parser.parse_args()
	cmd = ""
	#cmd += "cd " + args.repo + ";"
	cmd += "find " + args.repo +" -name *nodes.csv;"
	result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=True)
	result = result.stdout.decode('utf-8')
	result = result.split("\n")
	result = list(filter(None, result))
	with open('nodes_normalized_all.csv', 'w') as af:
		for csvFile in result:
			cDir = os.path.dirname(csvFile)
			normalizeFile(csvFile, cDir, af)

	

if __name__ == '__main__':
	main()
	


import os
import random
import gensim
#import json
import simplejson as json
import itertools
import numpy as np

#tokenizer=__import__('4-tokenizer')
#word2vec=__import__('5-word2vec')
#word2vec.main()
#print(tokenizer.num_funcs)

#draper
edge_types = ['IS_FILE_OF', 'CONTROLS', 'DECLARES', 'DEF', 'DOM', 'FLOWS_TO', 'IS_AST_PARENT', 'IS_CLASS_OF', 'IS_FUNCTION_OF_AST', 'IS_FUNCTION_OF_CFG', 'POST_DOM', 'REACHES', 'USE']  

#julier, sbabi
#edge_types = ['IS_FILE_OF', 'CONTROLS', 'DEF', 'DOM', 'FLOWS_TO', 'IS_AST_PARENT', 'IS_FUNCTION_OF_AST', 'IS_FUNCTION_OF_CFG', 'POST_DOM', 'REACHES', 'USE' ]

def get_label(func, labels):
    if len(labels) == 1:    #sbabi specific
        return labels[0]
    func_id = int(func.strip('.c'))
    offset = func_id - 1
    return int(labels[offset])

def create_labels(labels_dir):  #sbabi specific
    if 'nonvuln' in labels_dir:
        labels = [0]
    elif 'vuln' in labels_dir:
        labels = [1]
    else:
        raise Exception('unexpected dir name')
    return labels

def read_labels_file(labels_dir, labels_file):
    if not os.path.exists(labels_dir+'/'+labels_file):
        return create_labels(labels_dir)
    fd = open(labels_dir + '/' + labels_file)
    labels = fd.read().splitlines()
    return labels

def read_skip_list(filename):
   fd = open(filename,'r') 
   skip_list = fd.read().splitlines()
   return skip_list 


def get_edges(edge_file):
    fd = open(edge_file,'r')
    edges = fd.readlines()
    fd.close()
    return edges

def get_non_leaf_nodes(edges):
    non_leaf_nodes = []
    for edge in edges:
        nodes = edge.split()
        if nodes[2] == 'IS_AST_PARENT':
            non_leaf_nodes.append(nodes[0])
    return non_leaf_nodes


def get_avg_vec_org(tokens, w2v):
    vector = np.zeros(w2v.vector_size)
    num_tokens = 0
    try:
        for token in tokens:
            vector = vector + w2v[token]
            num_tokens = num_tokens + 1
    except:
        import pdb
        pdb.set_trace()
    return (vector/num_tokens).tolist()  #[node_vector] == [1,2,3]

def get_avg_vec(tokens, w2v):
    vector = np.zeros(w2v.vector_size)
    num_tokens = 0
    try:
        for token in tokens:
            vector = vector + get_w2v(token, w2v)
            num_tokens = num_tokens + 1
    except:
        import pdb
        pdb.set_trace()
    return (vector/num_tokens).tolist()  #[node_vector] == [1,2,3]


def get_concat_vec(tokens, w2v, max_num_tokens=4):  #max num of tokens to concat
    try:
        if len(tokens) > max_num_tokens:
            #print(tokens)
            #vectors = [w2v[token] for token in tokens[0:max_num_tokens-1]]  #lose information, leave last token dimension empty 
            vectors = [get_w2v(token, w2v) for token in tokens[0:max_num_tokens-1]]  #lose information, leave last token dimension empty 
        else:
            #vectors = [w2v[token] for token in tokens[0:max_num_tokens]]
            vectors = [get_w2v(token, w2v) for token in tokens[0:max_num_tokens]]
    except:
        import pdb
        pdb.set_trace()
    return np.concatenate(vectors)


def pad_node_vector(v, pad_len):
    pad_len = max(0, pad_len - len(v))
    v = np.pad(v,(0,pad_len),'constant')    #v will become a 128 wide vector padded with 0s if required
    return v.tolist() #numpy -> list

token_mapping = {
                    'size_t':'u64',
                    'int64_t':'long',
                }

def get_w2v(token, w2v):
    try:
        return w2v[token]
    except:
        if token in token_mapping:  #when doing cross-domain w2v mapping like juliet -> draper or openssl -> draper
            return w2v[token_mapping[token]]
        else:    
            print(token, "not in vocab")
            import pdb
            pdb.set_trace()
       

def get_node_vectors_concat(node_file, non_leaf_nodes, w2v):
    max_vec_len = 128
    token_len = 32
    node_vectors = []
    fd = open(node_file,'r')
    nodes = fd.readlines()
    for node in nodes:
        node_tokens = node.split()  #tokens = [node_id, node_type, node_code, node_code, .... ]
        node_id = node_tokens[0]
        if node_id in non_leaf_nodes:   #use node_type alone as vector
            #node_vector = w2v[node_tokens[1]]
            node_vector = get_w2v(node_tokens[1], w2v)
        else: #concat word2vec of node_type and tokens in 'code' column    
            node_vector = get_concat_vec(node_tokens[1:], w2v, max_num_tokens=int(max_vec_len/token_len))
        node_vector = pad_node_vector(node_vector, max_vec_len)
        node_vectors.append(node_vector)
    fd.close()
    return node_vectors #[[node_0_vector],... , [node_n-1_vector]] 


def get_node_vectors(node_file, non_leaf_nodes, w2v):   #w2v avg
    node_vectors = []
    fd = open(node_file,'r')
    nodes = fd.readlines()
    for node in nodes:
        node_tokens = node.split()  #tokens = [node_id, node_type, node_code, node_code, .... ]
        node_id = node_tokens[0]
        if node_id in non_leaf_nodes:   #use node_type alone as vector
            node_vector = get_w2v(node_tokens[1], w2v).tolist() #numpy -> list
        else: #avg word2vec of node_type and tokens in 'code' column    
            node_vector = get_avg_vec(node_tokens[1:], w2v)
        node_vectors.append(node_vector)
    fd.close()
    return node_vectors #[[node_0_vector],... , [node_n-1_vector]] 


def get_node_vectors_org(node_file, non_leaf_nodes, w2v):   #w2v avg
    node_vectors = []
    fd = open(node_file,'r')
    nodes = fd.readlines()
    for node in nodes:
        node_tokens = node.split()  #tokens = [node_id, node_type, node_code, node_code, .... ]
        node_id = node_tokens[0]
        if node_id in non_leaf_nodes:   #use node_type alone as vector
            node_vector = w2v[node_tokens[1]].tolist() #numpy -> list
        else: #avg word2vec of node_type and tokens in 'code' column    
            node_vector = get_avg_vec(node_tokens[1:], w2v)
        node_vectors.append(node_vector)
    fd.close()
    return node_vectors #[[node_0_vector],... , [node_n-1_vector]] 


def get_edge_types(_edges):
    global edge_types
    edges = []
    for edge in _edges[1:]:     #skip header line
        edge_tokens = edge.split()
        u = edge_tokens[0]
        v = edge_tokens[1]
        e = edge_tokens[2]
        if e == 'IS_FILE_OF':   #skip edge_type IS_FILE_OF
            continue
        edges.append([int(u)-2, edge_types.index(e), int(v)-2])  
        #subtract 2 from node_id to start from id 0
        #1 coz node_id starts from 1, and another 1 coz IS_FILE_OF is node 1 which is skipped
    return edges    # [[u_node_id, edge_type_id, v_node_id], ... ],


def convert_cpg_to_vector(func_dir, func, node_file, edge_file, w2v, labels):
    global do_select_samples
    label = get_label(func, labels)
    if do_select_samples == True:
        if label != 0:
            return None

    edges = get_edges(edge_file)
    if len(edges) <= 30: #0 for sbabi  #weeding out cases where joern incompletely parses funcs 
       return None

    non_leaf_nodes = get_non_leaf_nodes(edges)
    node_vectors = get_node_vectors(node_file,non_leaf_nodes, w2v_model)
    edge_vectors = get_edge_types(edges)
    cpg_vector_dict = { "graph": edge_vectors, 
                        "targets": [[label]],
                        "node_features": node_vectors,
                        "func": func,
                        "src": func_dir
                      }
    return cpg_vector_dict   #{"graph": [[u_node_id, edge_type_id, v_node_id], ... ], "targets": [[is_vuln)]], "node_features": [[node_0_vector],... , [node_n-1_vector]]}


def convert_cpgs_to_vectors(joern_dir, func_dir, skip_list, labels, w2v):
    #cpg_vector_dicts = []
    for func in os.listdir(joern_dir+'/'+func_dir):
        #print(func)
        if func_dir+'/'+func in skip_list:
            continue
        node_file = joern_dir+'/'+func_dir+'/'+func+'/'+func+'/'+func+'/nodes_normalized.csv'
        edge_file = joern_dir+'/'+func_dir+'/'+func+'/'+func+'/'+func+'/edges.csv'
        cvd = convert_cpg_to_vector(func_dir, func, node_file, edge_file, w2v, labels)
        if cvd:
            yield cvd
        #cpg_vector_dicts.append(convert_cpg_to_vector(func, node_file, edge_file, w2v))
    #return cpg_vector_dicts


def write_cpg_vectors(dict_obj, filename):
    fd = open(filename,'w')
    json.dump(dict_obj,fd, iterable_as_array=True)  #read as json.load(fd)
    fd.close()

def write_cpg_vectors_linebyline(dict_objs, filename):
    fd = open(filename,'w')
    for dict_obj in dict_objs:
        json.dump(dict_obj,fd)
        fd.write('\n')
    fd.close()

def verify_cpg_vectors(cpg_vector_dicts, joern_dir, func_dir, skip_list, labels, w2v):
    offset = 0
    func = 'none.c'

    while True:
        offset = random.randrange(0,100)
        func = os.listdir(joern_dir+'/'+func_dir)[offset]
        if func_dir+'/'+func not in skip_list:
            edge_file = joern_dir+'/'+func_dir+'/'+func+'/'+func+'/'+func+'/edges.csv'
            fd = open(edge_file,'r')
            edges = fd.read().splitlines()
            fd.close()
            if len(edges) <= 30:
               continue 
            print(func)
            break

    for cvd in cpg_vector_dicts:
        if cvd['func'] == func:
            break;
    #cvd =  next(itertools.islice(cpg_vector_dicts,offset,None))    
    #cvd = cpg_vector_dicts[offset]
    #len_cvd = sum(1 for _ in cpg_vector_dicts)

    node_file = joern_dir+'/'+func_dir+'/'+func+'/'+func+'/'+func+'/nodes_normalized.csv'
    edge_file = joern_dir+'/'+func_dir+'/'+func+'/'+func+'/'+func+'/edges.csv'
    
    assert get_label(func, labels) == cvd["targets"][0][0]
    
    fd = open(edge_file,'r')
    edges = fd.read().splitlines()
    fd.close()
    cvd_edges = cvd["graph"]
    assert len(edges)-2 == len(cvd_edges)
    
    offset = random.randrange(0,len(cvd_edges))
    edge_tokens = edges[offset + 1].split()
    u = edge_tokens[0]
    v = edge_tokens[1]
    e = edge_tokens[2]
    assert [int(u)-2, edge_types.index(e), int(v)-2] == cvd_edges[offset] 

    fd = open(node_file,'r')
    nodes = fd.read().splitlines()
    fd.close()
    cvd_nodes = cvd["node_features"]
    assert len(nodes) == len(cvd_nodes)

    offset = random.randrange(0,len(cvd_nodes))
    non_leaf_nodes = get_non_leaf_nodes(edges)
    node_tokens = nodes[offset].split()  #tokens = [node_id, node_type, node_code, node_code, .... ]
    node_id = node_tokens[0]
    if node_id in non_leaf_nodes:   #use node_type alone as vector
        node_vector = w2v[node_tokens[1]].tolist()
    else: #avg word2vec of node_type and tokens in 'code' column    
        node_vector = get_avg_vec(node_tokens[1:], w2v)
    assert node_vector == cvd_nodes[offset]








#cpg_vector_filename = 'cpg_vector_validate'
#cpg_vector_filename = '/data/validate_plus_train_label1/cpg_vector_validate_linebyline'
#cpg_vector_filename = '/data/validate_plus_train_label1/cpg_vector_train_label1_linebyline'
#w2v_model_filename = '/mnt/m1/cpg_vtl119_v0_tl0_400k/cpg_w2v_100/word2vec_model_validate_plus_train_label1_plus_train_label0_400k_32wide'
#cpg_vector_filename = '/mnt/m1/openssl/cpg_avg_w2v_draper/cpg_vector_openssl'
#w2v_model_filename = '/mnt/m1/cpg_validate_plus_train_label1/word2vec_model_validate_plus_train_label1_32wide_save'
cpg_vector_filename = '/mnt/m1/sbabi/cpg_vector_train'
w2v_model_filename = '/mnt/m1/sbabi/word2vec_model_sbabi_32wide'

labels_dir = '/mnt/m1/sbabi/sa-bAbI/data/train'
labels_file = 'labels-CWE-any'
joern_dir='/mnt/m1/joern_sbabi'
func_dir='functions_sbabi/train'
do_select_samples = False


#skip_list = read_skip_list('skip_list')
#skip_list = read_skip_list('skip_list_openssl')
skip_list = read_skip_list('skip_list_sbabi')
labels = read_labels_file(labels_dir, labels_file)
w2v_model = gensim.models.Word2Vec.load(w2v_model_filename)

cpg_vector_dicts = convert_cpgs_to_vectors(joern_dir, func_dir, skip_list, labels, w2v_model)
#cpg_vector_dicts, cpg_vector_dicts_copy = itertools.tee(cpg_vector_dicts) #clone generator
#verify_cpg_vectors(cpg_vector_dicts_copy,joern_dir, func_dir, skip_list, labels, w2v_model)
#above works but uses a lot of memory when combined with below
#write_cpg_vectors(cpg_vector_dicts, cpg_vector_filename)
write_cpg_vectors_linebyline(cpg_vector_dicts, cpg_vector_filename)


#for each file
    #read node and edge list
    #for each node
        #FOR THE CASE WHERE NODE-TYPE IS ALSO PART OF WORD2VEC VOCAB
            #if node features as IS_AST_PARENT in the edge list:
                #convert node to vector as the tuple (node_type's word2vec)
            #else #==LEAF_NODE
                #convert node to vector as the tuple (avg word2vec of node_type and tokens in 'code' column in node_list 
        #FOR THE CASE WHERE NODE-TYPE IS NOT PART OF WORD2VEC VOCAB
            #if node features as IS_AST_PARENT in the edge list:
                #convert node to vector as the tuple (one_hot_node_type, EMPTY word2vec)
            #else #==LEAF NODE
                #convert node to vector as the tuple (one_hot_node_type, avg word2vec of 'code' columnin node list)

    #read vuln label for file as 0/1 is_vuln
    #subtract 1 from node_id to start from id 0
    #add graph as dict item {"graph": [[u_node_id, edge_type_id, v_node_id], ... ], "targets": [[is_vuln)]], "node_features": [[node_0_vector],... , [node_n-1_vector]]}



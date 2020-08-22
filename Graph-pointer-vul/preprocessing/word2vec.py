import gensim
from gensim.models.word2vec import Word2Vec

with open ("/Users/yding/Documents/projects/Devign_data/Devign_loc_both/w2vInput_del_single", 'r') as tf:
	sentences = tf.readlines()
	tok_sentences = [stc.split() for stc in sentences]
	model = Word2Vec(sentences=tok_sentences, min_count=1, size=32, workers=16, iter=10)
	model.save("/Users/yding/Documents/projects/Devign_data/Devign_loc_single_del_only/devign_del_only_nonNorm")
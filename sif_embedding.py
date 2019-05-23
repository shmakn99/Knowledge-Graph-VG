import glove_util as gut 
import numpy as np 
from sklearn.decomposition import TruncatedSVD
import json
import numpy as np
import pickle


with open('freq_count_pred.json') as f:
	freq_count_pred = json.load(f)

def get_pc(sentences):
	svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
	svd.fit(sentences)
	return svd.components_


def weighted_avg(predicate,a,dim):

	predicate = predicate.lower().strip().split()

	if len(predicate) == 1:
		return gut.glove(predicate[0],dim)
	else:
		support = np.zeros(dim)
		for word in predicate:
			vector = gut.glove(word,dim)
			if len(vector) == 0:
				vector = np.zeros(dim)
			support += (a/(a+freq_count_pred[word]))*vector

		return support


dim = 50
adj = np.zeros((128,128,dim))

with open('KG_128.pkl') as f:
	KG_128 = pickle.load(f)

with open('classes_128.pkl') as f:
	classes = pickle.load(f)


print (len(KG_128.edges()))

sentences = []
predicate_embedding={}

i = 0
for ed in KG_128.edges():
	print (i)
	i+=1
	w_avg = weighted_avg(KG_128[ed[0]][ed[1]]['predicate'],0.001,dim)
	sentences.append(w_avg)
	predicate_embedding[ed[0]+ed[1]] = w_avg 

pc = get_pc(np.array(sentences))[0]
projection_space = np.outer(pc,pc)

i = 0
for ed in KG_128.edges():
	pe = predicate_embedding[ed[0]+ed[1]] 

	predicate_embedding[ed[0]+ed[1]] = pe  - np.matmul(projection_space,pe)
	adj[classes.index(ed[0]),classes.index(ed[0])] = predicate_embedding[ed[0]+ed[1]]

with open('adj_128_'+str(dim)+'.pkl','wb') as f:
	pickle.dump(adj,f)
		


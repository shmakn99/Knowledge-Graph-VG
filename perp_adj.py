import glove_util as gut 
import numpy as np 
from sklearn.decomposition import TruncatedSVD
import json

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
				vector = np.zeros(300)
			support += (a/(a+freq_count_pred[word]))*vector

		return support




with open('relationships.json') as f:
	relationships = json.load(f)

predicate_embedding = {}

sentences = []
i = 0
for image in relationships:
	i+=1
	if i%1000 == 0:
		print (i)

	for relation in image['relationships']:
		w_avg = weighted_avg(relation['predicate'],0.001,300)
		sentences.append(w_avg)
		predicate_embedding[relation['relationship_id']] = w_avg 

pc = get_pc(np.array(sentences))[0]
projection_space = np.outer(pc,pc)

i = 0
for image in relationships:
	i+=1
	if i%1000 == 0:
		print (i)

	for relation in image['relationships']:
		predicate_embedding[relation['relationship_id']] = predicate_embedding[relation['relationship_id']]  - np.matmul(projection_space,predicate_embedding[relation['relationship_id']])
		

with open('predicate_embedding_300.json','w') as f:
	json.dump(predicate_embedding,f)

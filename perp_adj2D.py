import glove_util as gut 
import numpy as np 
from sklearn.decomposition import TruncatedSVD
import json
import pickle
import networkx as nx

with open('KG_128.pkl') as f:
	KG_128 = pickle.load(f)

with open('classes_128.pkl') as f:
	classes = pickle.load(f)

adj = np.zeros((128,128))

for ed in KG_128.edges():
	adj[classes.index(ed[0]),classes.index(ed[1])] = KG_128[ed[0]][ed[1]]['weight']
	adj[classes.index(ed[1]),classes.index(ed[0])] = KG_128[ed[1]][ed[0]]['weight']

with open('adj_128_2D_weighted.pkl','wb') as f:
	pickle.dump(adj,f)	
	

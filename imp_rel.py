import json
import numpy as np
import pickle
import networkx as nx

G = nx.Graph()

with open('KG_128.pkl') as f:
	G = pickle.load(f)

nd_list = list(G.nodes())

with open('classes_128.pkl','wb') as f:
	pickle.dump(nd_list,f)

def xyz():

	objects =  list(G.nodes())

	with open('relationships.json') as f:
		relationships = json.load(f)


	i=0 


	images = []



	for img in relationships:
		if i%1000 == 0:
			print (i)
		i+=1

		for rlshn in img['relationships']:
			if 'name' in rlshn['object'].keys():
				name  = rlshn['object']['name']
			else:
				name = rlshn['object']['names'][0] 
			if name in objects:
				images.append(img['image_id'])		
		

	images = list(set(images))
	print (len(set(images)))

	with open('images_128.pkl','wb') as f:
		pickle.dump(images,f)




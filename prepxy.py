import pickle
import json
import numpy as np

with open('images_128.pkl') as f:
	images = pickle.load(f)

with open('classes_128.pkl') as f:
	classes = pickle.load(f)

with open('relationships.json') as f:
	relationships = json.load(f)

def get_y(classes,rlshnps):
	
	ret = np.zeros(128)
	for rlshn in rlshnps:
		if 'name' in rlshn['object'].keys():
			nameo = rlshn['object']['name']
		else:
			nameo = rlshn['object']['names'][0]

		if 'name' in rlshn['subject'].keys():
			names = rlshn['subject']['name']
		else:
			names = rlshn['subject']['names'][0]

		if nameo in classes:
			ret[classes.index(nameo)] = 1

		if names in classes:
			ret[classes.index(names)] = 1

	return ret


i = 0

y = []

for img in relationships:
	if i%1000 == 0:
		print (i)
	i+=1

	
	if img['image_id'] in images:
	
		y.append(get_y(classes, img['relationships']))
		
		


print (len(y))

with open('y_128_full.pkl','w') as f:
	pickle.dump(y,f)

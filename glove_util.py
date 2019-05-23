from scipy import spatial
import numpy as np
import time

def cosimi(u,v):
	# print ('def cosimi(u,v):')
	return np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))

def diff(u,v):
	print('def diff(u,v):')
	return np.linalg.norm(u-v)

def glove(w,dim):
	# print ('def glove(w):')
	
	raw_vec = ''

	with open('glove.6B/glove.6B.'+str(dim)+'d.txt') as f:
		for line in f:
			if str(w+' ') == line[:len(w)+1]:
				raw_vec = line
				break 

	vec = [float(_) for _ in raw_vec.split()[1:]]
	return np.array(vec)

def nearest_glove(vec):
	# print ('def nearest_glove(vec):')
	guess = ''

	high_simi = -10

	with open('/home/siemens/Documents/glove.6B/glove.6B.300d.txt') as f:
		for line in f:
			
			u = [float(_) for _ in line.split()[1:]]
			
			# print (guess)

			if cosimi(u,vec) > high_simi: 
				guess = line.split()[0]
				high_simi= cosimi(u,vec)

			if high_simi >= 0.8:
				break

	return guess




# vec = glove('gym') + glove('bag')

# print (nearest_glove(vec))

# print (cosimi(glove('car'),glove('train')))

# a = []
# a.append(np.array([0,1,1]))
# a.append(np.array([1,2,1]))

# print (a)

# print (np.sum(a))
# print([1,2,3,4][-1:])

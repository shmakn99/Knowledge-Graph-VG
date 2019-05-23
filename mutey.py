import random as rd
import pickle
import numpy as np



def mute(y):

	to_m_frm = [i for i in range(128) if y[i]==1]
	
	y[rd.sample(to_m_frm,1)[0]] = 0

	return y	
	

with open('y_128_full.pkl') as f:
	y= pickle.load(f)

x = [mute(i) for i in y]

print (len(x))

with open('x_128_full.pkl','wb') as f:
	pickle.dump(x,f)

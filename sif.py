import glove_util as gut 
import numpy as np 
from sklearn.decomposition import TruncatedSVD
import json
import numpy as np
import pickle



with open('glove_128.pkl') as f:
	g= pickle.load(f)

print (g[46])

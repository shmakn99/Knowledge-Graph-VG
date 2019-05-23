import json

def update(freq_count, aggregate):
	aggregate = aggregate.lower().strip().split()

	for word in aggregate:
		if word not in freq_count.keys():

			freq_count[word] = 1
		else:
			freq_count[word] += 1
		

	return freq_count
		

with open('freq_count_pred.json') as f:
	data = json.load(f)

print (data)



import json
import pickle
import sys
from processing import *

with open(sys.argv[1], 'r') as f:
    data = json.load(f)
    # data = make_names_dict(data)


unambiguous_cache = {}

for article in data:
    unambiguous_cache[article['articleID']] = []
    for name in article['names']:
        if len(name['ids']) == 1:
            unambiguous_cache[article['articleID']].append(name['ids'])


with open(sys.argv[2], 'wb') as f:
    pickle.dump(unambiguous_cache, f)

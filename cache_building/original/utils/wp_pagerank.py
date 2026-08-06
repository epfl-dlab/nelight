import numpy as np
import json
import pickle
from tqdm import tqdm
import os
print(os.getcwd())

with open('/dlabdata1/culjak/danker/wikipedia_pagerank.ranks', 'r') as f:
    f.readline()


    with open('../caches/ultimate_wikicache.pkl', 'rb') as c:
        cache = pickle.load(c)

    for line in tqdm(f):
        qid, rank = str(line).strip().split('\t')
        qid = 'Q' + qid
        if qid in cache:
            rank = float(rank)
            cache[qid]['pagerank'] = rank

with open('../caches/ultimate_wikicache.pkl', 'wb') as f:
    pickle.dump(cache, f)


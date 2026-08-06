import gzip
import pickle
import json
import sys
import numpy as np
from tqdm import tqdm

NP = np.zeros(int(sys.argv[1]))
NS = np.zeros(int(sys.argv[1]))



with gzip.open('/dlabdata1/wiki_embedding_project/Data/raw/Knowledge_Graph/20211101/wikidata-20211101-all.json.gz', 'rb') as f:
    for i, instance in tqdm(enumerate(f)):
        instance = instance.decode('utf-8')
        instance = instance[:-2]
        if len(instance) == 0:
            continue
        try:
            instance = json.loads(instance)
        except json.decoder.JSONDecodeError:
            print(instance)
            continue

        if instance['id'][0] != 'Q':
            continue
        qid = int(instance['id'][1:])
        NP[qid] = len(instance['claims'])
        NS[qid] = len(instance['sitelinks']) if 'sitelinks' in instance else 0

        if i % 1000000 == 0:
            np.savez('/dlabdata1/culjak/NPNS.npz', NP=NP, NS=NS)

        
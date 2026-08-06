import gzip
import pickle
import json
import sys
import numpy as np
from tqdm import tqdm

with open('/dlabdata1/culjak/qids_for_cache.pkl', 'rb') as f:
    qids = pickle.load(f)

highest = max(qids, key=lambda x: int(x[1:]))
cache = {}
total_qids = len(qids)
print(highest)
with gzip.open('/dlabdata1/wiki_embedding_project/Data/raw/Knowledge_Graph/20211101/wikidata-20211101-all.json.gz', 'rb') as f:
    pbar = tqdm(enumerate(f))
    for i, instance in pbar:
        pbar.set_postfix({'QIDs processed': 1 - len(qids) / total_qids})
        instance = instance.decode('utf-8')
        instance = instance[:-2]
        if len(instance) == 0:
            continue
        try:
            instance = json.loads(instance)
        except json.decoder.JSONDecodeError:
            print(instance)
            continue

        qid = instance['id']
        if qid not in qids:
            continue
        qids.remove(qid)
        qid = int(qid[1:])

        cache[qid] = {}

        for prop, value in instance['claims'].items():
            prop = int(prop[1:])
            cache[qid][prop] = []
            for snak in value:
                mainsnak = snak['mainsnak']
                if 'datavalue' not in mainsnak:
                    continue
                datatype = mainsnak['datatype']
                if datatype == 'wikibase-item':
                    if 'value' not in mainsnak['datavalue']:
                        continue
                    datavalue = int(mainsnak['datavalue']['value']['id'][1:])
                    cache[qid][prop].append(datavalue)
                if datatype == 'string':
                    if 'value' not in mainsnak['datavalue']:
                        continue
                    datavalue = mainsnak['datavalue']['value']
                    cache[qid][prop].append(datavalue)

            if len(cache[qid][prop]) == 0:
                del cache[qid][prop]

        if len(qids) == 0:
            break

with open('/dlabdata1/culjak/qb_cache_all.pkl', 'wb') as f:
    pickle.dump(cache, f)





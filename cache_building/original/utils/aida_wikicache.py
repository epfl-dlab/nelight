import gzip
import pickle
import json

from tqdm import tqdm

qid_to_label = {}
error_labels = set()

batch_idx = 0
processed = 0

with gzip.open('/dlabdata1/wiki_embedding_project/Data/raw/Knowledge_Graph/20211101/wikidata-20211101-all.json.gz', 'rb') as f:
    for instance in tqdm(f):
        processed += 1
        if processed < 90000001:
            continue

        instance = instance.decode('utf-8')
        instance = instance[:-2]
        if len(instance) == 0:
            continue
        try:
            instance = json.loads(instance)
        except json.decoder.JSONDecodeError:
            print(instance)
            continue
        try:
            qid = int(instance['id'][1:])
            labels = instance['labels']
            if 'en' not in labels:
                error_labels.append(qid)
            else:
                qid_to_label[qid] = labels['en']['value']

        except Exception as e:
            error_labels.add(qid)

with open(f'/dlabdata1/culjak/wikidata_labels/qtl_batch_9', 'wb') as f:
    pickle.dump(qid_to_label, f)
    qid_to_label = {}

print(error_labels)

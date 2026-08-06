import gzip
import pickle
import json
import sys

from tqdm import tqdm

qid_to_label = {}
qid_to_alias = {}
qid_to_desc = {}

errors_label = set()
errors_alias = set()
errors_desc = set()

batch_idx = 0
processed = 0

with gzip.open('/dlabdata1/wiki_embedding_project/Data/raw/Knowledge_Graph/20211101/wikidata-20211101-all.json.gz', 'rb') as f:
    for instance in tqdm(f):
        processed += 1
        instance = instance.decode('utf-8')
        instance = instance[:-2]

        if len(instance) == 0:
            continue
        try:
            instance = json.loads(instance)
        except json.decoder.JSONDecodeError:
            continue
        qid = instance['id']
        if qid[0] != 'Q':
            continue
        qid = int(qid[1:])
        try:
            aliases = instance['aliases']
            if 'en' not in aliases:
                errors_alias.add(qid)
            else:
                qid_to_alias[qid] = [alias['value'] for alias in aliases['en']]
                errors_alias.add(qid)
        except Exception as e:
            errors_label.add(qid)
        try:
            labels = instance['labels']
            if 'en' not in labels:
                errors_label.add(qid)
            else:
                qid_to_label[qid] = labels['en']['value']
        except Exception as e:
            errors_label.add(qid)
        try:
            descriptions = instance['descriptions']
            if 'en' not in descriptions:
                errors_desc.add(qid)
            else:
                qid_to_desc[qid] = descriptions['en']['value']
        except Exception as e:
            errors_desc.add(qid)

        if processed % 10000000 == 0:
            batch_idx = processed // 10000000 - 1
            with open(f'/dlabdata1/culjak/wikidata_labels/qta_batch_{batch_idx}', 'wb') as f:
                pickle.dump(qid_to_alias, f)
                qid_to_alias = {}

            with open(f'/dlabdata1/culjak/wikidata_labels/qtl_batch_{batch_idx}', 'wb') as f:
                pickle.dump(qid_to_label, f)
                qid_to_label = {}

            with open(f'/dlabdata1/culjak/wikidata_labels/qtd_batch_{batch_idx}', 'wb') as f:
                pickle.dump(qid_to_desc, f)
                qid_to_desc = {}


batch_idx = processed // 10000000

with open(f'/dlabdata1/culjak/wikidata_labels/qta_batch_{batch_idx}', 'wb') as f:
    pickle.dump(qid_to_alias, f)
    del qid_to_alias
with open(f'/dlabdata1/culjak/wikidata_labels/qtl_batch_{batch_idx}', 'wb') as f:
    pickle.dump(qid_to_label, f)
    del qid_to_label
with open(f'/dlabdata1/culjak/wikidata_labels/qtd_batch_{batch_idx}', 'wb') as f:
    pickle.dump(qid_to_desc, f)
    del qid_to_desc

with open(f'/dlabdata1/culjak/wikidata_labels/errors_desc', 'wb') as f:
    pickle.dump(errors_desc, f)
    del errors_desc
with open(f'/dlabdata1/culjak/wikidata_labels/errors_alias', 'wb') as f:
    pickle.dump(errors_alias, f)
    del errors_alias
with open(f'/dlabdata1/culjak/wikidata_labels/errors_label', 'wb') as f:
    pickle.dump(errors_label, f)
    del errors_label

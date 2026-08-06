import gzip
import pickle
import json
from tqdm import tqdm

with open('/dlabdata1/culjak/quotebank_qids.pkl', 'rb') as f:
    qids = pickle.load(f)

cache = {}
total_qids = len(qids)

prop_label = {
    'P26': 'spouse',
    'P793': 'significant_event',
    'P570': 'date_of_death',
    'P451': 'unmarried_partner',
    'P102': 'party'
}

counts = dict(zip(prop_label.keys(), [0] * len(prop_label)))


with gzip.open('/dlabdata1/wiki_embedding_project/Data/raw/Knowledge_Graph/20211101/wikidata-20211101-all.json.gz', 'rb') as f:
    lines = []
    with gzip.open('/dlabdata1/culjak/quotebank_wikidata_subgraph.json.gz', 'wb') as f_out:
        pbar = tqdm(enumerate(f))
        for i, instance in pbar:
            pbar.set_postfix({'QIDs processed': f'{(1 - len(qids) / total_qids) * 100:.2f}%'})
            instance_decoded = instance.decode('utf-8')
            instance_decoded = instance_decoded[:-2]
            if len(instance_decoded) == 0:
                continue
            try:
                instance_decoded = json.loads(instance_decoded)
            except json.decoder.JSONDecodeError:
                print(instance_decoded)
                continue

            qid = instance_decoded['id']
            if qid not in qids:
                continue
            qids.remove(qid)
            lines.append(instance)

            if len(qids) == 0:
                break
        f_out.writelines(lines)





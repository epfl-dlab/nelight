import pickle
import re
import bz2
import json
from tqdm import tqdm

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

aida_cache = load_pickle('/dlabdata1/culjak/speaker-disambiguation-quotebank/caches/aida_cache_val.pkl')

qid_pid = {}

with bz2.open('/dlabdata1/wiki_embedding_project/Data/preprocessing/matchings/Qid_pid/from_page_props/20220101/en/mapping.json.bz2', 'rb') as f:
    for i in tqdm(f):
        i = json.loads(i)
        qid = i['Qid']
        if qid not in aida_cache:
            continue
        pid = i['page_id']
        qid_pid[int(pid)] = qid

with bz2.open('/dlabdata1/wiki_embedding_project/Data/preprocessing/first_paragraphs/20220101/en/first_paragraphs.jsonl.bz2', 'rb') as f:
    for i in tqdm(f):
        i = json.loads(i)
        pid = i['page_id']
        fp = i['first_paragraph']
        if pid not in qid_pid:
            continue
        aida_cache[qid_pid[pid]]['first_paragraph'] = fp

save_pickle(aida_cache, '/dlabdata1/culjak/aida/cache_val_fp.pkl')
#


# first_paragraphs = load_pickle(

#
# print()



# for qid, data in aida_cache.items():
#     if


#
#
# lbatches = [load_pickle(f'/dlabdata1/culjak/wikidata_labels/qtl_batch_{i}') for i in range(10)]
# dbatches = [load_pickle(f'/dlabdata1/culjak/wikidata_labels/qtd_batch_{i}') for i in range(10)]
#
# qtl = {k: v for batch in lbatches for k, v in batch.items()}
# qtd = {k: v for batch in dbatches for k, v in batch.items()}
#
# aida_cache = load_pickle('../caches/aida_cache2.pkl')
# qtoi = lambda x: int(x[1:])
#
# for i, j in aida_cache.items():
#     for k, l in j.items():
#         aida_cache[i][k] = []
#         for q in l:
#             if re.match('Q[1-9][0-9]*', q):
#                 if qtoi(q) in qtl:
#                     aida_cache[i][k].append(qtl[qtoi(q)])
#                 # if qtoi(q) in qta :
#                 #     aida_cache[i][k].extend(qta[qtoi(q)])
#             else:
#                 aida_cache[i][k].append(q)
#
#         if len(aida_cache[i][k]) == 0:
#             aida_cache[i][k] = ['']
#     aida_cache[i]['description'] = [qtd[qtoi(i)] if qtoi(i) in qtd else '']
#
# save_pickle(aida_cache, '../caches/aida_cache2_p.pkl')

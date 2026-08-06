import pickle
import re
from tqdm import tqdm

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

# lbatches = [load_pickle(f'/dlabdata1/culjak/wikidata_labels/qtl_batch_{i}') for i in range(10)]
dbatch = load_pickle(f'/dlabdata1/culjak/wikidata_labels/qtd_batch_0')


# dbatches = [load_pickle(f'/dlabdata1/culjak/wikidata_labels/qtd_batch_{i}') for i in range(10)]

# qtl = {k: v for batch in lbatches for k, v in batch.items()}
# qtd = {k: v for batch in dbatches for k, v in batch.items()}
i = 0
for k, l in dbatch.items():
    try:
        print(k, l)
        i += 1
    except:
        continue
    if i > 10:
        break

# print(len(qtl))
# print(len(qtd))
# print(qtd[207])
# aida_cache = load_pickle('/dlabdata1/culjak/qb_cache_all.pkl')
qtoi = lambda x: x#int(x[1:])

# for i, j in tqdm(aida_cache.items()):
#     for k, l in j.items():
#         aida_cache[i][k] = []
#         for q in l:
#             if isinstance(q, int):
#                 if q in qtl:
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
# save_pickle(aida_cache, '/dlabdata1/culjak/qb_cache_processed.pkl')

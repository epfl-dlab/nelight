import pickle

import torch
from transformers import BartTokenizer, BartModel
from tqdm import tqdm

if __name__ == '__main__':
    print('Loading model...')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartModel.from_pretrained('facebook/bart-base').to('cuda:1').eval()
    print('Loaded model!')

    with open('/dlabdata1/culjak/aida/aida_cache_all.pkl', 'rb') as f:
        ultimate_wikicache = pickle.load(f)

    embedding_wikicache = {}

    for qid, prop_dict in tqdm(ultimate_wikicache.items()):
        embedding_wikicache[qid] = {}

        for prop, value_list in prop_dict.items():
            if prop not in ('n_sitelinks', 'pagerank', 'n_statements'):
                with torch.no_grad():
                    if len(value_list) > 0:
                        inputs = tokenizer(value_list, return_tensors='pt', truncation=True, padding=True)
                        value_embeddings = model(input_ids=inputs['input_ids'].to('cuda:1'),
                                                 attention_mask=inputs['attention_mask'].to('cuda:1')).last_hidden_state

                        embedding_wikicache[qid][prop] = value_embeddings.detach().cpu()

    with open('/dlabdata1/culjak/aida/embedding_wikicache_test.pkl', 'wb') as f:
        pickle.dump(embedding_wikicache, f)

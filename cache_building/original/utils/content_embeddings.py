import json
import pickle
from processing import *

import torch
from transformers import BartTokenizer, BartModel
from tqdm import tqdm

if __name__ == '__main__':
    print('Loading model...')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartModel.from_pretrained('facebook/bart-base').to('cuda:1').eval()
    print('Loaded model!')

    with open('../caches/AIDA/.json') as f:
        data = json.load(f)
        #data = make_names_dict(data)

    content_cache = {}

    for article in tqdm(data):
        with torch.no_grad():
            inputs = tokenizer(article['content'], return_tensors='pt', truncation=True, padding=True)
            article_embeddings = model(input_ids=inputs['input_ids'].to('cuda:1'),
                                     attention_mask=inputs['attention_mask'].to('cuda:1')).last_hidden_state
            content_cache[article['articleID']] = article_embeddings.detach().cpu()

    with open('/dlabdata1/culjak/aida/embedding_contentcache_val.pkl', 'wb') as f:
        pickle.dump(content_cache, f)




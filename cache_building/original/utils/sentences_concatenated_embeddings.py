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

    with open('../data_new.json') as f:
        data = json.load(f)
        data = make_names_dict(data)

    embedding_sentencecache = {}

    for article in tqdm(data):
        embedding_sentencecache[article['articleID']] = {}
        for name in article['names']:
            with torch.no_grad():
                sentences = sentences_with_name(name, article['content'])
                content = [' '.join(sentences)]
                inputs = tokenizer(sentences, return_tensors='pt', truncation=True, padding=True)
                sentence_embeddings = model(input_ids=inputs['input_ids'].to('cuda:1'),
                                            attention_mask=inputs['attention_mask'].to('cuda:1')).last_hidden_state
                embedding_sentencecache[article['articleID']][name['name']] = sentence_embeddings

    with open('/dlabdata1/culjak/embedding_contentcache_cat.pkl', 'wb') as f:
        pickle.dump(embedding_sentencecache, f)



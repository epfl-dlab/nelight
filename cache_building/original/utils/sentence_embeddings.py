import pickle

import torch

from processing import *
from transformers import BartTokenizer, BartModel
from tqdm import tqdm

with open('/dlabdata1/culjak/aida_val.json', 'r') as f:
    data = json.load(f)

embedding_sentencecache = {}

print('Loading model...')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartModel.from_pretrained('facebook/bart-base').to('cuda:1').eval()
print('Loaded model!')

for article in tqdm(data):
    embedding_sentencecache[article['articleID']] = {}
    try:
        for name in article['names']:
            with torch.no_grad():
                sentences = sentences_with_name(name, article['content'])
                inputs = tokenizer(sentences, return_tensors='pt', truncation=True, padding=True)
                sentence_embeddings = model(input_ids=inputs['input_ids'].to('cuda:1'),
                                         attention_mask=inputs['attention_mask'].to('cuda:1')).last_hidden_state
                embedding_sentencecache[article['articleID']][name['name']] = sentence_embeddings.detach().cpu()
    except ValueError:
        print(mark_name(name, article['content']))
        print(sentences)
        print(article['content'])
        print(article['names'])
        break

with open('/dlabdata1/culjak/aida/embedding_sentencecache_val.pkl', 'wb') as f:
    pickle.dump(embedding_sentencecache, f)

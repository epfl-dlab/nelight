from tqdm import tqdm
from scoring.semantic import EntityContentSimilarityScorer
from scoring.ensemble import WeightedEnsemble

from data_utils import load_pickle, save_pickle, load_json

data = load_json('../data/Quotebank/data.json')
wiki_cache = load_pickle('../caches/Quotebank/wikicache.pkl')
content_embeddings = load_pickle('../caches/Quotebank/content_embeddings.pkl')
sentence_embeddings = load_pickle('../caches/Quotebank/sentence_embeddings.pkl')
wikidata_embeddings = load_pickle('../caches/Quotebank/wikidata_embeddings.pkl')

we = WeightedEnsemble()

scorers = [
    EntityContentSimilarityScorer('iscore_narrow', stem=True, wiki_cache=wiki_cache, props_to_avoid={'first_paragraph'}),
    EntityContentSimilarityScorer('iscore', stem=True, wiki_cache=wiki_cache, props_to_avoid={'first_paragraph'}),

    EntityContentSimilarityScorer('cse_narrow', embeddings_cache=wikidata_embeddings, sentence_embeddings_cache=sentence_embeddings),
    EntityContentSimilarityScorer('cse', embeddings_cache=wikidata_embeddings, content_embeddings_cache=content_embeddings)
]

scores = [scorer(data) for scorer in tqdm(scorers)]

iscore_ensemble = we.combine_scores(scores[0], scores[1])
cse_ensemble = we.combine_scores(scores[2], scores[3])

scores.insert(2, iscore_ensemble)
scores.append(cse_ensemble)


save_pickle(scores, '../scores/narrow_entire_scores.pkl')

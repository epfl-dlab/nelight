from tqdm import tqdm
from scoring.semantic import *
from data_utils import load_pickle, save_pickle, load_json

data = load_json('../data/Quotebank/data.json')

cache = load_pickle('../caches/Quotebank/wikicache.pkl')
alias_cache = load_pickle('../caches/Quotebank/wikicache_alias.pkl')


iscorers = [
    #Single features
    # D
    EntityContentSimilarityScorer('iscore', wiki_cache=cache, props_to_keep={'description'}),
    EntityContentSimilarityScorer('iscore', lemmatize=True, wiki_cache=cache, props_to_keep={'description'}),
    EntityContentSimilarityScorer('iscore', stem=True, wiki_cache=cache, props_to_keep={'description'}),

    # P
    EntityContentSimilarityScorer('iscore', wiki_cache=cache, props_to_keep={'first_paragraph'}),
    EntityContentSimilarityScorer('iscore', lemmatize=True, wiki_cache=cache, props_to_keep={'first_paragraph'}),
    EntityContentSimilarityScorer('iscore', stem=True, wiki_cache=cache, props_to_keep={'first_paragraph'}),

    # S
    EntityContentSimilarityScorer('iscore', wiki_cache=cache, props_to_avoid={'description', 'first_paragraph'}),
    EntityContentSimilarityScorer('iscore', lemmatize=True, wiki_cache=cache, props_to_avoid={'description', 'first_paragraph'}),
    EntityContentSimilarityScorer('iscore', stem=True, wiki_cache=cache, props_to_avoid={'description', 'first_paragraph'}),

    # S_A
    EntityContentSimilarityScorer('iscore', wiki_cache=alias_cache, props_to_avoid={'description', 'first_paragraph'}),
    EntityContentSimilarityScorer('iscore', lemmatize=True, wiki_cache=alias_cache, props_to_avoid={'description', 'first_paragraph'}),
    EntityContentSimilarityScorer('iscore', stem=True, wiki_cache=alias_cache, props_to_avoid={'description', 'first_paragraph'}),

    #Pairs
    # D + P
    EntityContentSimilarityScorer('iscore', wiki_cache=cache, props_to_keep={'description', 'first_paragraph'}),
    EntityContentSimilarityScorer('iscore', lemmatize=True, wiki_cache=cache, props_to_keep={'description', 'first_paragraph'}),
    EntityContentSimilarityScorer('iscore', stem=True, wiki_cache=cache, props_to_keep={'description', 'first_paragraph'}),

    # D + S
    EntityContentSimilarityScorer('iscore', wiki_cache=cache, props_to_avoid={'first_paragraph'}),
    EntityContentSimilarityScorer('iscore', lemmatize=True, wiki_cache=cache, props_to_avoid={'first_paragraph'}),
    EntityContentSimilarityScorer('iscore', stem=True, wiki_cache=cache, props_to_avoid={'first_paragraph'}),

    # D + S_A
    EntityContentSimilarityScorer('iscore', wiki_cache=alias_cache, props_to_avoid={'first_paragraph'}),
    EntityContentSimilarityScorer('iscore', lemmatize=True, wiki_cache=alias_cache, props_to_avoid={'first_paragraph'}),
    EntityContentSimilarityScorer('iscore', stem=True, wiki_cache=alias_cache, props_to_avoid={'first_paragraph'}),

    # P + S
    EntityContentSimilarityScorer('iscore', wiki_cache=cache, props_to_avoid={'description'}),
    EntityContentSimilarityScorer('iscore', lemmatize=True, wiki_cache=cache, props_to_avoid={'description'}),
    EntityContentSimilarityScorer('iscore', stem=True, wiki_cache=cache, props_to_avoid={'description'}),

    # P + S_A
    EntityContentSimilarityScorer('iscore', wiki_cache=alias_cache, props_to_avoid={'description'}),
    EntityContentSimilarityScorer('iscore', lemmatize=True, wiki_cache=alias_cache, props_to_avoid={'description'}),
    EntityContentSimilarityScorer('iscore', stem=True, wiki_cache=alias_cache, props_to_avoid={'description'}),

    #Triples
    # D + P + S
    EntityContentSimilarityScorer('iscore', wiki_cache=cache),
    EntityContentSimilarityScorer('iscore', lemmatize=True, wiki_cache=cache),
    EntityContentSimilarityScorer('iscore', stem=True, wiki_cache=cache),

    # D + P + S_A
    EntityContentSimilarityScorer('iscore', wiki_cache=alias_cache),
    EntityContentSimilarityScorer('iscore', lemmatize=True, wiki_cache=alias_cache),
    EntityContentSimilarityScorer('iscore', stem=True, wiki_cache=alias_cache),
]

save_pickle([iscorer(data) for iscorer in tqdm(iscorers)], '../scores/fn_ablation_scores.pkl')



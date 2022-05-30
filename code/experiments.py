from data_utils import load_pickle, load_json
from metrics import precision_at_one, mrr
from scoring.ensemble import SameScoreRankingEnsemble
from scoring.centrality import KnowledgeGraphCentralityScorer

qb = load_json('../data/Quotebank/data.json')
qb_gt = load_json('../data/Quotebank/overall.json')

popularity_scores = load_pickle('../scores/popularity_scores.pkl')


def fn_ablation():
    scores = load_pickle('../scores/fn_ablation_scores.pkl')

    features = ['D', 'P', 'S', 'S_A', 'D + P', 'D + S', 'D + S_A', 'P + S', 'P + S_A', 'D + P + S', 'D + P + S_A']
    norms = ['No normalization', 'Lemmatization', 'Stemming']

    ssre = SameScoreRankingEnsemble()

    def iter_featrues_norms():
        for feature in features:
            for norm in norms:
                yield feature, norm

    string = ''
    for i, (feature, norm) in enumerate(iter_featrues_norms()):
        scores[i] = ssre.combine_scores(qb, scores[i], popularity_scores['qb']['ns'])
        string += (f'Feature: {feature:12s} '
                   f'Normalization: {norm:20s} '
                   f'P@1: {precision_at_one(qb_gt, scores[i], ignore_missing=True):.3f} '
                   f'MRR: {mrr(qb_gt, scores[i], ignore_missing=True):.3f}\n')

    with open('../results/fn_ablation_results.txt', 'w') as f:
        f.write(string)


if __name__ == '__main__':
    fn_ablation()

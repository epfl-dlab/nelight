import numpy as np
from scoring.evaluation import *
from itertools import groupby
import scipy.stats as ss


class ScoringEnsemble:
    def combine_scores(self, articles, *scores):
        raise NotImplementedError

    def __call__(self, articles, *scores):
        return self.combine_scores(articles, *scores)


class SameScoreRankingEnsemble(ScoringEnsemble):
    def combine_scores(self, articles, *scores):
        scores_first, scores_second = scores[0], scores[1]
        scores_final = {}
        for article in articles:
            aid = article['articleID']
            scores_final[aid] = {}
            for name in article['names']:
                if len(name['ids']) > 1:
                    try:
                        scores = scores_first[aid][name['name']]
                        other_scores = scores_second[aid][name['name']]
                        ranks = ss.rankdata(scores, method='min')
                        n = len(scores)

                        for i in range(1, n + 1):
                            ranks[ranks == i] = ranks[ranks == i] + ss.rankdata(other_scores[ranks == i], method='min') - 1

                        scores_final[aid][name['name']] = ranks
                    except KeyError:
                        pass
        return scores_final


class WeightedEnsemble(ScoringEnsemble): 
    def combine_scores(self, *scores, weights=None):
        if weights is None:
            weights = np.ones(len(scores))

        scores_final = {}
        for articleID, data in scores[0].items():
            scores_final[articleID] = {}
            for name, score in data.items():
                scores_final[articleID][name] = score * weights[0]

        for s, w in zip(scores[1:], weights[1:]):
            if w == 0:
                continue
            for articleID, data in s.items():
                for name, score in data.items():
                    scores_final[articleID][name] = scores_final[articleID][name] + score * w

        for i, j in scores_final.items():
            for k, l in j.items():
                highest = np.argwhere(np.abs(l - np.max(l)) <= 1e-6)
                l[highest] = np.max(l)

        return scores_final

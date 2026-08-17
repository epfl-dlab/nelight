import numpy as np

from runlib.scoring import Scorer


class WikidataCentralityScorer(Scorer):
    def __init__(self, scorer, wiki_cache=None):
        super().__init__(scorer)
        self.wiki_cache = wiki_cache

    def score(self, name, article=None):
        return self.scorer(name)

    def _metric(self, qid, key):
        return self.wiki_cache[qid][key] if qid in self.wiki_cache else -1

    def LQID(self, name):
        return np.array([-int(i[1:]) for i in name["ids"]], dtype=np.float64).argsort().argsort() + 1

    def NP(self, name):
        return np.array([self._metric(qid, "n_statements") for qid in name["ids"]])

    def NS(self, name):
        return np.array([self._metric(qid, "n_sitelinks") for qid in name["ids"]])

    def PRWP(self, name):
        return np.array([self._metric(qid, "pagerank") for qid in name["ids"]])

    def PRWD(self, name):
        return np.array([self._metric(qid, "pagerank_wd") for qid in name["ids"]])

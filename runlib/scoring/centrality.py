import pickle
import numpy as np

from runlib.scoring import Scorer

class KnowledgeGraphCentralityScorer(Scorer):
    def score(self, name, article_content=None):
        return self.scorer(name)


class WikidataCentralityScorer(KnowledgeGraphCentralityScorer):
    def __init__(self, scorer, wiki_cache=None):
        super().__init__(scorer)
        self.wiki_cache = wiki_cache

    def _get_centrality_metric(self, qid, centrality_metric):
        return self.wiki_cache[qid][centrality_metric] if qid in self.wiki_cache else -1

    def local_qid(self, name):
        return np.array([-int(i[1:]) for i in name['ids']], dtype=np.float64).argsort().argsort() + 1

    def n_statements(self, name):
        """
        Local only
        :param name: A dictionary containing the name of the mention, candiate qids, and positions in the text.
        :return: a numpy array of scores
        """
        return np.array([self._get_centrality_metric(qid, 'n_statements') for qid in name['ids']])

    def n_sitelinks(self, name):
        return np.array([self._get_centrality_metric(qid, 'n_sitelinks') for qid in name['ids']])

    def pagerank(self, name):
        return np.array([self._get_centrality_metric(qid, 'pagerank') for qid in name['ids']])

    def pagerank_wd(self, name):
        return np.array([self._get_centrality_metric(qid, 'pagerank_wd') for qid in name['ids']])

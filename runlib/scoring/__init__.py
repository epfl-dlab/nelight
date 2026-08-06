from abc import abstractmethod
from types import FunctionType
import numpy as np

class Scorer:
    def __init__(self, scorer, initial_sort_key=lambda x: int(x[1:])):
        if isinstance(scorer, FunctionType):
            self.scorer = scorer
        else:
            self.scorer = getattr(self, scorer)
        self.initial_sort_key = initial_sort_key

    @abstractmethod
    def score(self, name, article):
        raise NotImplementedError

    def _score_single_article(self, article, ignore_unambiguous=True):
        return {
            name['name']: (np.vstack([self.score(name, article)])).sum(axis=0)
            for name in article['names']
            if ignore_unambiguous and len(name['ids']) > 1
        }

    def score_all(self, data, ignore_unambiguous=True):
        return {
            article['articleID']: self._score_single_article(article, ignore_unambiguous)
            for article in data
        }

    def __call__(self, data, ignore_unambiguous=True):
        return self.score_all(data, ignore_unambiguous)

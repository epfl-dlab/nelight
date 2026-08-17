from types import FunctionType

import numpy as np


class Scorer:
    def __init__(self, scorer):
        self.scorer = scorer if isinstance(scorer, FunctionType) else getattr(self, scorer)

    def score(self, name, article):
        return self.scorer(name, article)

    def score_all(self, data, ignore_unambiguous=True):
        out = {}
        for article in data:
            scores = {}
            for name in article["names"]:
                if ignore_unambiguous and len(name["ids"]) <= 1:
                    continue
                scores[name["name"]] = np.asarray(self.score(name, article))
            out[article["articleID"]] = scores
        return out

    def __call__(self, data, ignore_unambiguous=True):
        return self.score_all(data, ignore_unambiguous)

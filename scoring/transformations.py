import numpy as np


def transform_scores(scores, transformation):
    transformed_scores = {}
    for article, name_scores in scores.items():
        transformed_scores[article] = {}
        for name, score in name_scores.items():
            transformed_scores[article][name] = transformation(score)

    return transformed_scores


def laplacian_smoothing_transformation(scores):
    return prob_transformation(scores + 1)


def prob_transformation(scores):
    return scores / np.sum(scores)


def clip(scores, value):
    return np.where(scores < value, 0, scores)

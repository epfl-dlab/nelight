import numpy as np
from data_utils import iter_gt, iter_data


def precision_at_one(gt, scores, ignore_missing=False):
    total = 0
    correct = 0
    for aid, name, gold in iter_gt(gt):
        if ignore_missing and gold is None:
            continue

        correct += (gold is not None) and (gold == np.argmax(scores[aid][name]))
        total += 1

    return correct / total


def mrr(gt, scores, ignore_missing=False):
    total = 0
    srr = 0
    for aid, name, gold in iter_gt(gt):
        if ignore_missing and gold is None:
            continue

        rank = np.argsort(-scores[aid][name])
        srr += 1 / (np.where(rank == gold)[0][0] + 1) if gold is not None else 0
        total += 1

    return srr / total


def eval_random(data, gt, ignore_missing=False):
    total = 0
    correct = 0
    srr = 0
    for aid, name in iter_data(data):
        if aid not in gt:
            continue
        if name['name'] not in gt[aid]:
            continue
        gold = gt[aid][name['name']]

        if ignore_missing and gold is None:
            continue

        if len(name['ids']) == 0:
            correct += 0
            srr += 0
        else:
            correct += (gold is not None) * 1 / len(name['ids']) if len(name['ids']) > 0 else 0
            srr += (gold is not None) * sum(1 / (i + 1) for i in range(len(name['ids']))) / len(name['ids'])
        total += 1

    return correct / total, srr / total
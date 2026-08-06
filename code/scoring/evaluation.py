import json

import numpy as np

import scipy.stats as ss
from randomdict import RandomDict
from tqdm.notebook import tqdm
from random import choices
from utils.processing import *
from scoring.semantic import *
DATA_PATH = '../data/data.json'
GT_PATH = '../data/gt_new.json'

def precision_at_one(gt, scores):
    return np.mean([
        np.argmax(scores_arr) == gt[articleID][name.lower()]
        for articleID, name_scores in scores.items()
        for name, scores_arr in name_scores.items()
        if articleID in gt and name.lower() in gt[articleID] and gt[articleID][name.lower()] is not None
    ])


def precision_at_one_flat(gt, scores):
    total = 0
    correct = 0
    for (articleID, name), gt_idx in gt:
        if gt_idx is not None:
            correct += np.argmax(scores[articleID][name.lower()]) == gt_idx
            total += 1
    return correct / total

def precision_at_one_flat_o(gt, scores, total=4478):
    correct = 0
    total = len(gt)
    for (articleID, name), gt_idx in gt:
        try:
            correct += np.argmax(scores[articleID][name.lower()]) == gt_idx
        except:
            print(gt[articleID])
            print(scores[articleID])
    return correct / total

def MRR_flat_o(gt, scores, total=4478):
    total = len(gt)
    correct = 0
    for (articleID, name), gt_idx in gt:
        rank = np.argsort(-scores[articleID][name.lower()])
        correct += 1 /(np.where(rank == gt_idx)[0][0] + 1)
    return correct / total

def MRR_flat(gt, scores):
    total = 0
    correct = 0
    for (articleID, name), gt_idx in gt:
        if gt_idx is not None:
            rank = np.argsort(-scores[articleID][name.lower()])
            correct += 1 /(np.where(rank == gt_idx)[0][0] + 1)
            total += 1
    return correct / total



def get_wrong_preds(gt, scores, data):
    wrong_ids_names = {
        (articleID, name)
        for articleID, name_scores in scores.items()
        for name, scores_arr in name_scores.items()
        if articleID in gt and name.lower() in gt[articleID] and gt[articleID][name.lower()] is not None
        if np.argmax(scores_arr) != gt[articleID][name.lower()]
    }


    wrong_predictions = [
        (article['articleID'],
         name['name'],
         name['ids'],
         scores[article['articleID']][name['name'].lower()],
         np.where(np.argsort(-scores[article['articleID']][name['name'].lower()]) == gt[article['articleID']][name['name'].lower()])[0][0],
         gt[article['articleID']][name['name'].lower()],
         mark_name(name, article['content']))
        for article in data
        for name in article['names']
        if (article['articleID'], name['name']) in wrong_ids_names
    ]

    return wrong_predictions

def MRR(gt, scores):
    RR = 0
    total = 0
    for articleID, name_scores in scores.items():
        for name, scores_arr in name_scores.items():
            if articleID in gt and name.lower() in gt[articleID]:
                gt_rank = gt[articleID][name.lower()]
                if gt_rank is not None:
                    rank = np.argsort(-scores_arr)
                    RR += 1 / (np.where(rank == gt_rank)[0][0] + 1)
                    total += 1

    return RR / total

def spearman(scores_1, scores_2  , gt):
    spearman_sum = 0
    total = 0
    for articleID, name_scores in scores_1.items():
        for name, scores_arr_1 in name_scores.items():
            if gt[articleID][name.lower()] is not None:
                scores_arr_2 = scores_2[articleID][name]
                ranks_1 = ss.rankdata(-scores_arr_1)
                ranks_2 = ss.rankdata(-scores_arr_2)
                d = (ranks_1 - ranks_2) ** 2
                n = len(ranks_1)
                spearman_sum += 1 - 6 * (d.sum()) / (n * (n ** 2 - 1))
                total += 1

    return spearman_sum / total

def pat1_random(gt, data):
    total = 0
    correct = 0
    for article in data:
        aid = article['articleID']
        for name in article['names']:
            if aid in gt:
                n = name['name']
                if n in gt[aid]:
                    if len(name['ids']) > 1 and gt[article['articleID']][name['name'].lower()] is not None:
                        total += 1
                        correct += 1 / len(name['ids'])

    return correct / total


def MRR_random(gt, data):
    total = 0
    correct = 0
    for article in data:
        aid = article['articleID']
        for name in article['names']:
            if aid in gt:
                n = name['name']
                if n in gt[aid]:
                    if len(name['ids']) > 1 and gt[article['articleID']][name['name'].lower()] is not None:
                        total += 1
                        correct += sum(1 / (i + 1) for i in range(len(name['ids']))) / len(name['ids'])

    return correct / total


def flatten_gt(gt, ignore_None=False):
    gt_flat = {}
    for article, names in gt.items():
        for name, gt_idx in names.items():
            if ignore_None and gt_idx is None:
                continue
            gt_flat[article, name] = gt_idx

    return gt_flat


def unflatten_gt(gt_flat):
    gt = {}
    for (articleID, name), gt_idx in gt_flat.items():
        if articleID not in gt:
            gt[articleID] = {}
        gt[articleID][name] = gt_idx
    return gt


def val_test_split(gt, val_size=0.2, dump=False):
    gt_flat = flatten_gt(gt)
    val_gt_flat = {}
    test_gt_flat = {}
    val_size = int(val_size * len(gt_flat))
    indices = np.array(range(len(gt_flat)))
    np.random.shuffle(indices)
    val_indices = set(indices[:val_size])

    for i, ((articleID, name), gt_idx) in enumerate(gt_flat.items()):
        if i in val_indices:
            val_gt_flat[articleID, name] = gt_idx
        else:
            test_gt_flat[articleID, name] = gt_idx

    val_gt = unflatten_gt(val_gt_flat)
    test_gt = unflatten_gt(test_gt_flat)

    if dump:
        with open('val_gt.json', 'w') as f:
            json.dump(val_gt, f)
        with open('test_gt.json', 'w') as f:
            json.dump(test_gt, f)

    return val_gt, test_gt

def sample_bootstrap(gt, n_samples, sample_size=None, verbose=True):
    gt_flat = flatten_gt(gt)
    if sample_size == None:
        sample_size = len(gt_flat)
    gt_rd = RandomDict(gt_flat)
    return [[gt_rd.random_item() for _ in range(sample_size)] for _ in (tqdm(range(n_samples)) if verbose else range(n_samples))]

def get_CI(results, confidence):
    return np.percentile(results, (1 - confidence) * 50), np.percentile(results, 100 - (1 - confidence) * 50)

def bootstrap_CI(gt, scores, metric, n_samples, confidence=0.95, sample_size=None, verbose=True):
    verbose and print('Sampling...')
    samples = sample_bootstrap(gt, n_samples, sample_size, verbose)
    verbose and print('Evaluating...')
    results = sorted([metric(sample, scores) for sample in (tqdm(samples) if verbose else samples)])
    verbose and print('Done!')
    return get_CI(results, confidence)

def bootstrap_random(data, gt, n_samples, sample_size=None, confidence=0.95):
    if sample_size is None:
        sample_size = len(data)
    ids_len_arr = []
    for article in data:
        aid = article['articleID']
        for name in article['names']:
            if aid in gt:
                n = name['name']
                if n in gt[aid]:
                    if len(name['ids']) > 1 and gt[article['articleID']][name['name'].lower()] is not None:
                        ids_len_arr.append(len(name['ids']))
                    else:
                        ids_len_arr.append(np.nan)


    harmonic_sum_mean = np.vectorize(
        lambda x: (np.mean(1 / (np.arange(1, x + 1))) if not np.isnan(x) else np.nan)
    )

    print('Sampling...')
    samples = np.array([choices(ids_len_arr, k=sample_size) for _ in tqdm(range(n_samples))])
    print('Evaluating...')
    pat1 = np.nanmean(1 / samples, axis=1)
    MRR = np.nanmean(harmonic_sum_mean(samples), axis=1)
    return get_CI(pat1, confidence), get_CI(MRR, confidence)

def bootstrap_random_o(data, gt, n_samples, sample_size=None, confidence=0.95):
    if sample_size is None:
        sample_size = len(data)
    ids_len_arr = []
    for article in data:
        aid = article['articleID']
        for name in article['names']:
            if aid in gt:
                n = name['name']
                if n in gt[aid]:
                    if len(name['ids']) > 1 and gt[article['articleID']][name['name'].lower()] is not None:
                        ids_len_arr.append(len(name['ids']))
                    else:
                        ids_len_arr.append(np.nan)


    harmonic_sum_mean = np.vectorize(
        lambda x: (np.mean(1 / (np.arange(1, x + 1))) if not np.isnan(x) else np.nan)
    )

    print('Sampling...')
    samples = np.array([choices(ids_len_arr, k=sample_size) for _ in tqdm(range(n_samples))])
    print('Evaluating...')
    pat1 = np.nanmean(1 / samples, axis=1)
    MRR = np.mean(harmonic_sum_mean(samples), axis=1)
    return get_CI(pat1, confidence), get_CI(MRR, confidence)



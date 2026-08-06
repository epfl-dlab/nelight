import pickle
import json


def iter_gt(gt):
    for articleID, name_scores in gt.items():
        for name, gold in name_scores.items():
            yield articleID, name, gold


def iter_data(data):
    for article in data:
        for name in article['names']:
            yield article['articleID'], name


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)

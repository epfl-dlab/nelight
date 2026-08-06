from __future__ import annotations

import xml.etree.ElementTree as ET
import gzip
from typing import Callable

from tqdm import tqdm
from helpers import load_pickle, save_pickle, _open, qtoi, sw, special_characters
from abc import abstractmethod
from imports import *
from nltk import word_tokenize
from nltk import sent_tokenize
from collections import defaultdict


class RedirectResolver:
    def __init__(self, redirect_tree_path):
        tree = ET.parse(redirect_tree_path)
        root = tree.getroot()
        self.redirect_dict = {}

        for i in tqdm(root[1]):
            source = re.sub('.*/', '', i[0][0].text)
            target = re.sub('.*/', '', i[1][0].text)
            self.redirect_dict[source] = target

    def resolve_redirects(self, cache):
        for i, j in tqdm(self.redirect_dict.items()):
            if i not in cache and j in cache:
                cache[i] = cache[j]

        return cache


class Serializable:
    @staticmethod
    def from_pickle(path):
        return load_pickle(path)

    def to_pickle(self, path):
        save_pickle(self, path)


class EntityRepresentationCache(Serializable):
    def __init__(self, cache_dict):
        self._cache_dict = cache_dict

    def __getitem__(self, entity):
        return self._cache_dict[entity]

    def __setitem__(self, key, value):
        self._cache_dict[key] = value

    def __contains__(self, entity):
        return self._cache_dict[entity]

    def __len__(self):
        return len(self._cache_dict)

    def __iter__(self):
        return iter(self._cache_dict.items())

    def entities(self):
        return self._cache_dict.keys()

    def representations(self):
        return self._cache_dict.values()



class Pipeline:
    def __init__(self, branched=False):
        self.pipeline_parts = []
        self.branched = branched

    def __call__(self, data):
        if self.branched:
            return self._branched_call(data)
        return self._unbranched_call(data)

    def add(self, processor: Callable):
        self.pipeline_parts.append(processor)
        return self

    def process(self, data):
        return self(data)

    def branch(self):
        pipeline = Pipeline(branched=True)
        pipeline.pipeline_parts = self.pipeline_parts
        return pipeline

    def unbranch(self):
        pipeline = Pipeline()
        pipeline.pipeline_parts = self.pipeline_parts
        return pipeline

    def _unbranched_call(self, data):
        for part in self.pipeline_parts:
            data = part(data)

        return data

    def _branched_call(self, data):
        for part in self.pipeline_parts:
            data = list(map(part, data))
        return data



class TokenSetRepresentationCache(EntityRepresentationCache):
    @staticmethod
    def from_dump(path, io='rb', compression=gzip, tokenizer=word_tokenize, wikidata_labels=None,
                  wikidata_descriptions=None, wikidata_aliases=None):
        if wikidata_labels is None:
            raise ValueError('Wikidata labels not provided')

        representation_cache = {}

        with _open(path, io, compression) as f:
            for line in tqdm(f):
                d = json.loads(line[:-2])
                qid = qtoi(d['id'])
                token_set = set()
                if wikidata_descriptions is not None and qid in wikidata_descriptions:
                    token_set = token_set.union(wikidata_descriptions[qid])

                # We don't want the tokens that belong to the label of an entity to be included in its representation
                if wikidata_labels is not None:
                    name = wikidata_labels[qid].lower()
                    alias_set = set(tokenizer(name))
                else:
                    alias_set = set()

                claims = d['claims']

                if wikidata_labels is not None:
                    for prop, values in claims.items():
                        for value in values:
                            if 'mainsnak' in value:
                                mainsnak = value['mainsnak']

                                if 'datatype' in mainsnak:
                                    type = mainsnak['datatype']
                                    if type != 'wikibase-item' and type != 'string':
                                        continue

                                if 'datavalue' not in mainsnak:
                                    continue

                                value = mainsnak['datavalue']['value']
                                if type == 'string':
                                    token_set = token_set.union(tokenizer(value))
                                elif type == 'wikibase-item':
                                    value_qid = qtoi(value['id'])

                                    if value_qid in wikidata_labels:
                                        token_set = token_set.union(tokenizer(wikidata_labels[value_qid].lower()))

                                    if wikidata_aliases is not None and value_qid in wikidata_aliases:
                                        aliases = wikidata_aliases[value_qid]

                                        for alias in aliases:
                                            token_set = token_set.union(tokenizer(alias.lower()))

                representation_cache[qid] = token_set.difference(sw).difference(alias_set).difference(
                    special_characters)

        return TokenSetRepresentationCache(representation_cache)

    def to_parquet(self, path):
        qids = self.entities()
        representations = [list(i) for i in self.representations()]
        df = pd.DataFrame(dict(qid=qids, representation=representations))
        df.to_parquet(path)

class NSCache(EntityRepresentationCache):
    @staticmethod
    def from_dump(path, io='rb', compression=gzip):
        ns_cache = {}
        with _open(path, io, compression) as f:
            for line in tqdm(f):
                d = json.loads(line[:-2])
                qid = d['id']
                sitelinks = d['sitelinks']
                ns_cache[qid] = len(sitelinks)

        return NSCache(ns_cache)


class NPCache(EntityRepresentationCache):
    @staticmethod
    def from_dump(path, io='rb', compression=gzip):
        np_cache = {}
        with _open(path, io, compression) as f:
            for line in tqdm(f):
                d = json.loads(line[:-2])
                qid = d['id']
                np_cache[qid] = len(d['claims'])

        return NPCache(np_cache)


class PRCache(EntityRepresentationCache):
    @staticmethod
    def from_ranks_file(path, io='rb', compression=gzip):
        pagerank_cache = {}
        with _open(path, io, compression) as f:
            for line in tqdm(f):
                line = line.strip().split('\t')
                pagerank_cache['Q' + line[0]] = float(line[1])

        return PRCache(pagerank_cache)


class StatementRepresentationCache(EntityRepresentationCache):
    @staticmethod
    def from_dump(path, io='rb', compression=gzip):
        attrs_cache = defaultdict(set)

        with _open(path, io, compression) as f:
            for line in tqdm(f):
                d = json.loads(line[:-2])
                qid = d['id']
                claims = d['claims']

                for prop, values in claims.items():
                    for value in values:
                        if 'mainsnak' in value:
                            mainsnak = value['mainsnak']

                            if 'datatype' in mainsnak:
                                type = mainsnak['datatype']
                                if type != 'wikibase-item' and type != 'string':
                                    continue

                            if 'datavalue' not in mainsnak:
                                continue

                            if type == 'string':
                                value = mainsnak['datavalue']['value']
                            elif type == 'wikibase-item':
                                value = mainsnak['datavalue']['value']['id']

                            attrs_cache[qid].add((prop, value))

        return StatementRepresentationCache(attrs_cache)

    def to_parquet(self, path):
        qids = self.entities()
        representations = [[list(j) for j in i] for i in self.representations()]
        df = pd.DataFrame(dict(qid=qids, representation=representations))
        df.to_parquet(path)


class DocumentProcessor(Callable):
    @abstractmethod
    def __call__(self, text):
        raise NotImplementedError


class NarrowTokenizeProcessor(DocumentProcessor):
    def __call__(self, content):
        text = content[0]
        offsets = content[1]
        content_tokens = text.split()
        content_sentences = sent_tokenize(text)
        reps = []
        for entity_offsets in offsets:
            mentions = []
            for offset in entity_offsets:
                mention = ' '.join(content_tokens[offset[0]: offset[1]])

                if offset[1] - offset[0] == 1:
                    if len(mention) == 1:
                        # If a mention has only one letter, it is faulty
                        continue
                    if not any(c.isalpha() for c in mention):
                        # If there are no letters in a mention, it is faulty
                        continue
                mentions.append(' '.join(content_tokens[offset[0]: offset[1]]))

            mentions = set(mentions)
            narrow_context = []

            for sentence in content_sentences:
                for mention in mentions:
                    if mention in sentence:
                        narrow_context.append(sentence)
                        break

            reps.append(' '.join(narrow_context))

        return reps


class UnambiguousEntitiesProcessor(DocumentProcessor):
    def __call__(self, unambiguous_representations):
        document_representation = []
        for representation in unambiguous_representations:
            document_representation.extend(representation)
        document_representation = set(document_representation)
        return document_representation

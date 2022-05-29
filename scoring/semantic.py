import re
import numpy as np
import torch
import torch.nn.functional as F

from pywsd.utils import lemmatize_sentence
from nltk import WordNetLemmatizer, PorterStemmer, TreebankWordTokenizer

from nltk.corpus import stopwords
from . import Scorer
from utils.processing import *


class KnowledgeGraphSemanticScorer(Scorer):
    def score(self, name, article_content):
        return self.scorer(name, article_content)


class EntityContentSimilarityScorer(KnowledgeGraphSemanticScorer):
    def __init__(self, scorer, wiki_cache=None,
                 stem=False,
                 lemmatize=False,
                 props_to_avoid=None,
                 props_to_keep=None,
                 qid_wiki_cache=None,
                 embeddings_cache=None,
                 content_embeddings_cache=None,
                 sentence_embeddings_cache=None):
        super().__init__(scorer)

        self.wiki_cache = wiki_cache
        self.embedding_cache = embeddings_cache
        self.qid_wiki_cache = qid_wiki_cache
        self.content_embeddings = content_embeddings_cache
        self.sentence_embeddings = sentence_embeddings_cache
        self._lemmatizer = WordNetLemmatizer() if lemmatize else None
        self._stemmer = PorterStemmer() if stem else None
        self._treebank_tokenizer = TreebankWordTokenizer()
        self._treebank_detokenizer = TreebankWordDetokenizer()
        self._props_to_avoid = [] if props_to_avoid is None else props_to_avoid
        self._props_to_keep = props_to_keep
        self._attribute_weigts = []

    def _remove_name(self, name, article_content_tokens):
        return article_content_tokens.difference(set(self._treebank_tokenizer.tokenize(name)))

    def _preprocess_content(self, article_content, remove_stopwords=True):
        article_content = article_content.replace(u'\xa0', u' ').lower()
        if self._lemmatizer:
            article_content = lemmatize_sentence(article_content)
            article_content = self._treebank_detokenizer.detokenize(article_content).lower()

        tokens = set(self._treebank_tokenizer.tokenize(article_content))

        if remove_stopwords:
            tokens = set(tokens).difference(stopwords.words('english'))
        if self._stemmer:
            tokens = set([self._stemmer.stem(token) for token in tokens])
        return tokens

    def _get_wikidata_bow(self, qid):
        entity_dict = self.wiki_cache[qid]
        bow = set()
        if self._props_to_keep is not None:
            for prop in self._props_to_keep:
                if prop == 'first_paragraph':
                    values = [self.wiki_cache[qid][prop]]
                else:
                    values = self.wiki_cache[qid][prop]
                for value in values:
                    if self._lemmatizer:
                        value = lemmatize_sentence(value)
                        value = self._treebank_detokenizer.detokenize(value).lower()

                    tokens = self._treebank_tokenizer.tokenize(value.lower())
                    if self._stemmer:
                        tokens = set([self._stemmer.stem(token) for token in tokens])
                    bow.update(set([token for token in tokens if re.match('[a-zA-Z]', token)]))
        else:
            for i, j in entity_dict.items():
                if i in {'n_statements', 'n_sitelinks', 'pagerank', 'pagerank_wd', 'indeg', 'outdeg',
                         *self._props_to_avoid}:
                    continue

                if re.match('^P[0-9]+$', i) or i == 'description' or i == 'first_paragraph':
                    if i == 'first_paragraph':
                        j = [j]
                    for value in j:
                        if self._lemmatizer:
                            value = lemmatize_sentence(value)
                            value = self._treebank_detokenizer.detokenize(value).lower()

                        tokens = self._treebank_tokenizer.tokenize(value.lower())
                        if self._stemmer:
                            tokens = set([self._stemmer.stem(token) for token in tokens])
                        bow.update(set([token for token in tokens if re.match('[a-zA-Z]', token)]))
        return bow

    def iscore(self, name, article):
        article_content = article['content']
        article_content_tokens = self._preprocess_content(article_content)

        return np.array([self._iscore_single(name['name'], qid, article_content_tokens) for qid in name['ids']])

    def _iscore_single(self, name, qid, article_content):
        try:
            article_content = self._remove_name(name, article_content)
            description_set = self._get_wikidata_bow(qid)
            intersection = article_content.intersection(description_set)
            return len(intersection)
        except KeyError:
            return 0

    def iscore_narrow(self, name, article):
        article_content = article['content']
        article_content = ' '.join(sentences_with_name(name, article_content))
        article_content_tokens = self._preprocess_content(article_content)

        return np.array([self._iscore_single(name['name'], qid, article_content_tokens) for qid in name['ids']])

    def _paragraph_content_single(self, qid, article_id):
        if qid not in self.embedding_cache or 'first_paragraph' not in self.embedding_cache[qid]:
            return 0
        first_paragraph_embedding = self.embedding_cache[qid]['first_paragraph'].mean(axis=1)
        content_embedding = self.content_embeddings[article_id].mean(axis=1)

        return \
            F.cosine_similarity(first_paragraph_embedding, content_embedding, dim=1, eps=1e-8).detach().cpu().numpy()[0]

    def paragraph_content_embeddings(self, name, article):
        article_id = article['articleID']
        return np.array([self._paragraph_content_single(qid, article_id) for qid in name['ids']])

    def _paragraph_content_narrow_single(self, qid, name, article_id):
        if qid not in self.embedding_cache or 'first_paragraph' not in self.embedding_cache[qid]:
            return -1

        first_paragraph_embedding = self.embedding_cache[qid]['first_paragraph'].mean(axis=1)
        content_embedding = self.sentence_embeddings[article_id][name].mean(axis=1)
        return \
            F.cosine_similarity(first_paragraph_embedding, content_embedding, dim=1, eps=1e-8).detach().cpu().numpy()[0]

    def paragraph_content_narrow(self, name, article):
        article_id = article['articleID']
        return np.array(
            [self._paragraph_content_narrow_single(qid, name['name'], article_id) for qid in name['ids']])

    def _paragraph_or_props_single(self, qid, article_id):
        if qid not in self.embedding_cache or len(self.embedding_cache[qid]) == 0:
            return -1
        if 'first_paragraph' not in self.embedding_cache[qid]:
            try:
                emb = torch.cat([emb.mean(axis=1) for prop, emb in self.embedding_cache[qid].items()]).mean(axis=0)
                content_embedding = self.content_embeddings[article_id].mean(axis=1)
                return F.cosine_similarity(emb, content_embedding, dim=1, eps=1e-8).detach().cpu().numpy()[0]
            except NotImplementedError:
                print(self.embedding_cache[qid])
                return -1
        else:
            return self._paragraph_content_single(qid, article_id)

    def cse(self, name, article):
        article_id = article['articleID']
        return np.array(
            [self._paragraph_or_props_single(qid, article_id) for qid in name['ids']])

    def _paragraph_or_props_narrow_single(self, qid, name, article_id):
        if qid not in self.embedding_cache or len(self.embedding_cache[qid]) == 0:
            return -1
        if 'first_paragraph' not in self.embedding_cache[qid]:
            emb = torch.cat([emb.mean(axis=1) for prop, emb in self.embedding_cache[qid].items()]).mean(axis=0)
            content_embedding = self.sentence_embeddings[article_id][name].mean(axis=1)
            return F.cosine_similarity(emb, content_embedding, dim=1, eps=1e-8).detach().cpu().numpy()[0]
        else:
            return self._paragraph_content_narrow_single(qid, name, article_id)

    def ncse(self, name, article):
        article_id = article['articleID']
        return np.array([self._paragraph_or_props_narrow_single(qid, name['name'], article_id) for qid in name['ids']])


class KnowledgeGraphEntityEntityScorer(Scorer):
    def score(self, name1, name2):
        return self.scorer(name1, name2)


class EntityEntitySimilarityScorer(KnowledgeGraphSemanticScorer):
    def __init__(self, scorer, wiki_cache,
                 stem=False,
                 lemmatize=False,
                 props_to_avoid=None,
                 props_to_keep=None,
                 embeddings_cache=None,
                 unambiguous_cache=None):

        super().__init__(scorer)

        self.wiki_cache = wiki_cache
        self.embedding_cache = embeddings_cache
        self.unambiguous_cache = unambiguous_cache
        self._lemmatizer = WordNetLemmatizer() if lemmatize else None
        self._stemmer = PorterStemmer() if stem else None
        self._treebank_tokenizer = TreebankWordTokenizer()
        self._treebank_detokenizer = TreebankWordDetokenizer()
        self._props_to_avoid = [] if props_to_avoid is None else props_to_avoid
        self._props_to_keep = props_to_keep

    def _matching_attributes_emb_single(self, qid, article_id):
        if len(self.unambiguous_cache[article_id]) == 0:
            return 0
        unambiguous_qids = self.unambiguous_cache[article_id][0]
        score = 0.
        for unambiguous_qid in unambiguous_qids:
            unambiguous_entity_embeddings = self.embedding_cache[unambiguous_qid]
            if qid not in self.embedding_cache:
                continue
            entity_dict = self.embedding_cache[qid]
            for prop, embeddings in unambiguous_entity_embeddings.items():
                if prop not in entity_dict or prop in {'first_paragraph', 'n_statements', 'n_sitelinks', 'pagerank'}:
                    continue

                entity_embeddings = entity_dict[prop]
                prop_scores = []
                for embedding in embeddings:
                    embedding = embedding.mean(dim=0)
                    for entity_embedding in entity_embeddings:
                        entity_embedding = entity_embedding.mean(dim=0)
                        prop_scores.append(F.cosine_similarity(embedding, entity_embedding, dim=0).item())
                score += sum(prop_scores)
        return score

    def cssve(self, name, article):
        return np.array([self._matching_attributes_emb_single(qid, article['articleID']) for qid in name['ids']])

    def _matching_attributes_single(self, qid, article_id):
        if len(self.unambiguous_cache[article_id]) == 0:
            return 0
        unambiguous_qids = self.unambiguous_cache[article_id][0]
        score = 0
        for unambiguous_qid in unambiguous_qids:
            if unambiguous_qid not in self.wiki_cache:
                continue
            unambiguous_entity_dict = self.wiki_cache[unambiguous_qid]
            if qid not in self.wiki_cache:
                continue
            entity_dict = self.wiki_cache[qid]
            for prop, values in unambiguous_entity_dict.items():
                if prop not in entity_dict or prop in {'first_paragraph', 'n_statements', 'n_sitelinks', 'pagerank',
                                                       'pagerank_wd', 'indeg', 'degree', 'outdeg'}:
                    continue
                entity_values = entity_dict[prop]
                if isinstance(entity_values, str):
                    entity_values = [entity_values]
                if isinstance(values, str):
                    values = [values]
                for value in values:
                    for entity_value in entity_values:
                        if entity_value == value:
                            score += entity_value == value
        return score

    def eeiscore(self, name, article):
        return np.array([self._matching_attributes_single(qid, article['articleID']) for qid in name['ids']])

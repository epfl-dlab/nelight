"""Entity/content scorers — Drive originals with packaging + numpy cosine helpers."""

import re

import numpy as np
from nltk import PorterStemmer, TreebankWordTokenizer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from pywsd.utils import lemmatize_sentence

from runlib.scoring import Scorer
from runlib.scoring._emb import cosine, mean_vec, stack_mean
from runlib.utils.processing import sentences_with_name

# Kept for parity with the research tree (unused here).
special_characters = {
    '.', ',', ')', '(', '/', '\\', '<', '>', '+', '-', '!', '?', '@', '"', '#',
    '$', '%', '&', '=', '[', ']', '{', '}', "'", '~', '*', '_', ':',
}


class KnowledgeGraphSemanticScorer(Scorer):
    def score(self, name, article_content):
        return self.scorer(name, article_content)


class EntityContentSimilarityScorer(KnowledgeGraphSemanticScorer):
    def __init__(
        self,
        scorer,
        wiki_cache=None,
        stem=False,
        lemmatize=False,
        props_to_avoid=None,
        props_to_keep=None,
        qid_wiki_cache=None,
        embeddings_cache=None,
        content_embeddings_cache=None,
        sentence_embeddings_cache=None,
    ):
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
        # Memoize BOW / article tokens (score-identical; speeds IScore).
        self._bow_cache = {}
        self._content_cache = {}

    def _remove_name(self, name, article_content_tokens):
        return article_content_tokens.difference(
            set(self._treebank_tokenizer.tokenize(name))
        )

    def _preprocess_content(self, article_content, remove_stopwords=True, cache_key=None):
        if cache_key is not None and cache_key in self._content_cache:
            return set(self._content_cache[cache_key])
        article_content = article_content.replace(u'\xa0', u' ').lower()
        if self._lemmatizer:
            article_content = lemmatize_sentence(article_content)
            article_content = self._treebank_detokenizer.detokenize(article_content).lower()

        tokens = set(self._treebank_tokenizer.tokenize(article_content))

        if remove_stopwords:
            tokens = set(tokens).difference(stopwords.words('english'))
        if self._stemmer:
            tokens = set([self._stemmer.stem(token) for token in tokens])
        if cache_key is not None:
            self._content_cache[cache_key] = frozenset(tokens)
        return tokens

    def _get_wikidata_bow(self, qid):
        if qid in self._bow_cache:
            return self._bow_cache[qid]
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
                    bow.update(set([token for token in tokens if re.match('[a-zA-Z0-9].+', token)]))
        else:
            for i, j in entity_dict.items():
                if i in {
                    'n_statements', 'n_sitelinks', 'pagerank', 'pagerank_wd',
                    'indeg', 'outdeg', *self._props_to_avoid,
                }:
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
        self._bow_cache[qid] = bow
        return bow

    def iscore(self, name, article):
        article_content = article['content']
        cache_key = article.get('articleID')
        article_content_tokens = self._preprocess_content(
            article_content, cache_key=cache_key
        )

        return np.array([
            self._iscore_single(name['name'], qid, article_content_tokens)
            for qid in name['ids']
        ])

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

        return np.array([
            self._iscore_single(name['name'], qid, article_content_tokens)
            for qid in name['ids']
        ])

    def _paragraph_content_single(self, qid, article_id):
        if qid not in self.embedding_cache or 'first_paragraph' not in self.embedding_cache[qid]:
            return 0
        first_paragraph_embedding = mean_vec(
            self.embedding_cache[qid]['first_paragraph'], axis=1
        )
        content_embedding = mean_vec(self.content_embeddings[article_id], axis=1)
        return cosine(first_paragraph_embedding, content_embedding)

    def paragraph_content_embeddings(self, name, article):
        article_id = article['articleID']
        return np.array([
            self._paragraph_content_single(qid, article_id) for qid in name['ids']
        ])

    def _paragraph_content_narrow_single(self, qid, name, article_id):
        if qid not in self.embedding_cache or 'first_paragraph' not in self.embedding_cache[qid]:
            return -1

        first_paragraph_embedding = mean_vec(
            self.embedding_cache[qid]['first_paragraph'], axis=1
        )
        content_embedding = mean_vec(
            self.sentence_embeddings[article_id][name], axis=1
        )
        return cosine(first_paragraph_embedding, content_embedding)

    def paragraph_content_narrow(self, name, article):
        article_id = article['articleID']
        return np.array([
            self._paragraph_content_narrow_single(qid, name['name'], article_id)
            for qid in name['ids']
        ])

    def _paragraph_or_props_single(self, qid, article_id):
        if qid not in self.embedding_cache or len(self.embedding_cache[qid]) == 0:
            return -1
        if 'first_paragraph' not in self.embedding_cache[qid]:
            try:
                emb = stack_mean(self.embedding_cache[qid].values())
                content_embedding = mean_vec(self.content_embeddings[article_id], axis=1)
                return cosine(emb, content_embedding)
            except (NotImplementedError, ValueError, TypeError):
                return -1
        else:
            return self._paragraph_content_single(qid, article_id)

    def cse(self, name, article):
        article_id = article['articleID']
        return np.array([
            self._paragraph_or_props_single(qid, article_id) for qid in name['ids']
        ])

    def _paragraph_or_props_narrow_single(self, qid, name, article_id):
        if qid not in self.embedding_cache or len(self.embedding_cache[qid]) == 0:
            return -1
        if 'first_paragraph' not in self.embedding_cache[qid]:
            emb = stack_mean(self.embedding_cache[qid].values())
            content_embedding = mean_vec(
                self.sentence_embeddings[article_id][name], axis=1
            )
            return cosine(emb, content_embedding)
        else:
            return self._paragraph_content_narrow_single(qid, name, article_id)

    def ncse(self, name, article):
        article_id = article['articleID']
        return np.array([
            self._paragraph_or_props_narrow_single(qid, name['name'], article_id)
            for qid in name['ids']
        ])


class KnowledgeGraphEntityEntityScorer(Scorer):
    def score(self, name1, name2):
        return self.scorer(name1, name2)


class EntityEntitySimilarityScorer(KnowledgeGraphSemanticScorer):
    def __init__(
        self,
        scorer,
        wiki_cache,
        stem=False,
        lemmatize=False,
        props_to_avoid=None,
        props_to_keep=None,
        embeddings_cache=None,
        unambiguous_cache=None,
    ):
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
                if prop not in entity_dict or prop in {
                    'first_paragraph', 'n_statements', 'n_sitelinks', 'pagerank',
                }:
                    continue

                entity_embeddings = entity_dict[prop]
                prop_scores = []
                for embedding in embeddings:
                    embedding = mean_vec(embedding, axis=0)
                    for entity_embedding in entity_embeddings:
                        entity_embedding = mean_vec(entity_embedding, axis=0)
                        prop_scores.append(cosine(embedding, entity_embedding))
                score += sum(prop_scores)
        return score

    def cssve(self, name, article):
        return np.array([
            self._matching_attributes_emb_single(qid, article['articleID'])
            for qid in name['ids']
        ])

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
                if prop not in entity_dict or prop in {
                    'first_paragraph', 'n_statements', 'n_sitelinks', 'pagerank',
                    'pagerank_wd', 'indeg', 'degree', 'outdeg',
                }:
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
        return np.array([
            self._matching_attributes_single(qid, article['articleID'])
            for qid in name['ids']
        ])

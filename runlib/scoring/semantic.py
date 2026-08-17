"""Entity/content scorers using paper method names (IScore, CSE, EEIScore, CSSVE)."""

import re

import numpy as np
from nltk import PorterStemmer, TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from pywsd.utils import lemmatize_sentence

from runlib.scoring import Scorer
from runlib.scoring._emb import cosine, mean_vec, stack_mean
from runlib.utils.processing import sentences_with_name

_SKIP = {
    "n_statements", "n_sitelinks", "pagerank", "pagerank_wd",
    "indeg", "outdeg", "degree", "first_paragraph",
}


class EntityContentSimilarityScorer(Scorer):
    def __init__(
        self,
        scorer,
        wiki_cache=None,
        stem=False,
        lemmatize=False,
        props_to_avoid=None,
        embeddings_cache=None,
        content_embeddings_cache=None,
        sentence_embeddings_cache=None,
    ):
        super().__init__(scorer)
        self.wiki_cache = wiki_cache
        self.embedding_cache = embeddings_cache
        self.content_embeddings = content_embeddings_cache
        self.sentence_embeddings = sentence_embeddings_cache
        self._lemmatize = lemmatize
        self._stemmer = PorterStemmer() if stem else None
        self._treebank_tokenizer = TreebankWordTokenizer()
        self._treebank_detokenizer = TreebankWordDetokenizer()
        self._props_to_avoid = [] if props_to_avoid is None else props_to_avoid
        self._bow_cache = {}
        self._content_cache = {}

    def _lemma(self, text):
        if not self._lemmatize:
            return text
        return self._treebank_detokenizer.detokenize(lemmatize_sentence(text)).lower()

    def _preprocess_content(self, article_content, remove_stopwords=True, cache_key=None):
        if cache_key is not None and cache_key in self._content_cache:
            return set(self._content_cache[cache_key])
        # Stopwords before stemming (Porter turns "was" into "wa").
        article_content = self._lemma(article_content.replace("\xa0", " ").lower())
        tokens = set(self._treebank_tokenizer.tokenize(article_content))
        if remove_stopwords:
            tokens = tokens.difference(stopwords.words("english"))
        if self._stemmer:
            tokens = {self._stemmer.stem(t) for t in tokens}
        if cache_key is not None:
            self._content_cache[cache_key] = frozenset(tokens)
        return tokens

    def _get_wikidata_bow(self, qid):
        if qid in self._bow_cache:
            return self._bow_cache[qid]
        bow = set()
        for key, values in self.wiki_cache[qid].items():
            if key in {
                "n_statements", "n_sitelinks", "pagerank", "pagerank_wd",
                "indeg", "outdeg", *self._props_to_avoid,
            }:
                continue
            if not (re.match(r"^P[0-9]+$", key) or key in {"description", "first_paragraph"}):
                continue
            if key == "first_paragraph":
                values = [values]
            for value in values:
                value = self._lemma(value)
                tokens = self._treebank_tokenizer.tokenize(value.lower())
                if self._stemmer:
                    tokens = [self._stemmer.stem(t) for t in tokens]
                bow.update(t for t in tokens if re.match("[a-zA-Z]", t))
        self._bow_cache[qid] = bow
        return bow

    def iscore(self, name, article):
        tokens = self._preprocess_content(article["content"], cache_key=article.get("articleID"))
        return np.array([self._iscore_single(name["name"], qid, tokens) for qid in name["ids"]])

    def iscore_narrow(self, name, article):
        text = " ".join(sentences_with_name(name, article["content"]))
        tokens = self._preprocess_content(text)
        return np.array([self._iscore_single(name["name"], qid, tokens) for qid in name["ids"]])

    def _iscore_single(self, name, qid, article_content):
        try:
            article_content = article_content.difference(self._treebank_tokenizer.tokenize(name))
            return len(article_content.intersection(self._get_wikidata_bow(qid)))
        except KeyError:
            return 0

    def _paragraph_content_single(self, qid, article_id):
        if qid not in self.embedding_cache or "first_paragraph" not in self.embedding_cache[qid]:
            return 0
        return cosine(
            mean_vec(self.embedding_cache[qid]["first_paragraph"], axis=1),
            mean_vec(self.content_embeddings[article_id], axis=1),
        )

    def _paragraph_content_narrow_single(self, qid, name, article_id):
        if qid not in self.embedding_cache or "first_paragraph" not in self.embedding_cache[qid]:
            return -1
        return cosine(
            mean_vec(self.embedding_cache[qid]["first_paragraph"], axis=1),
            mean_vec(self.sentence_embeddings[article_id][name], axis=1),
        )

    def _paragraph_or_props_single(self, qid, article_id):
        if qid not in self.embedding_cache or len(self.embedding_cache[qid]) == 0:
            return -1
        if "first_paragraph" in self.embedding_cache[qid]:
            return self._paragraph_content_single(qid, article_id)
        try:
            return cosine(
                stack_mean(self.embedding_cache[qid].values()),
                mean_vec(self.content_embeddings[article_id], axis=1),
            )
        except (NotImplementedError, ValueError, TypeError):
            return -1

    def cse(self, name, article):
        aid = article["articleID"]
        return np.array([self._paragraph_or_props_single(qid, aid) for qid in name["ids"]])

    def _paragraph_or_props_narrow_single(self, qid, name, article_id):
        if qid not in self.embedding_cache or len(self.embedding_cache[qid]) == 0:
            return -1
        if "first_paragraph" in self.embedding_cache[qid]:
            return self._paragraph_content_narrow_single(qid, name, article_id)
        return cosine(
            stack_mean(self.embedding_cache[qid].values()),
            mean_vec(self.sentence_embeddings[article_id][name], axis=1),
        )

    def ncse(self, name, article):
        aid = article["articleID"]
        return np.array([
            self._paragraph_or_props_narrow_single(qid, name["name"], aid) for qid in name["ids"]
        ])


class EntityEntitySimilarityScorer(Scorer):
    def __init__(self, scorer, wiki_cache, embeddings_cache=None, unambiguous_cache=None):
        super().__init__(scorer)
        self.wiki_cache = wiki_cache
        self.embedding_cache = embeddings_cache
        self.unambiguous_cache = unambiguous_cache

    def _unambiguous_qids(self, article_id):
        # Paper code used only the first unambiguous mention as the EEI/CSSVE anchor.
        entries = self.unambiguous_cache[article_id]
        return entries[0] if entries else []

    def cssve(self, name, article):
        return np.array([
            self._matching_attributes_emb_single(qid, article["articleID"]) for qid in name["ids"]
        ])

    def _matching_attributes_emb_single(self, qid, article_id):
        unambiguous_qids = self._unambiguous_qids(article_id)
        if not unambiguous_qids:
            return 0
        score = 0.0
        for unambiguous_qid in unambiguous_qids:
            if qid not in self.embedding_cache:
                continue
            entity_dict = self.embedding_cache[qid]
            for prop, embeddings in self.embedding_cache[unambiguous_qid].items():
                if prop not in entity_dict or prop in {"first_paragraph", "n_statements", "n_sitelinks", "pagerank"}:
                    continue
                prop_scores = []
                for embedding in embeddings:
                    embedding = mean_vec(embedding, axis=0)
                    for entity_embedding in entity_dict[prop]:
                        prop_scores.append(cosine(embedding, mean_vec(entity_embedding, axis=0)))
                score += sum(prop_scores)
        return score

    def eeiscore(self, name, article):
        return np.array([
            self._matching_attributes_single(qid, article["articleID"]) for qid in name["ids"]
        ])

    def _matching_attributes_single(self, qid, article_id):
        unambiguous_qids = self._unambiguous_qids(article_id)
        if not unambiguous_qids:
            return 0
        score = 0
        for unambiguous_qid in unambiguous_qids:
            if unambiguous_qid not in self.wiki_cache or qid not in self.wiki_cache:
                continue
            entity_dict = self.wiki_cache[qid]
            for prop, values in self.wiki_cache[unambiguous_qid].items():
                if prop not in entity_dict or prop in _SKIP:
                    continue
                entity_values = entity_dict[prop]
                if isinstance(entity_values, str):
                    entity_values = [entity_values]
                if isinstance(values, str):
                    values = [values]
                for value in values:
                    for entity_value in entity_values:
                        if entity_value == value:
                            score += 1
        return score

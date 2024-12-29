import gensim.downloader as gensim_loader
import numpy as np

from models import CompoundCombo


class SimilarityScorer:
    MISSING_SCORE = -1

    _embedding_models = {
        'fasttext': 'fasttext-wiki-news-subwords-300',
        'glove': 'glove-wiki-gigaword-300',
        'word2vec': 'word2vec-google-news-300',
    }

    def __init__(self, model):
        if model not in self._embedding_models:
            raise ValueError(f"Model {model} not supported")

        print('Loading gensim model', self._embedding_models[model])
        self.model = gensim_loader.load(self._embedding_models[model])


    def __contains__(self, word: str):
        return word in self.model


    # TODO: consider a better way to handle MWEs; we could average the embeddings or something
    # right now we just return the missing score, since GENSIM doesn't support MWEs

    def score(self, lhs: str, rhs: str) -> float:
        assert lhs in self.model
        return float(self.model.similarity(lhs, rhs)) if rhs in self else self.MISSING_SCORE

    def bulk_score(self, word: str, candidates: list[str]) -> list[float]:
        assert word in self.model
        return [
            float(self.model.similarity(word, candidate)) if candidate in self else self.MISSING_SCORE
            for candidate in candidates
        ]

class W2VecSimilarityScorer(SimilarityScorer):
    def __init__(self):
        super().__init__('word2vec')
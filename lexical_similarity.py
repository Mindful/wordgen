import gensim.downloader as gensim_loader
import numpy as np



class SimilarityScorer:
    _embedding_models = {
        'fasttext': 'fasttext-wiki-news-subwords-300',
        'glove': 'glove-wiki-gigaword-300',
        'word2vec': 'word2vec-google-news-300',
    }

    def __init__(self, model: str = 'word2vec'):
        if model not in self._embedding_models:
            raise ValueError(f"Model {model} not supported")

        print('Loading gensim model', self._embedding_models[model])
        self.model = gensim_loader.load(self._embedding_models[model])


    def __contains__(self, word: str):
        return word in self.model


    def score_compounds_for_word(self, word: str, candidates: list[str], softmax: bool = True) -> np.ndarray:
        scores = np.array([
            self.model.similarity(word, candidate) for candidate in candidates
        ])

        if softmax:
            scores = np.exp(scores) / np.sum(np.exp(scores))

        return scores
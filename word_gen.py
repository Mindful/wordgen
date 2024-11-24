from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable
from itertools import combinations

from conceptnet import load_conceptnet, RELATION_TYPES
from lexical_similarity import SimilarityScorer
from models import CompoundCombo


@dataclass
class WordGenerationConfig:
    max_relation_combos: int = 30
    max_word_combos: int = 100


class WordGenerator:
    def __init__(self, scorer: SimilarityScorer = SimilarityScorer(),
                 conceptnet_path: Path = Path('/home/josh/data/conceptnet'),
                 config: WordGenerationConfig = WordGenerationConfig()):
        self.scorer = scorer
        self.config = config
        print('Loading conceptnet from', conceptnet_path)
        self.conceptnet = load_conceptnet(conceptnet_path)

        self.usable_bases = None
        self.used_combinations = None


    def add_base_word(self, word: str):
        if self.usable_bases is None:
            self.usable_bases = set()
        self.usable_bases.add(word)

    def remove_base_word(self, word: str):
        assert self.usable_bases is not None, "No base words have been added yet"
        self.usable_bases.remove(word)

    def add_used_combination(self, lhs: str, rhs: str):
        # assert sorted - if it's not sorted we might add the same combination twice
        if self.used_combinations is None:
            self.used_combinations = set()
        self.used_combinations.add((lhs, rhs))

    def remove_used_combination(self, lhs: str, rhs: str):
        assert self.used_combinations is not None, "No combinations have been used yet"
        self.used_combinations.remove((lhs, rhs))

    def _get_relation_dict(self, word: str) -> Optional[dict[str, set[str]]]:
        concept_key = f'/c/en/{word}'
        return self.conceptnet.get(concept_key)

    def _clean_filter_rank_concepts(self, base_word: str, concepts: Iterable[str]) -> list[tuple[str, float]]:
        cleaned_iter = (
            concept[len('/c/en/'):] for concept in concepts
        )

        # TODO: consider a better way to handle MWEs; we could average the embeddings or something
        filtered_cleaned = [
            concept for concept in cleaned_iter
            if (self.usable_bases is None or concept in self.usable_bases)
            and concept != base_word
            and concept in self.scorer
        ]
        scores = self.scorer.score_compounds_for_word(base_word, filtered_cleaned)
        return sorted(zip(filtered_cleaned, scores), key=lambda x: x[1], reverse=True)


    @staticmethod
    def _normalize_combination_score(score: int) -> float:
        mx, mn = max(RELATION_TYPES.values()), min(RELATION_TYPES.values())
        return 1 - ((score - mn) / (mx - mn))


    @staticmethod
    def _generate_diagonals(lhs_size: int, rhs_size: int):
        """
        Lazily generate the sequence for two lists of lengths n and m.
        Avoid duplicate pairs and generate in the desired order.
        """
        seen = set()  # To track generated pairs and avoid duplicates
        for d in range(lhs_size + rhs_size - 1):  # Diagonal levels
            for i in range(d + 1):
                j = d - i
                if i < lhs_size and j < rhs_size and (i, j) not in seen:  # Valid pair
                    yield i, j
                    seen.add((i, j))  # Mark as seen
                if j < lhs_size and i < rhs_size and (j, i) not in seen:  # Valid symmetric pair
                    yield j, i
                    seen.add((j, i))


    def _generate_relation_combinations(self, relations_dict: dict[str, set[str]]) -> list[tuple[str, str, float]]:
        return sorted(
            (
                (min(lhs, rhs), max(lhs, rhs),
                 WordGenerator._normalize_combination_score(RELATION_TYPES[lhs] + RELATION_TYPES[rhs]))
                for lhs, rhs in combinations(relations_dict.keys(), 2)
                if lhs != rhs
            ),
            key=lambda x: x[2], reverse=True
        )[:self.config.max_relation_combos]


    def generate_word_combinations(self, word: str) -> list[CompoundCombo]:
        relations_dict = self._get_relation_dict(word)
        relation_combos = self._generate_relation_combinations(relations_dict)

        total = sum(x[2] for x in relation_combos)
        words_for_relation = dict()
        used_word_combos = 0
        output = []
        for lhs, rhs, score in relation_combos:
            allocated_words = min(int(self.config.max_word_combos * (score / total)), self.config.max_word_combos - used_word_combos)
            if allocated_words == 0:
                break

            if lhs not in words_for_relation:
                words_for_relation[lhs] = self._clean_filter_rank_concepts(word, relations_dict[lhs])
            if rhs not in words_for_relation:
                words_for_relation[rhs] = self._clean_filter_rank_concepts(word, relations_dict[rhs])

            lhs_words = words_for_relation[lhs]
            rhs_words = words_for_relation[rhs]

            iterator = self._generate_diagonals(len(lhs_words), len(rhs_words))
            for i in range(allocated_words):
                lhs_idx, rhs_idx = next(iterator, (None, None))
                if lhs_idx is not None and rhs_idx is not None:

                    # ensure the word combos have consistent ordering
                    if lhs_words[lhs_idx][0] < rhs_words[rhs_idx][0]:
                        word1, score1 = lhs_words[lhs_idx]
                        word2, score2 = rhs_words[rhs_idx]
                    else:
                        word1, score1 = rhs_words[rhs_idx]
                        word2, score2 = lhs_words[lhs_idx]

                    used_word_combos += 1
                    output.append(CompoundCombo(
                        base_word=word,
                        word1=word1,
                        word2=word2,
                        relation1=lhs,
                        relation2=rhs,
                        scores={
                            'relation_combo': score,
                            'word_combo': (score1 + score2) / 2
                        }
                    ))

        return sorted(output, key=lambda x: x.cumulative_score, reverse=True)

generator = WordGenerator()

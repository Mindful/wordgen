from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Callable, Union
from itertools import combinations

from conceptnet import load_conceptnet, RELATION_TYPES
from lexical_similarity import W2VecSimilarityScorer
from models import CompoundCombo, Node, State


@dataclass
class WordGenerationConfig:
    max_relation_combos: int = 30
    max_word_combos: int = 100
    score: bool = False
    conceptnet_path: Path = Path('/home/josh/data/conceptnet')
    scorer: type = W2VecSimilarityScorer



class WordGenerator:
    def __init__(self, config: WordGenerationConfig = WordGenerationConfig(), state: Optional[State] = None):
        self.config = config
        self.scorer = self.config.scorer()
        print('Loading conceptnet from', config.conceptnet_path)
        self.conceptnet = load_conceptnet(config.conceptnet_path)
        self.state = state



    def _get_relation_dict(self, word: str) -> Optional[dict[str, set[str]]]:
        concept_key = f'/c/en/{word}'
        return self.conceptnet.get(concept_key)

    def _clean_filter_rank_concepts(self, base_word: str, concepts: Iterable[str]) -> list[tuple[str, Union[float, Callable]]]:
        cleaned_iter = (
            concept[len('/c/en/'):] for concept in concepts
        )

        filtered_cleaned = [
            concept for concept in cleaned_iter
            if (self.state is None or concept in self.state.base_concepts)
            and concept != base_word
        ]
        if self.config.score:
            scores = self.scorer.bulk_score(base_word, filtered_cleaned)
            return sorted(zip(filtered_cleaned, scores), key=lambda x: x[1], reverse=True)
        else:
            return [(concept, lambda: self.scorer.score(base_word, concept)) for concept in filtered_cleaned]


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
        if not relations_dict:
            return []

        relation_combos = self._generate_relation_combinations(relations_dict)

        total = sum(x[2] for x in relation_combos)
        words_for_relation = dict()
        used_word_combos = 0
        output = []
        for lhs, rhs, relation_score in relation_combos:
            allocated_words = min(int(self.config.max_word_combos * (relation_score / total)),
                                  self.config.max_word_combos - used_word_combos)
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

                    already_used = (self.state is not None and (word1, word2) in self.state.used_combinations)
                    if word1 == word2 or already_used:
                        continue

                    used_word_combos += 1
                    output.append(CompoundCombo(
                        represented_concept=word,
                        word1=word1,
                        word2=word2,
                        relation1=lhs,
                        relation2=rhs,
                        relation_score=relation_score,
                        word_1_scorer=score1,
                        word_2_scorer=score2
                    ))

        if self.config.score:
            return sorted(output, key=lambda x: x.average_score, reverse=True)
        else:
            return output
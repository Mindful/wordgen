from dataclasses import field
from functools import cache
from random import Random
from typing import Optional, Iterable
from collections import Counter

from pydantic.dataclasses import dataclass

@dataclass(slots=True)
class CompoundCombo:
    represented_concept: str
    word1: str
    word2: str

    relation1: str
    relation2: str

    scores: dict[str, float]

    @property
    def cumulative_score(self) -> float:
        return sum(self.scores.values()) / len(self.scores)


    def __str__(self):
        # only include word1, word2 and cumulative score
        return f"Combo<{self.word1} {self.word2} ({self.cumulative_score:.2f})>"

    def pretty(self):
        # just show the combination of the two words
        return f"({self.word1}+{self.word2})"


class State:
    __slots__ = ('remaining_concepts', 'generations', 'base_concepts', 'used_combinations', 'rand')

    def __init__(self, base_concepts: Iterable[str], seed: int):
        self.remaining_concepts = set()
        self.generations = set()
        # concept: usage_count
        self.base_concepts = Counter({
            concept: 0 for concept in base_concepts
        })
        self.used_combinations = set()
        self.rand = Random(seed)


    def sample_concepts(self, n: int) -> list[str]:
        # TODO: this might be slow depending on the cost of set -> tuple
        return self.rand.sample(tuple(self.remaining_concepts), n)

@dataclass(slots=True)
class Node:
    parent: Optional['Node']
    processed_concept: Optional[str]  # words that were either combined or chosen as base words
    generation: Optional[CompoundCombo]
    children: list['Node'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0

    def apply(self, state: State):
        state.remaining_concepts.remove(self.processed_concept)
        state.generations.add(self.generation)
        state.base_concepts[self.generation.word1] += 1
        state.base_concepts[self.generation.word2] += 1
        state.used_combinations.add((self.generation.word1, self.generation.word2))

    def undo(self, state: State):
        state.remaining_concepts.add(self.processed_concept)
        state.generations.remove(self.generation)
        state.base_concepts[self.generation.word1] -= 1
        state.base_concepts[self.generation.word2] -= 1
        state.used_combinations.remove((self.generation.word1, self.generation.word2))






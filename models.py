import json
from dataclasses import field, asdict
from pathlib import Path
from random import Random
from typing import Optional, Iterable, Union, Callable, Iterator
from collections import Counter

import numpy as np
from dataclasses import dataclass

@dataclass(slots=True)
class CompoundCombo:
    represented_concept: str
    word1: str
    word2: str

    relation1: str
    relation2: str

    relation_score: float
    word_1_scorer: Union[Callable, float]
    word_2_scorer: Union[Callable, float]


    def __hash__(self):
        return hash((self.word1, self.word2))


    @property
    def average_score(self) -> float:
        return (self.word_1_score + self.word_2_score + self.relation_score) / 3


    @property
    def word_1_score(self) -> float:
        if callable(self.word_1_scorer):
            self.word_1_scorer = self.word_1_scorer()
        return self.word_1_scorer

    @property
    def word_2_score(self) -> float:
        if callable(self.word_2_scorer):
            self.word_2_scorer = self.word_2_scorer()
        return self.word_2_scorer


    def __str__(self):
        return f"Combo<{self.word1} {self.word2} ({self.average_score:.2f})>"

    def pretty(self):
        # just show the combination of the two words
        return f"({self.word1}+{self.word2})"


class State:
    __slots__ = ('remaining_concepts', 'generations', 'base_concepts', 'used_combinations', 'rand', 'target_count')

    def __init__(self, base_concepts: Iterable[str], target_percentage: float, seed: int):
        assert 0 < target_percentage < 1

        self.remaining_concepts = set(base_concepts)
        self.generations = set()
        # concept: usage_count
        self.base_concepts = Counter({
            concept: 0 for concept in base_concepts
        })
        self.used_combinations = set()
        self.rand = Random(seed)
        self.target_count = int(len(self.base_concepts) * target_percentage)


    @property
    def is_terminal(self):
        assert len(self.generations) <= self.target_count
        return len(self.generations) == self.target_count

    def random_iter_concepts(self) -> Iterator[str]:
        # TODO: this might be slow depending on the cost of set -> tuple
        concepts = tuple(self.remaining_concepts)
        indices = list(range(len(concepts)))
        self.rand.shuffle(indices)
        for idx in indices:
            yield concepts[idx]


    def score(self) -> float:
        # TODO: is the penalty not going to just obliterate the score?
        word_scores = sum(x.average_score for x in self.generations)
        concept_usage_penalty = np.square(np.array([x for x in self.base_concepts.values()])).sum()
        return word_scores  / concept_usage_penalty

    def output(self, path: Path):
        path.mkdir()
        with open(path / 'concepts.txt', 'w') as f:
            f.write('\n'.join(self.base_concepts.keys()))

        with open(path / 'generations.jsonl', 'w') as f:
            sorted_generations = sorted(self.generations, key=lambda x: x.average_score, reverse=True)
            f.write('\n'.join(json.dumps(asdict(x)) for x in sorted_generations))

        with open(path / 'generations_simple.txt', 'w'):
            f.write('\n'.join(f'{x.represented_concept} {x.pretty()}' for x in sorted_generations))


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
        assert self.generation not in state.generations
        state.generations.add(self.generation)
        assert self.generation.word1 in state.base_concepts
        assert self.generation.word2 in state.base_concepts
        state.base_concepts[self.generation.word1] += 1
        state.base_concepts[self.generation.word2] += 1
        state.used_combinations.add((self.generation.word1, self.generation.word2))

    def undo(self, state: State):
        state.remaining_concepts.add(self.processed_concept)
        assert self.generation in state.generations
        state.generations.remove(self.generation)
        assert self.generation.word1 in state.base_concepts
        assert self.generation.word2 in state.base_concepts
        state.base_concepts[self.generation.word1] -= 1
        state.base_concepts[self.generation.word2] -= 1
        state.used_combinations.remove((self.generation.word1, self.generation.word2))






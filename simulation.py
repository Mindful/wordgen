from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Iterable, Iterator, Callable, Optional

import spacy
from tqdm import tqdm
from wordfreq import iter_wordlist

from models import State, Node
from word_gen import WordGenerator, WordGenerationConfig
import math


BASE_CONCEPTS_CACHE = Path('base_concepts.txt')


# Load SpaCy model

def pos_lemma(nlp, word: str):
    """
    Scrappy (and not great) heuristic to get the most common POS and lemma for a word
    Wordnet POSs aren't granular enough
    """
    doc = nlp(word)
    if len(doc) == 1:
        token = doc[0]
        return token.pos_, token.lemma_
    else:
        return None, None


"""
1) Sample vocabulary words from the most frequent words
2) Monte Carlo Tree Search to generate word combinations
"""

def get_base_concepts(random_state: Random, generator: WordGenerator, target_count: int, acceptable_pos: Iterable[str] = ('NOUN',)) -> Iterator[str]:
    """Randomly sample from the words in order of frequency. 50% chance of success, and only consider words that can be
    of the POS we're looking for"""

    count = 0
    acceptable_pos = set(acceptable_pos)
    used_lemmas = set()
    nlp = spacy.load("en_core_web_sm")
    progress = tqdm(total=target_count, desc='Sampling base concepts')
    for word in iter_wordlist('en'):
        pos, lemma = pos_lemma(nlp, word)
        if (pos in acceptable_pos
                and lemma not in used_lemmas
                and not lemma.isnumeric()
                and lemma in generator.scorer
                and random_state.random() < 0.5):
            used_lemmas.add(lemma)
            yield lemma
            count += 1
            progress.update(1)
        if count >= target_count:
            progress.close()
            break




def ucb(node: Node) -> float:
    exploration_weight = 1.41

    if node.visits == 0:
        return float('inf')  # Favor unexplored nodes
    exploitation = node.value / node.visits
    exploration = exploration_weight * math.sqrt(math.log(node.parent.visits) / node.visits)
    return exploitation + exploration



class MCTS:

    def _load_base_concepts(self, base_concepts: Optional[int]) -> list[str]:
        if base_concepts is None and BASE_CONCEPTS_CACHE.exists():
            print('Loading base concepts from cache', BASE_CONCEPTS_CACHE, 'use --base_concepts to override')
            with BASE_CONCEPTS_CACHE.open('r') as f:
                concepts = [x.strip() for x in f.readlines()]
        else:
            concepts = list(get_base_concepts(Random(self.seed), self.generator, base_concepts))
            print('Saving base concepts to cache', BASE_CONCEPTS_CACHE)
            with BASE_CONCEPTS_CACHE.open('w') as f:
                f.write('\n'.join(concepts))

        return concepts

    def __init__(self, seed: int,
                 node_score: Callable = ucb,
                 base_concepts: Optional[int] = None,
                 target_percentage: float = 0.3,
                 simulation_depth: Optional[int] = 500):
        self.seed = seed
        self.node_score = node_score
        self.simulation_depth = simulation_depth

        concepts = self._load_base_concepts(base_concepts)
        self.base_state = State(concepts, target_percentage, seed)

        self.generator = WordGenerator(WordGenerationConfig())


    def _traverse_to_leaf(self, root: Node, state: State):
        """Traverse to a leaf node, updating the state to what it should be at that leaf node"""

        node = root
        while node.children:
            # we don't need to apply the first node, because it will be root
            node = max(node.children, key=self.node_score)
            node.apply(state)

        return node

    def _simulate(self, node: Node, state: State) -> float:
        depth = 0
        max_depth = self.simulation_depth
        progress = tqdm(total=max_depth, desc='Simulation', leave=False)
        while (max_depth is None or depth < max_depth) and not state.is_terminal:
            children = self._generate_children(node, state)
            # random in style of MCTS
            # TODO: we could also try some kind of weighted random by scores if we had them
            node = state.rand.choice(children)
            node.apply(state)
            depth += 1
            progress.update(1)

        return state.score()

    def _generate_children(self, node: Node, state: State, sample_j: int = 20, top_k: int = 5) -> list[Node]:
        """Sample J possible target concepts from the current state, generate combinations for each, and then
        return the top K highest scoring combinations"""
        possible_targets = state.sample_concepts(sample_j)
        all_candidates = []
        for target in possible_targets:
            target_candidates = self.generator.generate_word_combinations(target)
            # take the highest scoring combination per word
            if len(target_candidates) != 0:
                all_candidates.append(target_candidates[0])

        output = [Node(node, candidate.represented_concept, candidate) for candidate in all_candidates[:top_k]]
        assert len(output) != 0
        return output

    @staticmethod
    def _backpropagate(node: Node, value: float):
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def _greedy_rollout(self, node: Node, state: State) -> State:
        while not state.is_terminal:
            children = self._generate_children(node, state)
            node = children[0]
            node.apply(state)

        return state

    def run(self, iterations: int) -> State:
        root = Node(None, None, None)

        for _ in tqdm(range(iterations), desc='MCTS'):
            # operation under the assumption that re-copying the base state is a cheaper operation than
            # saving all our mutations and undoing them
            state = deepcopy(self.base_state)

            # SELECTION
            leaf = self._traverse_to_leaf(root, state)  # also updates the state

            # EXPANSION
            children = self._generate_children(leaf, state)
            leaf.children = children

            # SIMULATION
            value = self._simulate(leaf, state)

            # BACKPROPAGATION
            self._backpropagate(leaf, value)

        # GREEDY ROLLOUT
        final_state = deepcopy(self.base_state)
        leaf = self._traverse_to_leaf(root, final_state)
        final_state = self._greedy_rollout(leaf, final_state)

        return final_state




def main():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--base_concepts', type=int, default=None)

    args = parser.parse_args()

    mcts = MCTS(args.seed)
    final_state = mcts.run(1000)



if __name__ == '__main__':
    main()
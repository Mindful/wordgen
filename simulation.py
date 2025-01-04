from argparse import ArgumentParser
from collections import deque
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Iterable, Iterator, Callable, Optional, TypeVar

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
            if len(children) == 0:
                return -float('inf')

            node = state.rand.choice(children)
            node.apply(state)
            depth += 1
            progress.update(1)

        return state.score()

    def  _expand(self, node: Node, state: State) -> bool:
        children = self._generate_children(node, state)
        if len(children) == 0:
            node.value = float('inf')
            return False

        node.children = children
        return True

    def _generate_children(self, node: Node, state: State, count: int = 5) -> list[Node]:
        """Generate _count_ random children"""
        candidates = []
        random_iter = state.random_iter_concepts()

        backup_candidates = deque()
        while len(candidates) < count:
            target = next(random_iter, None)
            if target is None:
                break

            target_candidates = self.generator.generate_word_combinations(target)
            if len(target_candidates) != 0:
                candidates.append(target_candidates[0])
                if len(target_candidates) > 1:
                    backup_candidates.append(target_candidates[1:])

        output = [Node(node, candidate.represented_concept, candidate) for candidate in candidates]
        while len(output) < count and len(backup_candidates) != 0:
            # take the first item from each backup list until we have enough

            cur_backup = backup_candidates.popleft()
            if not isinstance(cur_backup, deque):
                # do this lazily to avoid unnecessary copying
                cur_backup = deque(cur_backup)

            cur_candidate = cur_backup.popleft()
            if len(cur_backup) != 0:
                backup_candidates.append(cur_backup)

            output.append(Node(node, cur_candidate.represented_concept, cur_candidate))

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
            if len(children) == 0:
                # we can't progress any further
                # TODO: try and deal with this by somehow backtracking?
                break
            node = children[0]
            node.apply(state)

        return state

    def run(self, iterations: int) -> State:
        root = Node(None, None, None)

        for _ in tqdm(range(iterations), desc='MCTS'):
            # operation under the assumption that re-copying the base state is a cheaper operation than
            # saving all our mutations and undoing them
            state = deepcopy(self.base_state)
            self.generator.state = state

            # SELECTION
            leaf = self._traverse_to_leaf(root, state)  # also updates the state

            # EXPANSION
            can_progress = self._expand(leaf, state)
            if not can_progress:
                # TODO: log warning about this?
                continue

            # SIMULATION
            retries = 0
            value = -float('inf')

            while value == -float('inf') and retries < 5:
                # simulation can fail due to being unable to generate any children, so we retry
                # it a few times before giving up if necessary
                value = self._simulate(leaf, state)
                retries += 1

            if value == -float('inf'):
                # TODO: log a warning here?
                # don't backprop if we failed to simulate
                continue

            # BACKPROPAGATION
            self._backpropagate(leaf, value)

        # GREEDY ROLLOUT
        final_state = deepcopy(self.base_state)
        self.generator.state = final_state
        leaf = self._traverse_to_leaf(root, final_state)
        final_state = self._greedy_rollout(leaf, final_state)

        return final_state




def main():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--base_concepts', type=int, default=None)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=Path('results'))

    args = parser.parse_args()
    assert not args.output.exists(), 'Output directory already exists'

    mcts = MCTS(args.seed)
    print('Init with', len(mcts.base_state.base_concepts), 'concepts and a target of', mcts.base_state.target_count, 'generations')
    final_state = mcts.run(args.iterations)
    if not final_state.is_terminal:
        print('Failed to reach terminal state')

    print('Final score of', final_state.score())
    print('Writing results to', f'{args.output}/')
    final_state.output(args.output)



if __name__ == '__main__':
    main()
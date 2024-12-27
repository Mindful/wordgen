from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Iterable, Iterator, Callable

import spacy
from tqdm import tqdm
from wordfreq import iter_wordlist

from models import State, Node
from word_gen import WordGenerator


# TODO: SCORE THE STATE, NOT INDIVIDUAL CHOICES - this is how we have to do MCTS for it to make sense
# (otherwise for example if we have dependent scores, somethign that scores highly early on may look better
# than it really is because it lowers the score later)


# TODO: it's theoretically possible to score even the word combinations lazily if we were willing to pick them totally
# at random. MCTS could then be more explorative... but I'm not sure if this is a good idea.


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


def traverse_to_leaf(state: State, root: Node, scoring_func: Callable):
    """Traverse to a leaf node, updating the state to what it should be at that leaf node"""

    node = root
    while node.children:
        # we don't need to apply the first node, because it will be root
        node = max(node.children, key=scoring_func)
        node.apply(state)

    return node


def generate_children(generator: WordGenerator, node: Node, sample_j: int = 20, top_k: int = 5) -> list[Node]:
    """Sample J possible target concepts from the current state, generate combinations for each, and then
    return the top K highest scoring combinations"""
    possible_targets = generator.state.sample_concepts(sample_j)
    all_candidates = []
    for target in tqdm(possible_targets, desc='Generating children', leave=False):
        target_candidates = generator.generate_word_combinations(target)
        # take the highest scoring combination per word
        all_candidates.append(target_candidates[0])

    all_candidates.sort(key=lambda x: x.cumulative_score, reverse=True)
    return [Node(node, candidate.represented_concept, candidate) for candidate in all_candidates[:top_k]]




def mcts(base_state: State, iterations: int, scoring_func: Callable):
    root = Node(None, None, None)

    for _ in tqdm(range(iterations), desc='MCTS'):
        state = deepcopy(base_state)  # TODO might not need to do this if we unpack it every time?

        # SELECTION
        leaf = traverse_to_leaf(state, root, scoring_func)  # also updates the state

        # EXPANSION
        children = generate_children(generator, leaf)
        leaf.children = children

        # SIMULATION






def main():
    default_base_concepts = 3000
    base_concepts_cache = Path('base_concepts.txt')

    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--base_concepts', type=int, default=None)
    parser.add_argument('--target_combination_percentage', type=float, default=0.3)

    args = parser.parse_args()
    generator = WordGenerator()

    if args.base_concepts is None and base_concepts_cache.exists():
        print('Loading base concepts from cache', base_concepts_cache, 'use --base_concepts to override')
        with base_concepts_cache.open('r') as f:
            concepts = [x.strip() for x in f.readlines()]
    else:
        concepts = list(get_base_concepts(Random(args.seed), generator, args.base_concepts or default_base_concepts))
        print('Saving base concepts to cache', base_concepts_cache)
        with base_concepts_cache.open('w') as f:
            f.write('\n'.join(concepts))


    state = State(concepts, args.seed)
    generator.state = state
    generator.config.score = False

    for concept in concepts[:20]:
        best_candidate = generator.generate_word_combinations(concept)
        if len(best_candidate) == 0:
            continue
        else:
            best = best_candidate[0]
            print(concept, best.pretty(), best.average_score)





if __name__ == '__main__':
    main()
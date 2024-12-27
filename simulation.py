from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Iterable, Iterator

import spacy
from tqdm import tqdm
from wordfreq import iter_wordlist

from models import State, Node
from word_gen import WordGenerator


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


def traverse_to_leaf(state: State, root: Node, scoring_func: callable):
    """Traverse to a leaf node, updating the state to what it should be at that leaf node"""

    node = root
    while node.children:
        # we don't need to apply the first node, because it will be root
        node = max(node.children, key=scoring_func)
        node.apply(state)

    return node


def generate_children(generator: WordGenerator, node: Node, sample_j: int = 20, top_k: int = 5):
    """Sample J possible target concepts from the current state, generate combinations for each, and then
    use the top K combinations to create children nodes"""
    possible_targets = generator.state.sample_concepts(sample_j)
    all_candidates = []
    for target in tqdm(possible_targets, desc='Generating children', leave=False):
        target_candidates = generator.generate_word_combinations(target)
        # take the highest scoring combination per word
        all_candidates.append(target_candidates[0])

    all_candidates.sort(key=lambda x: x.cumulative_score, reverse=True)
    for candidate in all_candidates[:top_k]:
        node.children.append(Node(node, candidate.represented_concept, candidate))




def mcts(base_state: State, iterations: int, scoring_func: callable):
    root = Node(None, None, None)

    for _ in tqdm(range(iterations), desc='MCTS'):
        state = deepcopy(base_state)  # TODO might not need to do this if we unpack it every time?
        leaf = traverse_to_leaf(state, root, scoring_func)





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

    for concept in concepts[:20]:
        best_candidate = generator.generate_word_combinations(concept)
        if len(best_candidate) == 0:
            continue
        else:
            print(concept, best_candidate[0].pretty())





if __name__ == '__main__':
    main()
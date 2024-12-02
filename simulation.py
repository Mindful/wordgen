from random import Random
from typing import Iterable, Iterator

from tqdm import tqdm

from lexical_similarity import SimilarityScorer
from wordfreq import iter_wordlist

from argparse import ArgumentParser
import spacy

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



def main():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--base_concepts', type=int, default=3000)
    parser.add_argument('--target_combination_percentage', type=float, default=0.3)

    args = parser.parse_args()
    random = Random(args.seed)
    generator = WordGenerator()

    concepts = list(get_base_concepts(random, generator, args.base_concepts))
    for concept in concepts:
        generator.add_base_word(concept)

    for concept in concepts[:20]:
        best_candidate = generator.generate_word_combinations(concept)
        if len(best_candidate) == 0:
            continue
        else:
            print(concept, best_candidate[0].pretty())





if __name__ == '__main__':
    main()
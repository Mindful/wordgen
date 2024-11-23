from argparse import ArgumentParser
from collections import defaultdict
from itertools import takewhile, repeat
from pathlib import Path
from csv import reader, writer
from typing import Union

from tqdm import tqdm

# https://github.com/commonsense/conceptnet5/wiki/Downloads
# URI   RELATION    START   END     JSON_METADATA

BANNED = 100
d = 5
RELATION_TYPES = {
    '/r/RelatedTo': 3,
    '/r/FormOf': 2,
    '/r/IsA': 1,
    '/r/PartOf': 2,
    '/r/HasA': 3,
    '/r/UsedFor': 1,
    '/r/CapableOf': 2,
    '/r/AtLocation': 2,
    '/r/Causes': 2,
    '/r/HasSubevent': 4,
    '/r/HasFirstSubevent': 4,
    '/r/HasLastSubevent': 4,
    '/r/HasPrerequisite': 4,
    '/r/HasProperty':2,
    '/r/MotivatedByGoal': 3,
    '/r/ObstructedBy': BANNED,
    '/r/Desires': 4,
    '/r/CreatedBy': 4,
    '/r/Synonym': 2,
    '/r/Antonym': BANNED,
    '/r/DistinctFrom': BANNED,
    '/r/DerivedFrom': BANNED,
    '/r/SymbolOf': 4,
    '/r/DefinedAs': 3,
    '/r/MannerOf': 4,
    '/r/LocatedNear': 3,
    '/r/HasContext': 4,
    '/r/SimilarTo': 4,
    '/r/EtymologicallyRelatedTo': BANNED,
    '/r/EtymologicallyDerivedFrom': BANNED,
    '/r/CausesDesire': BANNED,
    '/r/MadeOf': 1,
    '/r/ReceivesAction': 2,
    '/r/ExternalURL': BANNED,
    '/r/InstanceOf': 1
}
RTYPE_TO_IDX = {r: i for i, r in enumerate(RELATION_TYPES.keys())}
IDX_TO_RTYPE = {i: r for i, r in enumerate(RELATION_TYPES.keys())}

ASSERTIONS_FILE = 'assertions.csv'
FILTERED_FILE  = 'filtered.csv'

# https://stackoverflow.com/a/27518377/4243650
def fast_linecount(filename: Union[str, Path]) -> int:
    if isinstance(filename, str):
        filename = Path(filename)

    with filename.open('rb') as f:
        bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
        linecount = sum(buf.count(b"\n") for buf in bufgen)

    return linecount


def load_conceptnet(path: Path):
    conceptnet = defaultdict(lambda: defaultdict(set))

    requirements = [
        lambda x: x[2].startswith('/c/en/'),
        lambda x: x[3].startswith('/c/en/'),
        lambda x: RELATION_TYPES.get(x[1], BANNED) != BANNED,
    ]

    if (path / FILTERED_FILE).exists():
        print('Found filtered file at', path / FILTERED_FILE, 'reading from it')
        input_file = (path / FILTERED_FILE)
        output_file = None
    else:
        print('Found assertion file at', path / ASSERTIONS_FILE,
              'writing filtered output to', path / FILTERED_FILE)
        input_file = (path / ASSERTIONS_FILE)
        output_file = (path / FILTERED_FILE).open('w')
        output_file = writer(output_file, delimiter='\t')

    with input_file.open('r') as file_contents:
        for row in tqdm(reader(file_contents, delimiter='\t'), total=fast_linecount(input_file)):
            if not all(r(row) for r in requirements):
                continue

            conceptnet[row[2]][row[1]].add(row[3])
            if output_file is not None:
                output_file.writerow(row)

    print('Loaded', len(conceptnet), 'concepts')
    return conceptnet


def main():
    parser = ArgumentParser()
    parser.add_argument("input", type=Path)

    args = parser.parse_args()
    conceptnet = load_conceptnet(args.input)


if __name__ == '__main__':
    main()





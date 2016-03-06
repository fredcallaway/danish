"""Creates LENS training and testing files"""
from __future__ import division, print_function
import csv
import random
from functools import lru_cache

def get_corpus(lang, word_boundaries=True):
    """Returns corpus as continuous string of phonemes and utterance boundaries"""
    with open('corpora/%s-corpus.txt' % lang, 'r') as corpus:
        corpus = (line[1:-1] for line in corpus)  # remove leading Q and trailing \n
        corpus = ''.join(corpus)
    if not word_boundaries:
        corpus = corpus.replace('X', '')

    # Remove adjacent boundaries.
    corpus = corpus.replace('QX', 'Q')
    corpus = corpus.replace('XQ', 'Q')
    corpus = corpus.replace('QQ', 'Q')
    corpus = corpus.replace('\r', '')  # windows line endings... yuck
    return corpus


@lru_cache(None)
def get_encoding(file='encodings/distributed.csv', distributed=True):
    """Returns phoneme-unit mappings for the given language."""
    with open(file,'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        if distributed:
            return {row[0]: row[1:] for row in reader}
        else:
            alphabet = [row[0] for row in reader]
            return localist_encoding(alphabet)

def localist_encoding(alphabet):
    """Returns an encoding with each phoneme mapped to a single unit."""
    encoding = {}
    base = '0' * len(alphabet)
    for idx, item in enumerate(alphabet):
        encoding[item] = base[:idx] + '1' + base[idx+1:]
    return encoding


def train_test_split(corpus, num_train, num_test, mode='end'):
    """Returns (train, test) a division of a corpus."""
    corpus = list(corpus)
    total = num_train + num_test
    if len(corpus) < total:
        raise ValueError('Corpus is too short!')
    corpus = corpus[:total]
    if mode == 'random':
        indices = random.sample(range(len(corpus)), num_test)
        train = corpus
        test = [train.pop(i) for i in sorted(indices)]

    elif mode == 'begin':
        test, train = corpus[:num_test], corpus[num_test:]
        
    elif mode == 'end':
        train, test = corpus[:-num_test], corpus[-num_test:]
    
    return train, test


def summarize_corpus(lang):
    word_corpus = get_corpus(lang, word_boundaries=True)
    num_phonemes = len(get_corpus(lang).replace('Q', ''))

    return {'length': num_phonemes,
            'avg_word_length': num_phonemes / word_corpus.count('X'),
            'avg_utterance_length': num_phonemes / word_corpus.count('Q')}


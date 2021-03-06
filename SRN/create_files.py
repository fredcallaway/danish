"""Creates LENS training and testing files"""
from __future__ import division, print_function
import csv
import random

def get_corpus(lang, word_boundaries=False):
    """Returns (str): corpus as continuous string of phonemes and utterance boundaries"""
    with open('corpora/%s-corpus.txt' % lang, 'r') as corpus:
        if lang is 'cas98':
            corpus = (line[:line.find('\t')] for line in corpus)  # remove stress
            if word_boundaries:
                corpus = 'X'.join(corpus)
                corpus = corpus.replace('#X', '#')
                corpus = corpus.replace('X#', '#')
            else:
                corpus = ''.join(corpus)
            corpus = corpus.replace('/', '')
            corpus = corpus.replace('E', 'e')

        else:
            corpus = (line[1:-1] for line in corpus)  # remove leading Q and trailing \n
            corpus = ''.join(corpus)
            if not word_boundaries:
                corpus = corpus.replace('X', '')
            corpus = corpus.replace('QX', 'Q')
            corpus = corpus.replace('XQ', 'Q')
            corpus = corpus.replace('\r', '')
            corpus = corpus.replace('QQ', 'Q')
        return corpus


def summarize_corpus(lang):
    # DEPRECATED
    word_corpus = get_corpus(lang, word_boundaries=True)
    num_phonemes = len(get_corpus(lang).replace('Q', ''))

    return {'length': num_phonemes,
            'avg_word_length': num_phonemes / word_corpus.count('X'),
            'avg_utterance_length': num_phonemes / word_corpus.count('Q')}

def avg_word_length(lang):
    word_corpus = get_corpus(lang, word_boundaries=True)
    num_phonemes = len(get_corpus(lang).replace('Q', ''))
    return num_phonemes / word_corpus.count('X')

# def convertSampa(corpus):
#     """Convert Danish SAMPA to our SAMPA and remove word boundaries"""
#     corpus = corpus.replace('0', '@')
#     corpus = corpus.replace('U', 'V')
# corpus = corpus.replace('Y', '2')
# other stuff
#     return corpus

# def convertArpabet(corpus):
#     """convert arpabet to our SAMPA"""
#     with open('arpabet.csv','r')  as f:
#         reader = list(csv.reader(f))
#         arpadict= {row[0]:row[1:] for row in reader[1:]}
# todo: how is the corpus formated?
#     return corpus

def get_encodings(lang, distributed):
    """Returns input and output neuron mappings for lang."""

    if lang is 'cas98':
        if distributed:
            # read distributed encodings
            with open('encodings/cas98-distributed.csv', 'r') as f:
                reader = list(csv.reader(f))
                distributed_encoding = {row[0]: '0 ' + str(row[1:]).translate(
                                          None, ",[]'") for row in reader[1:]}
                distributed_encoding['#'] = '1' + ' 0' * len(reader[0][1:])

        # read localist encodings (output is always localist)
        with open('encodings/cas98-distributed.csv', 'r') as f:
            reader = list(csv.reader(f))
            localist_encoding = {}
            for i, row in enumerate(reader[1:]):
                localist_encoding[row[0]] = str(i + 1)
                localist_encoding['#'] = '0'

    else:
        with open('encodings/distributed.csv','r')  as f:
            reader = list(csv.reader(f))
        distributed_encoding = {}
        localist_encoding = {}
        for i, row in enumerate(reader[1:]):
            symbol = row[0]
            localist_encoding[symbol] = str(i + 1)
            encoding = ['0'] + row[1:]  # assumes encoding begins at 2nd col
            encoding = ' '.join(encoding)
            distributed_encoding[symbol] = encoding

        distributed_encoding['Q'] = '1'
        localist_encoding['Q'] = '0'

    incoding = distributed_encoding if distributed else localist_encoding
    outcoding = localist_encoding
    return (incoding, outcoding)



def train_test_split(corpus, num_train, num_test, mode='end'):
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


def neighbors(iterable, n=2):
    """Iterates through adjacent groups in the iterable.

    neighbors([1,2,3,4], n=3) -> [1,2,3], [2,3,4]
    """
    return (iterable[i:i+n] for i in range(len(iterable) - n))


def create_training_files(lang, corpus, num_train, num_test,
                          incoding, outcoding, distributed):
    """Populates lens/ with .ex files for training and testing segmentation

    Returns (int): the number of training examples.
    """
    train, test = train_test_split(corpus, num_train, num_test)

    test_file = 'lens/test.ex'
    with open(test_file, 'w+') as f:
        for current, nxt, in neighbors(test):
            write_example(f, current, nxt, incoding, outcoding, distributed)

    train_file = 'lens/train.ex'
    with open(train_file, 'w+') as f:
        for current, nxt, in neighbors(train):
            write_example(f, current, nxt, incoding, outcoding, distributed)


def creat_exp_files(lang, incoding, outcoding, distributed):
    """Populates lens/ with .ex files for the experiment"""

    for exp in ('A', 'B'):
        # write training file
        with open('experiment/{0}/train{1}.txt'.format(lang, exp), 'r') as f:
            exp_training = f.read()
        train_file = 'lens/exp-train%s.ex' % exp
        with open(train_file, 'w+') as f:
            k = 0
            while True:  # use all examples
                i = exp_training[k]
                try:
                    t = exp_training[k + 1]
                except IndexError:
                    # no more examples
                    break
                write_example(f, i, t, incoding, outcoding, distributed)
                k += 1

        # create test files
        with open('experiment/{0}/test{1}.txt'.format(lang, exp), 'r') as f:
            trials = f.read()
        trials = trials.replace('\r', '')
        trials = trials.replace('\n', '')
        trials = trials.replace('QQ', 'Q')
        trials = trials.split('Q')[1:-1]
        trials = [t + 'Q' for t in trials]
        # if this changes, make sure to change all 72's in project files
        assert len(trials) is 72
        for k, trial in enumerate(trials):
            with open('lens/exp-test%s%s.ex' % (exp, k), 'w+') as f:
                for j in range(len(trial) - 1):
                    i = trial[j]
                    t = trial[j + 1]
                    write_example(f, i, t, incoding, outcoding)


def fmt_example(i, t, incoding, outcoding, distributed=False):
    """Writes one example to a .ex file"""
    result = []
    result.append('name: {%s -> %s}' % (i, t))
    result.append('1')
    code = 'I' if distributed else 'i'
    result.append('{code}: {incoding[i]} t: {outcoding[t]};'.format(**locals()))
    return '\n'.join(result)

def write_example(f, i, t, incoding, outcoding, distributed=False):
    """Writes one example to a .ex file"""
    f.write('name: {%s -> %s}\n' % (i, t))
    f.write('1\n')
    # upper case letter indicates dense encoding
    if distributed:
        f.write('I: ' + incoding[i] + ' t: ' + outcoding[t] + ';\n')
    else:
        f.write('i: ' + incoding[i] + ' t: ' + outcoding[t] + ';\n')


def write_example_files(lang, distributed, num_train, num_test=1000):
    """Populates lens/ with .ex files for training, testing, and experiment"""
    corpus = get_corpus(lang)
    incoding, outcoding = get_encodings(lang, distributed)
    create_training_files(lang, corpus, num_train, num_test,
                          incoding, outcoding, distributed)
    creat_exp_files(lang, incoding, outcoding, distributed)

    return incoding, outcoding

if __name__ == '__main__':
    # for debugging
    write_example_files('danish', False, 1000)
    #print(summarize_corpus('english'))
    #print(summarize_corpus('danish'))

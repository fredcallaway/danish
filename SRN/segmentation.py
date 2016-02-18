"""Compares a list of word boundary predictions to true word boundaries"""
from __future__ import division, print_function
from collections import Counter
import joblib

def get_predicted_word_boundaries(net_out_file, algorithm, lang=None):
    """Returns predicted word boundaries based on boundary unit activation"""
    # get output for neuron 0
    with open(net_out_file, 'r') as f:
        net_out = f.readlines()
    frames = [line.split(' ')[:-1] for line in net_out[1:]]
    frames = [ [float(n) for n in f] for f in frames ]
    break_out = [f[0] for f in frames]
    print(joblib.dump(break_out, 'pickles/%s_break' % lang))

    if algorithm == 'break_average':
        # break activation > average activation of the break unit
        threshold = sum(break_out) / len(break_out)

    elif algorithm == 'total_average':
        # break activation > average activation for all units
        total_activation = 0
        count = 0
        for line in net_out:
            if line[1] == '.':
                total_activation += float(line.split(' ')[0])
                count += 1
        threshold = total_activation / count

    else:
        raise ValueError('Invalid algorithm!')

    predicted_boundaries = [True if activation > threshold else False
                            for activation in break_out]
    return predicted_boundaries


def get_correct_word_boundaries(lang, num_examples):
    from create_files import get_corpus
    corpus = get_corpus(lang, word_boundaries=True)
    if corpus[0] == 'X':
        # causes a bug in indexing when testing segmentation
        corpus = corpus[1:]
    # extract boundaries
    correct_boundaries = []
    i = 1  # i is the index of the *next* character
    while len(correct_boundaries) < num_examples - 1:
        if corpus[i] in 'XQ#':
            correct_boundaries.append(True)
            if corpus[i] == 'X':
                i += 1  # skip the Xs
        else:
            correct_boundaries.append(False)
        i += 1
    correct_boundaries.append(True)  # because the last char is followed word boundary

    assert len(correct_boundaries) == num_examples
    print(joblib.dump(correct_boundaries, 'pickles/%s_boundaries' % lang))
    return correct_boundaries


def get_random_boundaries(lang, num_examples):

    from create_files import avg_word_length
    from random import random
    threshold = 1 / avg_word_length(lang)
    return [True if random() > threshold else False
            for _ in range(num_examples)]


def test_boundary_prediction(correct_boundaries, predicted_boundaries):
    assert len(correct_boundaries) == len(predicted_boundaries)

    result_map = {(True, True): 'hit',
                  (True, False): 'alarm',
                  (False, True): 'miss',
                  (False, False): 'reject'}

    results = Counter(result_map[(p, c)] for p, c in
                      zip(predicted_boundaries, correct_boundaries))

    if results['hit'] == 0:
        # avoid zero division
        precision = 0
        recall = 0
        F = 0
    else:
        precision = results['hit'] / (results['hit'] + results['alarm'])
        recall = results['hit'] / (results['hit'] + results['miss'])
        F = 2 * (precision * recall) / (precision+recall)

    return {'boundary_precision': precision,
            'boundary_recall': recall,
            'boundary_F': F}


def test_word_segmentation(correct_boundaries, predicted_boundaries):
    # get words as tuples of beginning and end indices
    predicted_words = []
    last_boundary = 0
    for i, prediction in enumerate(predicted_boundaries):
        if prediction:
            predicted_words.append((last_boundary,i))
            last_boundary = i

    correct_words = []
    last_boundary = 0
    for i, boundary in enumerate(correct_boundaries):
        if boundary:
            correct_words.append((last_boundary,i))
            last_boundary = i

    hits = sum(1 for x in predicted_words if x in correct_words)
    alarms = sum(1 for x in predicted_words if x not in correct_words)
    misses = sum(1 for x in correct_words if x not in predicted_words)
    
    if hits == 0:
        # avoid zero division
        precision = 0
        recall = 0
        F = 0
    else:
        precision = hits / (hits+alarms)
        recall = hits / (hits+misses)
        F = 2 * (precision * recall) / (precision+recall)

    return {'word_precision': precision,
            'word_recall': recall,
            'word_F': F}


def test_segmentation(lang, boundary_algorithm):
    net_out_file = 'lens/segmentation.out'

    predicted_boundaries = get_predicted_word_boundaries(net_out_file, boundary_algorithm, lang)
    correct_boundaries = get_correct_word_boundaries(lang, len(predicted_boundaries))

    random_boundaries = get_random_boundaries(lang, len(predicted_boundaries))
    
    print(test_boundary_prediction(correct_boundaries, random_boundaries))
    print(test_boundary_prediction(correct_boundaries, predicted_boundaries))
    
    results = test_boundary_prediction(correct_boundaries, predicted_boundaries)
    results.update(test_word_segmentation(correct_boundaries, predicted_boundaries))

    return results


if __name__ == '__main__':
    # get_predicted_word_boundaries('english/english.out','break_average')
    # get_correct_word_boundaries('english',1000)
    print(test_segmentation('danish','break_average'))

"""Compares a list of word boundary predictions to true word boundaries"""
from __future__ import division, print_function
from collections import Counter
import joblib
import numpy as np

import utils

def extract_boundaries(corpus):
    assert 0
    corpus = iter(corpus)
    if next(corpus) == 'X':
        next(corpus)  # skip a leading X
    
    for phone in corpus:
        if phone == 'X':
            yield 1
            next(corpus)  # skip because input doesn't have X's
        elif phone == 'Q':
            yield 1
        else:
            yield 0
    yield 1  # assume the corpus ends at a word boundary

def test_extract_boundaries():
    out = list(extract_boundaries('XABXDEFXHXIJKLX'))
    correct = list(map(int, '0100110001'))
    if out != correct:
        print('ERROR', out)
        print('     ', correct)


def get_random_boundaries(lang, num_examples):
    assert 0  # DEPRECATED
    from create_files import avg_word_length
    from random import random
    threshold = 1 / avg_word_length(lang)
    return [True if random() > threshold else False
            for _ in range(num_examples)]


def get_predicted_word_boundaries(break_out):
    """Returns predicted word boundaries based on boundary unit activation"""
    threshold = np.sum(break_out) / len(break_out)
    predicted_boundaries = break_out > threshold
    return predicted_boundaries


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


def segmentation(row):
    correct_boundaries = row.test_bounds
    break_out = row.test_outputs[:, -1]
    boundary_auc = metrics.roc_auc_score(row.test_bounds, break_out)

    predicted_boundaries = get_predicted_word_boundaries(break_out)
    boundary_results = test_boundary_prediction(correct_boundaries, predicted_boundaries)
    word_results = test_word_segmentation(correct_boundaries, predicted_boundaries)

    return {'boundasy_auc': boundary_auc,
            **boundary_results,
            **word_results}



def test_word_segmentation(correct_boundaries, predicted_boundaries):
    # get words as tuples of beginning and end indices
    correct_indices = np.nonzero(correct_boundaries)[0]
    predicted_indices = np.nonzero(predicted_boundaries)[0]
    correct_words = set(map(tuple, utils.neighbors(correct_indices)))
    predicted_words = set(map(tuple, utils.neighbors(predicted_indices)))
    joblib.dump((correct_words, predicted_words), 'words.pkl')
    hits = len(correct_words & predicted_words)
    alarms = len(predicted_words - correct_words)
    misses = len(correct_words - predicted_words)

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


def test_segmentation(break_out, boundaries):
    predicted_boundaries = get_predicted_word_boundaries(break_out)
    
    #random_boundaries = get_random_boundaries(lang, len(predicted_boundaries))
    #print(test_boundary_prediction(correct_boundaries, random_boundaries))
    #print(test_boundary_prediction(correct_boundaries, predicted_boundaries))
    
    results = test_boundary_prediction(boundaries, predicted_boundaries)
    #results.update(test_word_segmentation(boundaries, predicted_boundaries))

    return results


if __name__ == '__main__':
    # get_predicted_word_boundaries('english/english.out','break_average')
    # get_correct_word_boundaries('english',1000)
    #print(test_segmentation('danish','break_average'))
    test_get_correct_word_boundaries()

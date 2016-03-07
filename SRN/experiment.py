"""Evaluates net performance on an experiment"""
from __future__ import division

import random
import logging

import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def logistic(x, steepness=1):
    x = np.array(x)
    return 1. / (1. + np.exp(-steepness * (x)))

def get_network_choices(trial_errors):
    """Returns ([int], [float]): choices and reaction time
    choices is a list of 0 or 1 corresponding to word indices in one trial
    reaction time is a list of floats

    Args:
      trial_errors [(float, float)]: each tuple represents
        the reaction times for the two words in a trial
    """
    # pairs of consecutive words correspond to trials
    

    def choose(pair):
        threshold = pair[0]/sum(pair)

        threshold = logistic(pair[0] - pair[1], 0.1)
        #print('thresh', threshold)

        # p(choose word1) = error2/(error1+error2)
        choice =  0 if random.random() > threshold else 1
        
        #choice =  0 if 0.5 > threshold else 1

        # reaction time is inverse of percentage difference between values
        rt = 1 / (abs(pair[0]-pair[1]) / (sum(pair)/2))

        return (choice, rt)

    results = [choose(p) for p in trial_errors]
    choices = [r[0] for r in results]
    reaction_times = [r[1] for r in results]
    return choices, reaction_times


def get_correct_choices():
    """Returns [int]: index of correct word for each trial"""
    with open('experiment/answer-key.txt', 'r') as f:
        key = f.read()
    return [int(k) for k in key.split('\n')]


def test_word_choices(choices):
    """Returns [int]: indices of trials net was incorrect for

    Args:
      choices: [int] network's word choices, 1 or 0"""
    key = get_correct_choices()
    assert len(choices) == len(key)
    incorrect = []
    for i in range(len(key)):
        if choices[i] != key[i]:
            incorrect.append(i)
    return incorrect
    

# import cPickle as pickle
import pickle
from glob import glob
import pandas as pd
import seaborn as sns
import sys
import os
import random
import np
import numpy as np

sys.path.append('../SRN')
from network import Network  # for pickle

sns.set(context='notebook', style='whitegrid', palette='muted', font_scale=1.2)
sns.plt.switch_backend('TkAgg')  # MacOS backend doesn't work as well


def create_csv(file_name='net-log.csv', columns=None, rounding=3):
    """Creates a .csv file for all nets in nets/ with parameters and results"""
    df = get_data_frame(columns=columns)
    df.to_csv(file_name, index=False)


def get_data_frame(columns=None):
    """Returns a pandas data frame for nets in ../SRN/nets/"""

    default_cols = ['time', 'lang', 'distributed', 'seed', 'word_F',
                    'contoid_accuracy', 'vocoid_accuracy']

    all_cols = ['time', 'lang', 'distributed', 'seed', 'hidden', 'rate', 'momentum',
                'ticks', 'rand_range', 'num_train', 'boundary_precision',
                'boundary_recall', 'boundary_F', 'word_precision', 'word_recall',
                'word_F', 'contoid_accuracy', 'vocoid_accuracy', 'contoid_reaction',
                'vocoid_reaction', 'contoid_std', 'vocoid_std', 'contoid_rts',
                'vocoid_rts', 'contoid_errors', 'vocoid_errors']

    if columns is None:
        cols = default_cols
    elif columns == 'all':
        cols = all_cols
    else:
        cols = default_cols + columns
        cols = sorted(cols, key=all_cols.index)

    d = {col: [] for col in cols}  
    # for each net, append each attribute to the respective column
    for net_file in glob('../SRN/nets/*.p'):
        net = pickle.load(open(net_file))
        for attr in d.keys():
            d[attr].append(getattr(net, attr, None))

    df = pd.DataFrame.from_dict(d)
    df = df.reindex_axis(cols, axis=1)
    return df


def aov(expression, data):
    lm = ols(experssion, data=data).fit()
    anova = sm.stats.anova_lm(lm, typ=2)
    return anova


def fix_column_labels(df):
    """Fixes column labels in place"""
    replacements = {'lang': 'language', '_': ' ', 'reaction': 'reaction time',
                    'F': 'segmentation F-score'}
    new_labels = []
    for label in df.columns:
        for old, new in replacements.items():
            label = label.replace(old, new)
        label = label.capitalize()
        new_labels.append(label)
    df.columns = new_labels


def plot_experimental_accuracies(df):
    df = df[['Language', 'Distributed', 'Contoid accuracy', 'Vocoid accuracy']]
    df = pd.melt(df, ['Language', 'Distributed'], var_name='Condition', value_name='Accuracy')

    g = sns.factorplot('Language', 'Accuracy', hue='Condition', col='Distributed',
                       data=df, kind='bar')
    g.despine(left=True)
    fig_path = 'figs/experimental_accuracies.png'
    sns.plt.savefig(fig_path)
    os.system('open ' + fig_path)


def plot_reaction_times(df):
    df = df[['Language', 'Distributed', 'Contoid reaction time', 'Vocoid reaction time']]
    df = pd.melt(df, ['Language', 'Distributed'],
                 var_name='Condition', value_name='Reaction time')
    print len(df)
    df = df[(df['Reaction time'] - df['Reaction time'].mean()).abs() <= (2 * df['Reaction time'].std())]
    print len(df)

    g = sns.factorplot('Language', 'Reaction time', hue='Condition', col='Distributed',
                       data=df, kind='bar')
    g.despine(left=True)
    fig_path = 'figs/reaction_times.png'
    sns.plt.savefig(fig_path)
    os.system('open ' + fig_path)



def plot_word_segmentation(df):
    df = df[['Language', 'Distributed', 'Word segmentation f-score']]

    g = sns.factorplot('Language', 'Word segmentation f-score', hue='Language',
                       col='Distributed', data=df, kind='bar')
    g.despine(left=True)
    fig_path = 'figs/word_segmentation.png'
    sns.plt.savefig(fig_path)
    os.system('open ' + fig_path)



def update_accuracies(df):
    def errors_to_accuracy(errors):
        def get_network_choices(trial_errors, use_sigmoid=True, k=7):
            """Returns ([int], [float]): choices and reaction time
            choices is a list of 0 or 1 corresponding to word indices in one trial
            reaction time is a list of floats

            Args:
              trial_errors [(float, float)]: each tuple represents
                the reaction times for the two words in a trial
            """
            def sigmoid(t):
                return 1 / (1 + np.e ** (-k * t))

            def percent_difference(a, b):
                return (a - b)/ ((a + b) / 2)
                
            # pairs of consecutive words correspond to trials
            def choose(pair):
                if use_sigmoid:
                    p_first = sigmoid(percent_difference(pair[1], pair[0]))
                else:
                    p_first = pair[1] / (pair[1] + pair[0])

                choice = 0 if random.random() < p_first else 1
                rt = 1 / (abs(pair[0]-pair[1])/(sum(pair)/2)) 
                return (choice, rt)

            results = [choose(p) for p in trial_errors]
            choices = [r[0] for r in results]
            reaction_times = [r[1] for r in results]
            return choices, reaction_times


        def get_correct_choices():
            """Returns [int]: tuple index of correct word for each trial"""
            with open('../SRN/experiment/answer-key.txt', 'r') as f:
                key = f.read()
            key = key.split('\r')
            key = map(int, key)
            return key


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

        trial_errors = [(errors[i], errors[i+1]) for i in xrange(0, len(errors), 2)]
        choices, reaction_times = get_network_choices(trial_errors)
        incorrect = test_word_choices(choices)
        accuracy = 1 - (float(len(incorrect)) / len(choices))
        return accuracy

    df['contoid_accuracy'] = df.apply(lambda row: errors_to_accuracy(
                                      row['contoid_errors']), axis=1)
    df['vocoid_accuracy'] = df.apply(lambda row: errors_to_accuracy(
                                     row['vocoid_errors']), axis=1)


def print_aovs(df):
    pass

def main(args):
    if 'csv' in args:
        cols = args[args.index('csv') + 1:]
        if 'all' in cols:
            cols = 'all'
        create_csv(columns=cols)
    else:
        # test_choice_function()
        df = get_data_frame('all')
        update_accuracies(df)
        df.to_csv('net-log2.csv')
        # fix_column_labels(df)
        # plot_experimental_accuracies(df)
        
        # plot_reaction_times(df)
        # plot_word_segmentation(df)

if __name__ == '__main__':
    main(sys.argv[1:])
    # create_csv()
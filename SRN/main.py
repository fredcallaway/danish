import numpy as np
import corpora
import utils
from sklearn import metrics
from network import Network
from joblib import Parallel, delayed
import joblib


def extract_boundaries(corpus):
    """Returns a corpus with boundaries removed, and boundary markers.

    XABXDEFXHXIJKLX -> ABDEFHIJKL, 0100110001"""
    pairs = utils.neighbors(corpus)
    phones_and_boundaries = ((phone, nxt == 'X')      # phone, precedes_boundary
                             for phone, nxt in pairs  # for all adjacent pairs
                             if phone != 'X')         # except ones that lead with a boundary
    return phones_and_boundaries

def prepare(corpus, distributed):
    # Nets are trained to predict the next phoneme.
    inputs, targets = zip(*utils.neighbors(corpus))

    # Encode phonemes into numeric representations.
    encoding = corpora.get_encoding(distributed=distributed)

    return [encoding[c] for c in inputs], [encoding[c] for c in targets]


def get_corpora(lang, num_train=500000, num_test=10000, distributed=False):
        full_corpus = corpora.get_corpus(lang, word_boundaries=True)

        # A list of (phoneme, precedes_boundary) tuples.
        phones_and_boundaries = extract_boundaries(full_corpus)

        # Divide into train and test.
        train, test = corpora.train_test_split(phones_and_boundaries, 
                                               num_train, num_test, mode='begin')

        # Separate phones from boundary markers.
        train_phones, _ = map(list, zip(*train))
        test_phones, test_bounds = map(list, zip(*test))

        # Construct targets and encode phonemes.
        train_in, train_out = prepare(train_phones, distributed)
        test_in, test_out = prepare(test_phones, distributed)
        
        # Remove the trailing bound to match test_out.
        del test_bounds[-1]
        assert len(train_in) == len(train_out)
        assert len(test_in) == len(test_out) == len(test_bounds)

        return (train_in, train_out), (test_in, test_out), test_bounds


def run_experiment(net, lang, exp):
    with open('experiment/{0}/train{1}.txt'.format(lang, exp), 'r') as f:
        exp_train = f.read()
    train = prepare(exp_train, net.distributed)
    net.fit(*train)

    with open('experiment/{0}/test{1}.txt'.format(lang, exp), 'r') as f:
        trials = ['Q' + word.strip() + 'Q' for word in f]

    for word in trials:
        trial_in, trial_out = prepare(word, net.distributed)
        result = net.test(trial_in, trial_out)
        yield result


def run_net(net, lang, num_train, num_test, name=None):
    name = name or lang + str(net.seed) + ('d' if net.distributed else 'l')
    save_dir = 'nets/' + name
    train, test, test_bounds = get_corpora(lang, num_train, num_test, net.distributed)
    net.fit(*train)
    print('saved', name)
    net.save(save_dir)
    test_result = net.test(*test)
    test_errors = test_result['error_total']
    test_outputs = test_result['out_activations']

    exp_a_results = run_experiment(net, lang, 'A')
    exp_a_errors = [result['error_total'] for result in exp_a_results]
    net = Network.load(save_dir)  # reset to before experiment A training
    exp_b_results = run_experiment(net, lang, 'B')
    exp_b_errors = [result['error_total'] for result in exp_b_results]

    return {'lang': lang,
            'name': name,
            'distributed': net.distributed,
            'test_errors': test_errors,
            'test_outputs': test_outputs,
            'exp_a_errors': exp_a_errors,
            'exp_b_errors': exp_b_errors,
            'test_bounds': test_bounds}


def main(num_nets=1, num_train=50000, num_test=1000):
    results = []
    for lang in ['english', 'danish']:
        nets = [Network(i, distributed=d) for i in range(num_nets) for d in (True, False)]
        jobs = (delayed(run_net)(net, lang, num_train, num_test) for net in nets)
        results.extend(Parallel(n_jobs=-1)(jobs))
    
    joblib.dump(results, 'results.pkl', compress=3)




    


if __name__ == '__main__':
    main(3, 100000, 1000)
import numpy as np
import corpora
import utils
from sklearn import metrics
from network import Network
import segmentation
import scratch
from joblib import Parallel, delayed
import joblib


def extract_boundaries(corpus):
    """Returns a corpus with boundaries removed, and boundary markers.

    XABXDEFXHXIJKLX -> ABDEFHIJKL, 0100110001"""
    pairs = utils.neighbors(corpus)
    phones_and_boundaries = ((phone, nxt == 'X')      # phone, precedes_boundary
                             for phone, nxt in pairs  # for all adjacent pairs
                             if phone != 'X')         # except ones that lead with a boundary
    #return map(list, zip(*phones_and_boundaries))
    return phones_and_boundaries

def prepare(corpus):
    # Nets are trained to predict the next phoneme.
    inputs, targets = zip(*utils.neighbors(corpus))

    # Encode phonemes into numeric representations.
    encoding = corpora.get_encoding(distributed=False)
    #def encode(corpus):
        #return [encoding[c] for c in corpus]

    return [encoding[c] for c in inputs], [encoding[c] for c in targets]
    #return map(encode, (inputs, targets))


def get_corpora(lang, num_train=500000, num_test=10000):
        full_corpus = corpora.get_corpus(lang, word_boundaries=True)

        # A list of (phoneme, precedes_boundary) tuples.
        phones_and_boundaries = extract_boundaries(full_corpus)

        # Divide into train and test.
        train, test = corpora.train_test_split(phones_and_boundaries, num_train, num_test)

        # Separate phones from boundary markers.
        train_phones, _ = map(list, zip(*train))
        test_phones, test_bounds = map(list, zip(*test))

        # Nets are trained to predict the next phoneme.
        train_in, train_out = prepare(train_phones)
        test_in, test_out = prepare(test_phones)
        
        # Remove the trailing bound to match test_out.
        del test_bounds[-1]
        assert len(train_in) == len(train_out)
        assert len(test_in) == len(test_out) == len(test_bounds)

        return (train_in, train_out), (test_in, test_out), test_bounds


def run_net2(net, lang, name=None):
    #return
    name = name or lang + str(net.seed)
    save_dir = 'nets/' + name
    train, test, test_bounds = get_corpora(lang, 100, 100)
    net.fit(*train)
    net.save(save_dir)
    test_results = net.test(*test)

    exp_a_results = list(run_experiment(net, lang, 'A'))
    net = Network.load(save_dir)  # reset to before experiment A training
    exp_b_results = list(run_experiment(net, lang, 'B'))

    return test_results, exp_a_results, exp_b_results, test_bounds


def main2(num_nets=1):
    results = {}
    for lang in ['english', 'danish']:
        nets = [Network(i) for i in range(num_nets)]
        jobs = (delayed(run_net2)(net, lang) for net in nets)
        results[lang] = Parallel(n_jobs=2)(jobs)
    
    joblib.dump(results, 'results.pkl')


def run_net(name, net, train, test, A_train, A_trials, B_train, B_trials):
    #return
    save_dir = 'nets/' + name

    net.fit(*train)
    net.save(save_dir)
    test_result = net.test(*test)

    net.fit(*A_train)
    a_results = [net.test(*trial) for trial in A_trials]
    #a_results = [net.test(*trial) for trial in A_trials]
    net = Network.load(save_dir)  # reset to before running experiment A
    net.fit(*B_train)
    b_results = [net.test(*trial) for trial in B_trials]
    #b_results = [net.test(*trial) for trial in B_trials]

    return test_result, a_results, b_results


def main(num_train=500000, num_test=10000, num_nets=1):
    for lang in ['english', 'danish']:
        train, test, test_bounds = get_corpora(lang)
        A_train, A_trials = get_experiment(lang, 'A')
        B_train, B_trials = get_experiment(lang, 'B')

        all_data = {'train': train,
                    'test': test,
                    'A_train': A_train,
                    'A_trials': A_trials,
                    'B_train': B_train,
                    'B_trials': B_trials}
        
        # Create several networks with different initial random weights.
        nets = {lang + str(i): Network(i) for i in range(num_nets)}

        jobs = (delayed(run_net)(name, net, **all_data) for name, net in nets.items())
        results = Parallel(n_jobs=2)(jobs)
        joblib.dump(results, 'results.pkl')

        ## Can activation of boundary unit predict word boundaries?
        #for result in results:
        #    boundary_activations = result['out_activations'][:, -1]
        #    auc = metrics.roc_auc_score(test_bounds, boundary_activations)
        #    print(auc)
        #    yield {'lang': lang, 'roc_auc': auc}


        # TODO experiment: just run test() on every item


def get_experiment(lang, exp):
    with open('experiment/{0}/train{1}.txt'.format(lang, exp), 'r') as exp_train:
        train = prepare(exp_train.read())

    with open('experiment/{0}/test{1}.txt'.format(lang, exp), 'r') as exp_test:
        trials = [prepare('Q' + word.strip() + 'Q') for word in exp_test]

    return train, trials



def run_experiment(net, lang, exp):
    return
    with open('experiment/{0}/train{1}.txt'.format(lang, exp), 'r') as f:
        exp_train = f.read()
    train = prepare(exp_train)
    net.fit(*train)

    with open('experiment/{0}/test{1}.txt'.format(lang, exp), 'r') as f:
        trials = ['Q' + word.strip() + 'Q' for word in f]

    for word in trials:
        trial_in, trial_out = prepare(word)
        result = net.test(trial_in, trial_out)
        yield result



def play():
    net = Network()
    *corpora, _ = get_corpora('english', 50000, 1000)
    result = run_net('test', net, *corpora)


if __name__ == '__main__':
    #list(run(100, 100))
    #with utils.Timer('1'):
        #main(100, 100)
    with utils.Timer('2'):
        main2()
    #play()
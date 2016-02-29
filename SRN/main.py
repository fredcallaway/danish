import numpy as np
import corpora
import utils
from sklearn import metrics
from network import Network
import segmentation
import scratch
from joblib import Parallel, delayed

def extract_boundaries(corpus):
    """Returns a corpus with boundaries removed, and boundary markers.

    XABXDEFXHXIJKLX -> ABDEFHIJKL, 0100110001"""
    pairs = utils.neighbors(corpus)
    phones_and_boundaries = ((phone, nxt == 'X')      # phone, precedes_boundary
                             for phone, nxt in pairs  # for all adjacent pairs
                             if phone != 'X')         # except ones that lead with a boundary
    #return map(list, zip(*phones_and_boundaries))
    return phones_and_boundaries

def main(num_train=500000, num_test=10000):
    results = {}
    for lang in ['english', 'danish']:
        full_corpus = corpora.get_corpus(lang, word_boundaries=True)

        # A list of (phoneme, precedes_boundary) tuples.
        phones_and_boundaries = extract_boundaries(full_corpus)

        # Divide into train and test.
        train, test = corpora.train_test_split(phones_and_boundaries, num_train, num_test)

        # Separate phones from boundary markers.
        train_phones, _ = map(list, zip(*train))
        test_phones, test_bounds = map(list, zip(*test))

        # Nets are trained to predict the next phoneme.
        train_in, train_out = zip(*utils.neighbors(train_phones))
        test_in, test_out = zip(*utils.neighbors(test_phones))

        # Remove the trailing bound to match test_out.
        del test_bounds[-1]
        assert len(train_in) == len(train_out)
        assert len(test_in) == len(test_out) == len(test_bounds)

        # Encode phonemes into numeric representations.
        encoding = corpora.get_encoding(distributed=False)
        def encode(corpus):
            return [encoding[c] for c in corpus]

        train_in, train_out, test_in, test_out = [encode(corpus)
            for corpus in (train_in, train_out, test_in, test_out)]

        ## Train and test network.
        #net = Network()
        #net.fit(train_in, train_out)
        #results[lang] = net.test(test_in, test_out)

        ## Can activation of boundary unit predict word boundaries?
        #boundary_activations = results[lang]['out_activations'][:, -1]
        #print(metrics.roc_auc_score(test_bounds, boundary_activations))

        # Create several networks with different initial random weights.
        nets = {lang+'0': Network(0),
                #lang+'1': Network(1),
                #lang+'2': Network(2),
                }

        argss = [(net, train_in, train_out, test_in, test_out, name)
                 for name, net in nets.items()]


        results = Parallel(n_jobs=1)(delayed(run_net)(*args) for args in argss)

        # TODO experiment: just run test() on every item


def run_net(net, train_in, train_out, test_in, test_out, name):
    save_dir = 'nets/' + name
    net.fit(train_in, train_out)
    result = net.test(test_in, test_out)
    net.save(save_dir)
    return result


def play():
    net = Network()
    in_, out = np.random.rand(10, 5), np.random.rand(10, 5)
    net.fit(in_, out)
    net.save('nets/test')

    net2 = Network.load('nets/test/')
    print(net2.num_input)

if __name__ == '__main__':
    main(100, 100)
    #main()
    #play()
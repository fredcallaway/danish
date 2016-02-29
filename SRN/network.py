"""Python interface to LENS"""
import tempfile
import numpy as np
import re
from collections import namedtuple
import itertools
import shutil
import logging
import time
import subprocess
import pickle
from copy import deepcopy
import os
import psutil
import sys

import create_files
from lens import write_lens_files
from experiment import evaluate_experiment
from segmentation import test_segmentation


LENS_LOCATION = '/Applications/LensOSX.app/Contents/MacOS/LensOSX'
LENS_NAME = 'LensOSX'

if not os.path.isfile(LENS_LOCATION):
    LENS_LOCATION = '/Users/fred/Applications/LensOSX.app/Contents/MacOS/LensOSX'

logging.basicConfig(level=logging.WARNING)


class Network(object):
    """A simple recurrent network to be used in experimental modeling

    Attributes:
        seed (int): seed used for random initial net weights.
        hidden (int): number of hidden units.
        rate (float): learning rate.
        momentum (float): momentum.
        ticks (int): number of examples that backpropogation goes back.
        num_train (int): number of training examples.
        incoding (dict): maps phonemes to input layer network representations.
        outcoding (dict): maps phonemes to outupt layer network representations.
        time (str): last time the network was modified.


        """
    def __init__(self, seed=0, num_hidden=80, learning_rate=0.1, momentum=0.95,
                 backprop_ticks=1, rand_range=0.25, architecture='templates/architecture.txt'):
        super(Network, self).__init__()
        self.seed = seed
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.backprop_ticks = backprop_ticks
        self.rand_range = rand_range

        #self.incoding = incoding
        #self.outcoding = outcoding
        #self.num_input = len(next(iter(self.incoding.values())))
        #self.num_output = len(next(iter(self.outcoding.values())))

        self.num_input = None  # set by first call to fit()
        self.num_output = None

        self.dir = os.path.abspath(tempfile.mkdtemp(dir='temp-lens')) + '/'
        self.weight_file = os.path.abspath(self.dir + 'weights.wt')

    def save(self, dir):
        """Saves the network for later use."""
        #return
        os.makedirs(dir, exist_ok=True)
        with open(dir + '/network.pkl', 'wb+') as f:
            pickle.dump(self, f)
        shutil.copy(self.weight_file, dir)

    @staticmethod
    def load(dir):
        with open(dir + '/network.pkl', 'rb') as f:
            net = pickle.load(f)
        shutil.copy(dir + '/weights.wt', net.weight_file)
        return net

    def fit(self, inputs, targets):
        """Trains the network and tests segmentation ability.

        Updates self.segmentation_results with results.
        """
        #return
        if len(inputs) != len(targets):
            raise ValueError('inputs and targets have different lengths: %s and %s'
                             % len(inputs), len(targets))
        if self.num_input is None:
            self.num_input = len(inputs[0])
            self.num_output = len(targets[0])
        elif self.num_input != len(inputs[0]) or self.num_output != len(targets[0]):
            raise ValueError('incompatible input or output size')

        self.num_updates = len(inputs)

        self._write_ex_file('train.ex', inputs, targets)
        self._write_in_file('train.in')
        self._run_lens('train.in')

        #pickle.dump(self, open('nets/%s.p' % self._id, 'wb'))

    def test(self, inputs, targets):
        #return
        if len(inputs) != len(targets):
            raise ValueError('inputs and targets have different lengths: %s and %s'
                             % len(inputs), len(targets))

        self._write_ex_file('test.ex', inputs, targets)
        self._write_in_file('test.in')
        out = self._run_lens('test.in')

        out_activations = []
        with open(self.dir + '/output-activations.out', 'r') as acts_file:
            next(acts_file) # skip first line
            for line in acts_file:
                activations = line.split(' ')[:-1]
                activations = [float(a) for a in activations]
                out_activations.append(activations)
        out_activations = np.array(out_activations)

        def parse_test_out(out):
            labels = {
                'Error total:       ',
                'Error per example: ',
                'Error per tick:    ',
                'Unit cost per tick:'
            }
            for line in out.split('\n'):
                if line[:19] in labels:
                    yield re.split(': +', line)[-1]

        field_names = ['error_total', 'error_per_example', 'error_per_tick',
                       'unit_cost_per_tick', 'out_activations']
        values = (*parse_test_out(out), out_activations)

        return dict(zip(field_names, values))
        #TestResult = namedtuple('TestResult', field_names)
        #return TestResult(*parse_test_out(out), out_activations)


    def _write_ex_file(self, file, inputs, targets):
        with open(self.dir + file, 'w+') as f:      
            for input, target, in zip(inputs, targets):
                """Writes one example to a .ex file"""
                #f.write('name: {%s -> %s}\n' % (input, target))
                f.write('name: {A -> B}\n')
                f.write('1\n')
                # Ppper case letter indicates dense encoding.
                f.write('I: %s T: %s;\n' % (' '.join(map(str, input)), 
                                            ' '.join(map(str, target))))
                #if distributed:
                #else:
                    #f.write('i: ' + self.incoding[input] + ' t: ' + self.outcoding[target] + ';\n')

    def _write_in_file(self, file):
        with open('templates/architecture.in') as f:
            architecture = f.read()
        with open('templates/' + file) as f:
            full = '\n'.join([architecture, f.read()])
            formatted = full % self.__dict__  # replaces variables in template file
        with open(self.dir + file, 'w+') as f:
            f.write(formatted)

    def _create_id(self):
        """Create id which uniquly identifies self."""
        distributed = 'd' if self.distributed else 'l'
        rate = str(self.rate)[2:]
        momentum = str(self.momentum)[2:]
        rand_range = str(self.rand_range)[2:]
        self._id = ('%s_%s_%s_%s_%s_%s_%s_%s_%s'
                    % (self.lang, distributed, self.seed, self.hidden, rate,
                       momentum, self.ticks, rand_range, self.num_train,))

    def _write_example_files(self):
        create_files.write_example_files(self.lang, self.distributed, self.num_train)

    def _write_lens_files(self):
        input_ = len(self.incoding)
        output = len(self.outcoding)
        write_lens_files(self._id, self.seed, input_, self.hidden,
                         output, self.num_train, self.rate,
                         self.momentum, self.ticks, self.rand_range)

    def _run_lens(self, in_file):
        """Executes a lens .in file and returens output.

        Raises RuntimeError if there is an error in the lens script."""
        try:
            in_file = self.dir + in_file
            # todo: pipe output to terminal and python
            out = subprocess.check_output([LENS_LOCATION, '-b', in_file]).decode('utf-8')
            if out[-8:] != 'success\n':  # script echos 'success' at the end
                raise RuntimeError('Error whil executing %s:\n%s' % (in_file, out))
            out = out[:out.rindex('\n')]  # remove success message
            with open('%s-log.out' % in_file, 'w+') as f:
                logging.debug(out)
                f.write(out)
            return out
        finally:
            # kill any remaining Lens processes
            for proc in psutil.process_iter():
                if proc.name() == LENS_NAME:
                    proc.kill()


def run_nets(parameters, rerun=None):
    """Runs a series of nets."""
    if not isinstance(parameters, list):
        parameters = [parameters]
    print('NOW RUNNING %s NETS' % len(parameters))
    for param in parameters:
        net = Network(**param)
        net.train_network()
        net.run_experiment()


def generate_permutations(parameters):
    """Returns [dict]: all permutations of params in parameters.

    parmeters must be a list of (str, list), where str is
    the key and list is a list of values"""

    def recurse(parameters, permutations):
        if not parameters:
            return permutations

        # return a copy of permutations with all possible values
        #   for param added to each permutation
        # this multiplies len(permutations) by len(values)
        param, values = parameters.pop(0)
        new_perms = []
        for v in values:
            perm_copy = deepcopy(permutations)
            for perm in perm_copy:
                perm[param] = v
            new_perms += perm_copy

        return recurse(parameters, new_perms)

    parameters.reverse()  # so that result is sorted by first parameter
    param, values = parameters.pop(0)
    permutations = [{param: val} for val in values]
    return recurse(parameters, permutations)


def main(*args):
    params = generate_permutations([('lang', ['danish', 'english']),
                                    ('seed', range(28)),
                                    ('distributed', [True, False])])
    run_nets(params)


def test():
    ins = 'abc' * 100
    outs = 'xyz' * 100
    incoding = outcoding = generate_coding('abcxyz')
    net = Network(incoding, outcoding)
    net.fit(ins, outs)

    ins = 'abc' * 100
    outs = 'xyy' * 100
    print(net.test(ins, outs))


if __name__ == '__main__':
    test1()








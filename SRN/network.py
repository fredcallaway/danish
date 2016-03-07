"""Python interface to LENS"""
import tempfile
import numpy as np
import re
import shutil
import logging
import subprocess
import pickle
import os

import utils

LENS_LOCATION = '/Applications/LensOSX.app/Contents/MacOS/LensOSX'
LENS_NAME = 'LensOSX'

if not os.path.isfile(LENS_LOCATION):
    LENS_LOCATION = '/Users/fred/Applications/LensOSX.app/Contents/MacOS/LensOSX'

logging.basicConfig(level=logging.INFO)


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
                 backprop_ticks=1, rand_range=0.25, architecture='templates/architecture.txt',
                 **kwargs):
        super(Network, self).__init__()
        self.seed = seed
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.backprop_ticks = backprop_ticks
        self.rand_range = rand_range

        self.num_input = None  # set by first call to fit()
        self.num_output = None

        os.makedirs('temp-lens', exist_ok=True)
        self.dir = os.path.abspath(tempfile.mkdtemp(dir='temp-lens')) + '/'
        logging.info('directory: ' + self.dir)
        self.weight_file = os.path.abspath(self.dir + 'weights.wt')

        self.__dict__.update(kwargs)

    def save(self, dir):
        """Saves the network for later use."""
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
        if len(inputs) != len(targets):
            raise ValueError('inputs and targets have different lengths: %s and %s'
                             % (len(inputs), len(targets)))
        if self.num_input is None:
            self.num_input = len(inputs[0])
            self.num_output = len(targets[0])
        elif self.num_input != len(inputs[0]):
            raise ValueError('incompatible input size: %s !=  %s'
                             % (len(inputs[0]), self.num_input))
        elif self.num_output != len(targets[0]):
            raise ValueError('incompatible output size: %s !=  %s'
                             % (len(targets[0]), self.num_output))

        self._write_ex_file('train.ex', inputs, targets)
        self._write_in_file('train.in')
        with utils.Timer(print_func=None) as t:
            self._run_lens('train.in')
        logging.info('trained on %s items in %s seconds' % (len(inputs), t.elapsed))


    def test(self, inputs, targets):
        if len(inputs) != len(targets):
            raise ValueError('inputs and targets have different lengths: %s and %s'
                             % (len(inputs), len(targets)))

        self._write_ex_file('test.ex', inputs, targets)
        self._write_in_file('test.in')
        out = self._run_lens('test.in')

        # Recover saved unit activations.
        out_activations = []
        with open(self.dir + '/output-activations.out', 'r') as acts_file:
            next(acts_file) # skip first line
            for line in acts_file:
                activations = line.split(' ')[:-1]
                activations = [float(a) for a in activations]
                out_activations.append(activations)
        out_activations = np.array(out_activations)

        def parse_test_out(out):
            # Get the useful information out of the lens stdout.
            labels = {
                'Error total:       ',
                'Error per example: ',
                'Error per tick:    ',
                'Unit cost per tick:'
            }
            for line in out.split('\n'):
                if line[:19] in labels:
                    value = re.split(': +', line)[-1]
                    yield float(value)

        field_names = ['error_total', 'error_per_example', 'error_per_tick',
                       'unit_cost_per_tick', 'out_activations']
        try:
            values = (*parse_test_out(out), out_activations)
        except Exception:
            logging.error('Lens output:\n' + out)
            raise

        return dict(zip(field_names, values))
        

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

    def _run_lens(self, in_file):
        """Executes a lens .in file and returens output.

        Raises RuntimeError if there is an error in the lens script."""
        in_file = self.dir + in_file
        # todo: pipe output to terminal and python
        out = subprocess.check_output([LENS_LOCATION, '-b', in_file]).decode('utf-8')
        if out[-8:] != 'success\n':  # script echos 'success' at the end
            raise RuntimeError('Error whil executing %s:\n%s' % (in_file, out))
        #out = out[:out.rindex('\n')]  # remove success message
        with open('%s-log.out' % in_file, 'w+') as f:
            logging.debug(out)
            f.write(out)
        return out




def example():
    train_in, train_targets = np.random.rand(1000, 20), np.random.rand(1000, 10)
    test_in, test_targets = np.random.rand(100, 20), np.random.rand(100, 10)
    net = Network(num_hidden=40, momentum=0.9)
    net.fit(train_in, train_targets)
    error = net.test(test_in, test_targets)['error_total']
    print(error)




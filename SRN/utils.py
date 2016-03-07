import time

class Timer(object):
    """A context manager which times the block it surrounds.

    Args:
        name (str): The name of the timer for time messages
        print_func (callable): The function used to print messages
          e.g. logging.debug

    Based heavily on https://github.com/brouberol/contexttimer

    Example:
    >>> with Timer('Busy') as timer:
            [i**2 for i in range(100000)]
            timer.lap()
            [i**2 for i in range(100000)]
            timer.lap()
            [i**2 for i in range(100000)]
    Busy (0): 0.069 seconds
    Busy (1): 0.126 seconds
    Busy (total): 0.176 seconds
    """
    def __init__(self, name='Timer', print_func=print):
        self.name = name
        self.print_func = print_func or (lambda *args: None)  # dummy function.
        self._lap_idx = 0

    @property
    def elapsed(self):
        return time.time() - self.start

    def lap(self, label=None):
        if label is None:
            label = self._lap_idx
        self._lap_idx += 1
        self.print_func("%s (%s): %0.3f seconds" % (self.name, label, self.elapsed))

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self,ty,val,tb):
        self.lap('total')


def neighbors(lst, n=2):
    """Iterates through adjacent groups in the list.

    neighbors([1,2,3,4], n=3) -> [1,2,3], [2,3,4]
    """
    num_groups = len(lst) - n + 1
    for i in range(num_groups):
        yield lst[i:i+n]


if __name__ == '__main__':
    print(list(neighbors('abcdefg')))
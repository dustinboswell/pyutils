import time

def timeit(func):
    """A simple function/method decorator to print how long the function call took."""
    def wrapper(*args, **kwds):
        t_start = time.time()
        ret = func(*args, **kwds)
        print "%s() took %f secs" % (func.__name__, time.time() - t_start)
        return ret
    return wrapper

class WithTimer:
    """Usage:
            with WithTime("executing the query"):
                ...
       This will print the time taken inside that with block."""
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t_start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        print "%s took %2.2f secs" % (self.name, time.time() - self.t_start)

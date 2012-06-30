#!/usr/bin/python

"""Treasure-trove of statistics-related classes and functions."""

import collections
import time
import math
import random
from threading import Lock

class SummaryStats:
    """Simple class to keep avgerages and other summary stats of a set of data.
    Note this class uses O(1) space because it doesn't keep each data point."""
    def __init__(self):
        self._total = 0.0
        self._total_squares = 0.0
        self._count = 0

    def add(self, x, weight=1):
        """Add a data point 'x' (a number).
        If weight = 2, it's as if add() was called twice."""
        self._total += x * weight
        self._total_squares += (x*x) * weight
        self._count += weight

    def remove(self, x, weight=1):
        """Un-does a previous add()."""
        self._total -= x * weight
        self._total_squares -= (x*x) * weight
        self._count -= weight

    def total(self):
        """Returns the (weighted) sum of all the data."""
        return self._total

    def count(self):
        """Returns the (weighted) count of how many data points have been added."""
        return self._count

    def avg(self):
        """Returns the mean of the data."""
        if self._count:
            return self._total / self._count
        else:
            return float('nan')

    def var(self):
        """Returns the sample variance."""
        if self._count < 2: return float('nan')
        # Note: this algorithm isn't numerically stable, so sometimes the variance
        # will be very close to 0, but be negative (which is why we need the max())
        return max(0, (self._total_squares - (self._total*self.avg())) / (self._count - 1))

    def std(self):
        """Returns the sample standard deviation."""
        if self._count < 2: return float('nan')
        return self.var() ** 0.5

    def update(self, summary_stats):
        """Add another SummaryStats into us."""
        self._total += summary_stats.total()
        self._total_squares += summary_stats._total_squares
        self._count += summary_stats.count()

    def __str__(self):
        return "total=%2.2f, avg=%3.3f, std=%2.2f, count=%d" % (
                self.total(), self.avg(), self.std(), self._count)

class SummaryStatsDict:
    """A dictionary of SummaryStats.  Thread-safe for most methods.
    Supports all the methods of SummaryStats -- you just have to insert the key
    as the first argument.

    Example Usage:
        ssd = SummaryStatsDict()
        ssd.add("key1", 10)
        ssd.add("key2", 20)
        ...
    """

    def __init__(self):
        self.stats_dict = {}  # dict of "key" => SummaryStats
        self.lock = Lock()

    def __len__(self):
        with self.lock:
            return len(self.stats_dict)

    def __getattr__(self, attr_name):
        """TODO: memoize this function somehow to speed it up"""
        def method(name, *args, **kwds):
            with self.lock:
                ss = self.stats_dict.get(name)
                if not ss:
                    ss = SummaryStats()
                    self.stats_dict[name] = ss
                return getattr(ss, attr_name)(*args, **kwds)
        return method

    def sum_total(self):
        """Return the sum of the total() of each SummaryStats."""
        with self.lock:
            total = 0
            for key, ss in self.stats_dict.iteritems():
                total += ss.total()
            return total

    def dict_of_counts(self):
        """Return a dict, where each value is the count() of the corresponding
        SummaryStats."""
        with self.lock:
            return dict((key, ss.count()) for (key, ss) in self.stats_dict.iteritems())

    def dict_of_totals(self):
        """Return a dict, where each value is the total() of the corresponding
        SummaryStats."""
        with self.lock:
            return dict((key, ss.total()) for (key, ss) in self.stats_dict.iteritems())

    def update_add(self, summary_stats_dict):
        """Warning: not thread-safe. (TODO: re-entrant lock)"""
        for other_key, other_ss in summary_stats_dict.stats_dict.iteritems():
            self.update(other_key, other_ss)


    def print_top(self, min_count=0, max_print=1000, key_str=lambda k:k, sort_key=lambda k: -k.count()):
        """Print the "top" SummaryStats, sorted by the given options."""
        with self.lock:
            items = self.stats_dict.items()
            items = sorted(items, key=lambda k: sort_key(k[1]))
            num_printed = 0
            for (key, ss) in items:
                if ss.count() < min_count: continue
                print "%s  ->  %s" % (str(key_str(key)).ljust(35), ss)
                num_printed += 1
                if num_printed == max_print: break

    def __getitem__(self, key):
        """Note: the preferred way to manipulate an underlying dict is by one of the
        methods above, since they do the "create-if-not-there" logic."""
        return self.stats_dict.get(key)

    def __str__(self):
        s = ""
        with self.lock:
            for (key, ss) in self.stats_dict.iteritems():
                s += "%s  ->  %s\n" % (str(key).ljust(15), ss)
        return s

class RecentCounterDict:
    """A Thread-safe dictionary of RecentCounters that automatically creates new
    counters as needed.  It provides the same methods as RecentCounters, but with
    'name' as the first argument.  Usage:

    rcd = RecentCounterDict()
    rcd.add("counter-1", 20)
    rcd.add("counter-2", 50)
    print rcd.minute_count("counter-1")  # prints "20"
"""
    def __init__(self):
        self.counters = {}
        self.lock = Lock()

    def __str__(self):
        s = ""
        with self.lock:
            for name in sorted(self.counters.keys()):
                rc = self.counters[name]
                s += "%s: %d/min, %d/hr, %d total\n" % (
                    name, rc.minute_count(), rc.hour_count(), rc.total_count())
        return s

    def __getattr__(self, attr_name):
        def method(name, *args, **kwds):
            with self.lock:
                rc = self.counters.get(name)
                if not rc:
                    rc = RecentCounter()
                    self.counters[name] = rc
                return getattr(rc, attr_name)(*args, **kwds)
        return method

class RecentCounter:
    """A counter to keep track of totals that "slide" over recent intervals.
    See "The Art of Readable Code" chapter 15: "Minute/Hour Counter"
    Usage:

        recent_counter = RecentCounter()
        recent_counter.add(5)
        recent_counter.add(10)
        time.sleep(100)
        assert recent_count.minute_count() == 0
        assert recent_count.hour_count() == 15

    You an also do custom-sized windows by doing:

        TEN_MINUTES = 10 * 60
        recent_counter.create_counter(TEN_MINUTES)
        recent_counter.add(...)
        recent_counter.recent_count(TEN_MINUTES)

    The implementation is very space & time efficient, only using a ~500B per counter.
    """

    def __init__(self):
        self.bucket_counters = {}  # num_secs -> TrailingBucketCounter
        # minute, ten_minute, and hour come built-in; the rest you create yourself.
        self.create_counter(60)
        self.create_counter(600)
        self.create_counter(3600)
        self.total = 0

    def create_counter(self, num_secs, num_buckets=60):
        assert num_secs % num_buckets == 0
        secs_per_bucket = num_secs / num_buckets
        self.bucket_counters.setdefault(
            num_secs, TrailingBucketCounter(num_buckets, secs_per_bucket))

    def add(self, count):
        now = time.time()
        for b in self.bucket_counters.itervalues():
            b.add(count=count, now=now)
        self.total += count

    def total_count(self):
        return self.total

    def minute_count(self):
        return self.recent_count(60)

    def ten_minute_count(self):
        return self.recent_count(600)

    def hour_count(self):
        return self.recent_count(3600)

    def recent_count(self, num_secs):
        assert num_secs in self.bucket_counters.keys()
        now = time.time()
        return self.bucket_counters[num_secs].trailing_count(now)

class TrailingBucketCounter:
    def __init__(self, num_buckets, secs_per_bucket):
        self.secs_per_bucket = secs_per_bucket
        self.num_buckets = num_buckets
        self.total = 0
        self.last_update_time = 0
        self.q = collections.deque([0] * num_buckets)

    def update(self, now):
        elapsed_buckets = (int(now / self.secs_per_bucket)
                           - int(self.last_update_time / self.secs_per_bucket));

        self.shift(elapsed_buckets)
        self.last_update_time = now

    def shift(self, num_shifted):
        if num_shifted == 0: return

        # In case too many items shifted, just clear the queue.
        if num_shifted >= self.num_buckets:
            self.q.clear()
            self.q.extend([0] * self.num_buckets)
            self.total = 0
            return

        # Push all the needed zeros.
        self.q.extend([0] * num_shifted)

        # Let all the excess items fall off.
        while len(self.q) > self.num_buckets:
            self.total -= self.q.popleft()
        assert len(self.q) == self.num_buckets

    def add(self, count, now):
        self.update(now)
        self.q[-1] += count
        self.total += count

    def trailing_count(self, now):
        self.update(now)
        return self.total

import unittest
class RecentCounterTest(unittest.TestCase):
    def testAll(self):
        rc = RecentCounter()
        self.assertEqual(0, rc.minute_count())
        self.assertEqual(0, rc.hour_count())

        rc.add(5)
        rc.add(5)
        self.assertEqual(10, rc.minute_count())
        self.assertEqual(10, rc.hour_count())

        print "this is gonna take 62 seconds..."
        time.sleep(62)  # TODO: mock out 'time()' so we can just advance clock.
        self.assertEqual(0, rc.minute_count())
        self.assertEqual(10, rc.hour_count())

z_table = {  # maps '9X-percentile' -> z_{1-alpha/2}
    '0.80':   1.281551565545,
    '0.90':   1.644853626951,
    '0.95':   1.959963984540,
    '0.98':   2.326347874041,
    '0.99':   2.575829303549,
    '0.995':    2.807033768344,
    '0.998':    3.090232306168,
    '0.999':    3.290526731492,
    '0.9999':   3.890591886413,
    '0.99999':  4.417173413469,
    '0.999999':     4.891638475699,
    '0.9999999':    5.326723886384,
    '0.99999999':   5.730728868236,
    '0.999999999':  6.109410204869
}

def conf_interval(k, n, conf='0.95'):
    """Given 'k' successes out of 'n' observations, return a (lower, upper)
    interval that contains the underlying probability with the given confidence.
    Currently uses the 'Wilson Interval', but this may change in the future.
    """
    assert k >= 0 and n >= 0 and k <= n
    if n == 0: return (0, 1)  # should we just assert n > 0 instead?
    k = float(k)
    n = float(n)
    z = z_table[conf]
    p = k / n
    bottom = 1.0 + ((z*z)/n)
    topleft = p + ((z*z)/(2*n))
    topright = z * (((p * (1-p))/n + (z*z)/(4*n*n)) ** 0.5)
    lower = (topleft - topright) / bottom
    upper = (topleft + topright) / bottom
    return (lower, upper)

def prob_beta_greater_than(k, n, p):
    """Suppose you flip a coin 'n' times, and get heads 'k' times.
    This function returns the probability that the true heads-bias of that coin is
    greater than 'p'. For example,
    prob_beta_greater_than(1000, 1000, 0.5) => close to 1.0
    prob_beta_greater_than(0   , 1000, 0.5) => close to 0.0
    prob_beta_greater_than(500 , 1000, 0.5) => close to 0.5
    """
    alpha = k + 1
    beta = (n-k) + 1
    NUM_TRIALS = 1000
    # out of 1000 random trials (sampling from <alpha,beta>), how often was it > p ?
    return sum(random.betavariate(alpha, beta) > p for x in xrange(NUM_TRIALS)) / float(NUM_TRIALS)

def bigram_likelihood_ratio(c1, c2, c12, n):
    """Ted Dunning's likelihood ratio test for bigrams
       @n - number of words in the corpus.
       @c1 - count of word1 in corpus
       @c2 - count of word2 in corpus
       @c12 - count of bigram (word1, word2) in corpus
    """
    if c1 == 0 or c2 == 0 or c12 == 0: return 0
    def logL(k, n, x):
        assert x >= 0 and x <= 1
        return (k * math.log(x)) + ((n-k) * math.log(1-x))
    p = float(c2)/n
    p1 = float(c12)/c1
    p2 = float(c2-c12) / (n-c1)
    return -2 * (logL(c12, c1, p) + logL(c2-c12, n-c1, p)
                 - logL(c12, c1, p1) - logL(c2-c12, n-c1, p2))

def combine_gaussians(mean_var_list):
    """@mean_var_list is like [(mean1, var1), (mean2, var2), ... ]
    returns a (mean, variance) that is the "product" of the input gaussians."""
    variance = 1.0 / sum([1.0 / v  for (m, v) in mean_var_list])
    mean_top = sum([m   / (2.0*v)  for (m, v) in mean_var_list])
    mean_bot = sum([1.0 / (2.0*v)  for (m, v) in mean_var_list])
    return (mean_top/mean_bot, variance)

class Histogram(object):
    def __init__(self):
        self.buckets = {}  # label -> count
        self.total_count = 0

    def add(self, label, count=1):
        self.buckets.setdefault(label, 0)
        self.buckets[label] += count
        self.total_count += count

    def top_buckets(self, num_buckets, normalize=True):
        label_counts = self.buckets.items()
        label_counts = sorted(label_counts, key=lambda k: -k[1])[0:num_buckets]
        if normalize:
            return [(a, float(x)/self.total_count if self.total_count else 0.0) for (a, x) in label_counts]
        else:
            return label_counts

    def __str__(self):
        buckets = self.top_buckets(99)
        pieces = []
        for key, value in sorted(self.buckets.items()):
            if type(key) == float: key = "%2.2f" % key
            if type(value) == float: value = "%2.2f" % value
            pieces.append("%s => %s " % (key, value))
        return "[" + ", ".join(pieces) + "]"


class NormalDist(object):
    """A class to collect data, and fit a Gaussian to it."""
    def __init__(self):
        self.data = []

    def add(self, x):
        self.data.append(x)

    def fit_gaussian(self, prior_data=[]):
        """Returns (mean, variance) of best fit"""
        if prior_data:
            combined_data = self.data[:]  # deep copy
            combined_data.extend(prior_data)
        else:
            combined_data = self.data

        sum_squares = sum([x*x for x in combined_data])
        sum_data = sum(combined_data)
        num_data = len(combined_data)
        mean = float(sum_data) / num_data
        variance = (1.0 / (num_data-1)) * sum([(x - mean)*(x - mean) for x in combined_data])
        return (mean, variance)

    def __str__(self):
        (mean, variance) = self.fit_gaussian()
        return "mean=%2.2f, std=%2.2f, N=%d" % (mean, variance**0.5, len(self.data))

class DistSampler:
    """A class that lets you sample from an arbitrary distribution.
    TODO: replace this implementation with that smart O(1) algorithm..."""
    def __init__(self, prob_dict):
        """prob_dict maps from key -> probability"""
        self.prob_key_list = [(prob, key) for (key, prob) in prob_dict.iteritems()]
        self.prob_key_list.sort(reverse=True) # highest prob first

    def sample_key(self):
        """This is the dumb linear algorithm - replace with a faster one."""
        p = random.random()
        for (prob, key) in self.prob_key_list:
            if p <= prob: return key
            p -= prob
        assert False  # how did we get here?

    def sample_dict(self, count):
        d = {}
        for x in xrange(int(count)):
            key = self.sample_key()
            d.setdefault(key, 0)
            d[key] += 1
        return d

import unittest
class DistSampleTest(unittest.TestCase):
    def assertAboutEqual(self, a, b, delta):
        assert abs(a - b) <= delta

    def testAll(self):
        dist_sampler = DistSampler({'a': 0.85, 'b': 0.14, 'c': 0.01})
        sample_counts = dist_sampler.sample_dict(10000)
        self.assertAboutEqual(sample_counts['a'], 8500, delta=50)
        self.assertAboutEqual(sample_counts['b'], 1400, delta=50)
        self.assertAboutEqual(sample_counts['c'], 100, delta=10)


if __name__ == "__main__":
    unittest.main()

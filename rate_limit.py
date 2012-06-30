#!/usr/bin/python

import logging
import threading
import time

class rate_limit(object):
    """A decorator to rate-limit how often the function is called, sleeping if needed.
    Usage to rate-limit at 5/sec:
        @rate_limit(max_calls=5, in_seconds=1.0)
        def dont_abuse_me(): ...

    In the above case, if the 6th call is made during 1 sec, it will sleep the rest
    of the 1 sec, so that we obey "5 max calls in 1.0 seconds".
    """
    def __init__(self, max_calls, in_seconds, func_name=None):
        """Note: You have both 'max_calls' and 'in_seconds' to give you flexibility about
           how aggressively to clamp things down.  For example, both of these:
               @rate_limit(max_calls=1, in_seconds=1)
               @rate_limit(max_calls=100, in_seconds=100)
           will result in an average rate of 1/s, but the second one will allow bursts
           of up to 100 inside that 100 second window.
           """
        self.max_calls_per_bucket = max_calls
        self.bucket_secs = in_seconds
        self.time_bucket_start = 0
        self.calls_this_bucket = 0
        self.func_name = func_name
        self.lock = threading.Lock()

    def __call__(self, func):
        func_name = func.__name__ if (self.func_name is None) else self.func_name

        def new_func(*args, **kwds):
            self.lock.acquire()
            now = time.time()
            # way past last bucket - start over
            if now - self.time_bucket_start >= self.bucket_secs:
                self.time_bucket_start = now
                self.calls_this_bucket = 0

            # do rate limiting
            if self.calls_this_bucket >= self.max_calls_per_bucket:
                time_left = (self.time_bucket_start + self.bucket_secs) - now
                logging.info(
                  "Rate-limiting %s() (exceeded %d-per-%ss, sleeping for %.2f secs)" % \
                  (func_name, self.max_calls_per_bucket, self.bucket_secs, time_left))
                time.sleep(max(0, time_left))

                # okay, start a new bucket now
                self.time_bucket_start = time.time()
                self.calls_this_bucket = 0

            self.calls_this_bucket += 1
            self.lock.release()
            # Note: even if func() throws an Exception, proper rate-limiting occurs
            return func(*args, **kwds)
        return new_func


if __name__ == "__main__":
    @rate_limit(max_calls=3, in_seconds=1.0, func_name="print_hello")
    def print_hello():
        print "hello"

    while True:
        print_hello()

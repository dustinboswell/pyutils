"""A handful of high-level helpers for dealing with threads."""

import logging
import threading
import time
import traceback
from Queue import Queue, Full

# all threads created by this module are stored here
_threads = Queue()

def start_background_thread(func, args=(), kwargs={}, daemon=True):
    t = threading.Thread(target=func, args=args, kwargs=kwargs)
    _threads.put(t)
    t.daemon = daemon
    t.start()

def start_periodic_thread(func, sleep_secs, args=(), kwargs={}, daemon=True):
    def exec_loop(*args, **kwargs):
        while True:
            func(*args, **kwargs)
            time.sleep(sleep_secs)
    start_background_thread(exec_loop, args, kwargs, daemon)

class ThreadPool:
    """Note: not fully tested (but seems to work :)"""
    def __init__(self, num_threads, max_tasks=-1):
        self.tasks = Queue(max_tasks)

        def do_tasks():
            (func, args, kwargs) = self.tasks.get()
            try:
                func(*args, **kwargs)
            except Exception, e:
                logging.error("Uncaught exception in ThreadPool: " + str(e))
                logging.error(traceback.format_exc())

        for x in xrange(num_threads):
            start_periodic_thread(do_tasks, sleep_secs=0)

    def add_task(self, func, args=(), kwargs={}):
        """Add a task to the queue. If the queue is full (max_tasks), raise Full."""
        self.tasks.put_nowait((func, args, kwargs))

def test_ThreadPool():
    MAX_TASKS = 2
    thread_pool = ThreadPool(1, max_tasks=MAX_TASKS)
    total_output = []
    def task(message):
        time.sleep(1)
        total_output.append(message)

    for i in xrange(MAX_TASKS):
        thread_pool.add_task(task, ("*",))

    try:
        thread_pool.add_task(task, ("X",))
    except Full:
        pass
    else:
        assert False  # I expected an exception dammit!

    time.sleep(3)
    assert total_output == (["*"] * MAX_TASKS)

    def bad_task():
        raise Exception()
    print "**** An Exeption trace should be printed below..."
    thread_pool.add_task(bad_task)
    time.sleep(1)  # wait for the task to be finished and Exception to be swallowed.

if __name__ == "__main__":
    test_ThreadPool()

"""An OptionParser with commonly used options.
Usage:

from option_parser import option_parser

if __name__ == "__main__":
    option_parser.add_int("-n", "--num_days", default=10,
        help="How many days worth of logs to process")

    (options, args) = option_parser.parse_args()
    print options.num_days
    # args are the 'leftover' args from sys.argv
"""

from optparse import OptionParser

class BetterOptionParser(OptionParser):
    def __init__(self):
        OptionParser.__init__(self)

    def add_bool(self, *args, **kwds):
        kwds["action"] = "store_true"
        if "default" not in kwds:
            kwds["default"] = False
        self.add_option(*args, **kwds)

    def add_int(self, *args, **kwds):
        kwds["type"] = "int"
        self.add_option(*args, **kwds)

    def add_float(self, *args, **kwds):
        kwds["type"] = "float"
        self.add_option(*args, **kwds)

    def add_str(self, *args, **kwds):
        self.add_option(*args, **kwds)

option_parser = BetterOptionParser()

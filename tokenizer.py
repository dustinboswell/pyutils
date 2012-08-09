"""Code that splits a long piece of text into smaller chunks (words or ngrams)."""

import re
import sys
import math
from collections import defaultdict

PUNCTUATION = """`~!$%^&*()_-+={[}]|\:;"'?/>.<, """

class Tokenizer(object):
    def __init__(self):
        self.ngram_count = defaultdict(int)
        self.num_trained_words = 0  # how many words have been shown to train_ngrams()

    def train_ngrams(self, text):
        for word_run in self.tokenize_word_runs(text):
            words = self.tokenize_words(word_run)

            for i in xrange(len(words)):
                # add all unigrams, bigrams, and trigrams
                self.add_ngram(words[i])
                if i >= 1:
                    self.add_ngram(words[i-1] + " " + words[i])
                if i >= 2:
                    self.add_ngram(words[i-2] + " " + words[i-1] + " " + words[i])

            self.num_trained_words += len(words)
            if self.num_trained_words % 50000 == 0:
                print "Tokenizer.train_ngrams(%dM words)    \r" % (
                    self.num_trained_words/1e6),
                sys.stdout.flush()
        print

    def add_ngram(self, ngram):
        self.ngram_count[ngram] += 1

    def tokenize_words(self, text):
        """
        E.g. "Hi. My sister's name, that's hyphenated, is Mary-Anne." =>
             ["hi", "my", "sister's", "name", "that's", "hyphenated", "is", "mary-anne"]
        """
        words = []
        for word_run in self.tokenize_word_runs(text):
            for word in word_run.split(" "):
                word = word.strip()
                if word:
                    words.append(word)
        return words

    def tokenize_word_runs(self, text):
        """A "word run" is a string of text containing 1+ words that are not separated
        by any serious punctuation.  The idea is that any set of adjacent words in a
        word run is a viable ngram candidate.
     .
        E.g. "Hi. My sister's name, that's hyphenated, is Mary-Anne." =>
               ["hi", "my sister's name", "that's hyphenated", "is mary-anne"]
        """
        text = text.lower()
        # A word is separated from other words by " "
        # A word run is separated from other word runs by "\t"

        # certain characters are definite word "barriers"
        text = re.sub("[/&]", " ", text)
        # certain characters are definite word run "barriers"
        text = re.sub("[\r\n]", "\t", text)
        # other characters are only word run barriers if there are multiple in a row
        text = re.sub("[^a-zA-Z0-9]{2,}", "\t", text)
        return [word_run.strip(PUNCTUATION) for word_run in text.split("\t")]

    def tokenize_ngrams(self, text):
        """strips punctuations, and 'bracketizes' common phrases.
        Input: "Dave23 says: 'I went to New York City'"
        Output: ["dave23", "says", "i", "went", "to", "new york city"]
        """
        def log_prob(words):
            """Return the log-probability of this array of words"""
            ngram = " ".join(words)
            prob = self.ngram_count.get(ngram, 0) / float(self.num_trained_words)
            if prob == 0:
                return -100 * len(words)  # i.e. as if each word had 1/e^100 prob
            return math.log(prob)

        # If there are multiple runs, recurse
        word_runs = self.tokenize_word_runs(text)
        if len(word_runs) != 1:
            ngrams = []
            for word_run in word_runs:
                ngrams.extend(self.tokenize_ngrams(word_run))
            return ngrams

        # Break this word run into words.  Then ngram them in the most likely way.
        words = self.tokenize_words(word_runs[0])
        if not words: return []
        num_words = len(words)

        # We use a Dynamic Programming solution. dp_*[i] contains the solution to the
        # subproblem where only words[0:i+1] existed.
        dp_len = [-1] * num_words  # the size ngram that words[i] is the end of
        dp_score = [-1] * num_words  # the sum of log-probs for this subproblem solution

        # base case
        dp_len[0] = 1
        dp_score[0] = log_prob(words[0:1])

        for i in xrange(1, num_words):
            best_ngram_len = None
            best_total_score = -10e20
            for ngram_len in [1,2,3]:
                if i - ngram_len + 1 < 0: continue
                score = dp_score[i-ngram_len] + log_prob(words[i-ngram_len+1:i+1])
                if score > best_total_score:
                    best_total_score = score
                    best_ngram_len = ngram_len
            dp_len[i] = best_ngram_len or 1
            dp_score[i] = best_total_score

        tokens = []
        i = num_words - 1
        while i >= 0:
            tokens.append(" ".join(words[i-dp_len[i]+1:i+1]))
            i -= dp_len[i]
        tokens.reverse()
        return tokens


if __name__ == "__main__":
    from option_parser import option_parser
    option_parser.add_str("--train_ngram_text")
    option_parser.add_str("--test_ngram_text")
    (options, args) = option_parser.parse_args()

    tokenizer = Tokenizer()
    if options.train_ngram_text:
        tokenizer.train_ngrams(open(options.train_ngram_text).read())

    if options.test_ngram_text:
        for line in open(options.test_ngram_text):
            print line.strip(), " => ",
            print " ".join("[%s]" % ngram for ngram in tokenizer.tokenize_ngrams(line))

    EXAMPLES = [
        '''Hi. My sister's name, that's hyphenated, is Mary-Anne.''',
            ]
    for example in EXAMPLES:
        print "raw: ", example
        print "tokenize_word_runs: ", tokenizer.tokenize_word_runs(example)
        print "tokenize_words: ", tokenizer.tokenize_words(example)

    while True:
        line = raw_input()
        print "tokenize_word_runs => ", tokenizer.tokenize_word_runs(line)
        print "tokenize_words => ", tokenizer.tokenize_words(line)
        print "tokenize_ngrams => ", tokenizer.tokenize_ngrams(line)

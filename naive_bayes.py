"""A Naive Bayes Classifier, for the case where there are 2 output classes: {0,1},
and the input features are also 0/1-valued.
See http://en.wikipedia.org/wiki/Naive_Bayes_classifier#Document_Classification
"""

import math

from training_examples import TrainingExamples
from classifier import BinaryClassifier

class NaiveBayesClassifier(BinaryClassifier):
    def prob_output1(self, example, explain=False):
        """Return the estimated probability that @example belongs to the '1' class.
        If @explain is True, also returns a string explanation of how the estimate
        was calculated."""
        # Note: 'llhr' means 'log-likelihood-ratio' in this function.
        llhr = self.log_likelihood_ratio["PRIOR"]
        explanation = ""
        if explain:
            explanation += "%+2.2f PRIOR (log-likelihood-ratio)\n" % llhr

        for (name, value) in example.iteritems():
            if name.startswith("_"): continue

            value = int(value)
            assert value in [0,1]
            real_name = ("" if value == 1 else "NOT_") + name

            llhr_delta = self.log_likelihood_ratio[real_name]
            llhr += llhr_delta

            if explain:
                explanation += "%+2.2f  %s\n" % (llhr_delta, real_name)

        # Turn llhr back into a probability (simple algebra)
        prob = math.exp(llhr) / (1 + math.exp(llhr))

        if explain:
            explanation += "TOTAL SUM = %+2.2f\n" % llhr
            explanation += "PROB = %+2.2f\n" % prob

        if explain:
            return (prob, explanation)
        else:
            return prob

    def train(self, training_examples, **kwds):
        """Learns the params for self.log_likelihood_ratio based on training_examples.
        - Assumes each example has the same set of feature names (no sparse features).
        - Assumes the output for each example is 0 or 1.
        - Assumes the value for each feature is 0 or 1.

        As a side-effect, print out Python code representing self.log_likelihood_ratio.
        That Python code can be saved to a file (copy/paste/modify) and then loaded via
        load_weights()."""
        pos_count = training_examples.count(output_value=1)
        neg_count = training_examples.count(output_value=0)
        assert pos_count + neg_count == len(training_examples)  # only 0/1 outputs!

        s = "log_likelihood_ratio = {\n"
        s += "    'PRIOR': %2f,\n" % math.log(float(pos_count) / neg_count)

        # Now analyze each feature's predictiveness of positive/negative
        for name in training_examples.feature_names:
            # Make sure all features are boolean (0/1 valued)
            assert training_examples.count(feature_name=name) == \
                training_examples.count(feature_name=name, feature_value=0) + \
                training_examples.count(feature_name=name, feature_value=1)

            pos_count_f = training_examples.count(
                feature_name=name, feature_value=1, output_value=1)
            neg_count_f = training_examples.count(
                feature_name=name, feature_value=1, output_value=0)

            ls = 1  # Laplace smoothing constant
            p_f_given_pos = (ls + pos_count_f) / float(2*ls + pos_count)
            p_f_given_neg = (ls + neg_count_f) / float(2*ls + neg_count)

            llhr = math.log(p_f_given_pos / p_f_given_neg)

            s += "    %s: %2.2f," % (("'%s'" % name).ljust(28), llhr)
            s += "  # P(*|positive) = %2.2f (%d of %d)" % (
                p_f_given_pos, pos_count_f, pos_count)
            s += "  P(*|negative) = %2.2f (%d of %d)\n" % (
                p_f_given_neg, neg_count_f, neg_count)

            p_not_f_given_pos = 1 - p_f_given_pos
            p_not_f_given_neg = 1 - p_f_given_neg
            llhr = math.log(p_not_f_given_pos / p_not_f_given_neg)

            s += "    %s: %2.2f,\n" % (("'NOT_%s'" % name).ljust(28), llhr)

        s += "}"
        print s
        exec s
        self.log_likelihood_ratio = log_likelihood_ratio

    def load_weights(self, filename):
        """Sets self.log_likelihood_ratio based on the contents of the file,
        which is assumed to have the same syntax as that printed by train()"""
        py_str = open(filename).read()
        exec py_str
        self.log_likelihood_ratio = log_likelihood_ratio

def test_NaiveBayesClassifier():
    # Setup a training set where "fat" is the only input feature, and
    # the output value is whether they died of diabetes.
    # This is an easy training set where the input is almost always equal
    # to the output.
    te = TrainingExamples()
    te.add_example({ 'fat': 1}, 1)
    te.add_example({ 'fat': 1}, 1)
    te.add_example({ 'fat': 1}, 1)
    te.add_example({ 'fat': 1}, 1)
    te.add_example({ 'fat': 1}, 1)
    te.add_example({ 'fat': 1}, 1)
    te.add_example({ 'fat': 0}, 1)  # almost always

    te.add_example({ 'fat': 0}, 0)
    te.add_example({ 'fat': 0}, 0)
    te.add_example({ 'fat': 0}, 0)
    te.add_example({ 'fat': 0}, 0)
    te.add_example({ 'fat': 0}, 0)
    te.add_example({ 'fat': 0}, 0)
    te.add_example({ 'fat': 1}, 0)  # almost always

    classifier = NaiveBayesClassifier()
    classifier.train(te)

    assert classifier.prob_output1({ 'fat': 1}) > classifier.prob_output1({ 'fat': 0})

    # let's see the underlying weights
    print classifier.prob_output1({ 'fat': 1}, explain=True)
    print classifier.prob_output1({ 'fat': 0}, explain=True)

if __name__ == "__main__":
    test_NaiveBayesClassifier()

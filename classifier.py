from stats import SummaryStats, Histogram

from training_examples import TrainingExamples

class BinaryClassifier:
    """Abstract base class for any classifier that is trained on examples whose
    outputs are either 0 or 1.  Also provides a test() method."""

    def prob_output1(self, example):
        """What is the probability that this example has an _OUTPUT of 1?"""
        raise "NotImplemented"

    def train(self, training_examples):
        raise "NotImplemented"

    def test(self, examples, print_level=1):
        """Computes the "area under the ROC curve". This is a way to measure the
        precision/recall WITHOUT choosing a cutoff-threshold.  It is mathematically
        equivalent to:
           "the probability that a random positive example has a higher
            prob_output1 than a random negative case"
        (This equivalence is non-obvious).

        The algorithm below computes this average probability by effectively trying
        all combinations of positive-vs-negative examples, but does this in O(NlgN)
        instead of O(N^2)"""
        if type(examples) is TrainingExamples:
            examples = examples.examples

        prob_stats = SummaryStats()
        prob_hist = Histogram()
        output1_scores = list()
        output0_scores = list()
        for example in examples:
            assert example["_OUTPUT"] in [0,1]
            prob = self.prob_output1(example)
            prob_stats.add(prob)
            prob_key = "%1.1f-%1.1f" % (int(prob*10)/10.0, (int(prob*10)+1)/10.0)
            if prob == 1: prob_key = "0.9-1.0"  # don't create a 1.0-1.1 bucket
            prob_hist.add(prob_key)
            real_output = (example["_OUTPUT"] == 1)
            if real_output:
                output1_scores.append(prob)
            else:
                output0_scores.append(prob)

        output1_scores.sort()
        output0_scores.sort()

        if print_level >= 2:
            print "%d output1 scores:" % len(output1_scores),
            print ["%2.2f" % i for i in output1_scores[0:5]],
            print " ... ",
            print ["%2.2f" % i for i in output1_scores[-5:]]

            print "%d output0 scores:" % len(output0_scores),
            print ["%2.2f" % i for i in output0_scores[0:5]],
            print " ... ",
            print ["%2.2f" % i for i in output0_scores[-5:]]

        j = 0
        num_correct_rank = 0
        for i in xrange(len(output1_scores)):
            o1_score = output1_scores[i]
            while j < len(output0_scores) and output0_scores[j] < o1_score: j += 1
            num_correct_rank += j
        rank_accuracy = float(num_correct_rank) / (len(output0_scores)*len(output1_scores))

        if print_level >= 1:
            print "Ranking accuracy: ('area under the ROC curve') %2.3f" % rank_accuracy
            print "Prob output: mean=%1.3f std=%1.3f" % (prob_stats.avg(), prob_stats.std())
            print "Histogram of prob output:", prob_hist
        return rank_accuracy

import random
class RandomBinaryClassifier(BinaryClassifier):
    def prob_output1(self, example):
        return random.random()

if __name__ == "__main__":
    # data set where:  x > 0 <==> output is 1
    te = TrainingExamples()
    for x in xrange(1, 1000):
        te.add_example({ 'x': x}, 1)
        te.add_example({ 'x': -x}, 0)

    # A random classifier should have 50% area under the curve.
    random_binary_classifier = RandomBinaryClassifier()
    area_under_curve = random_binary_classifier.test(te)
    assert 0.45 <= area_under_curve <= 0.55

    # A perfectly correct classifier should have 100% area under curve.
    class PerfectBinaryClassifier(BinaryClassifier):
        def prob_output1(self, example):
            if example['x'] > 0: return 1
            else: return 0

    perfect_binary_classifier = PerfectBinaryClassifier()
    area_under_curve = perfect_binary_classifier.test(te)
    assert 0.99 <= area_under_curve <= 1.01

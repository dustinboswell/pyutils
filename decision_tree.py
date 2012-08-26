import collections
import random

from stats import SummaryStats
from training_examples import TrainingExamples

class DecisionTree:
    """http://www-users.cs.umn.edu/~kumar/dmbook/ch4.pdf
    A real-output-valued decision tree, whose nodes split on real-valued
    inputs. You can still use this for binary classification by having 0/1 outputs.
    """
    def __init__(self):
        self.examples = []
        self.example_output_stats = SummaryStats()
        # For each feature, store a list of examples sorted by that feature's value
        self.examples_sorted_by_feature = collections.defaultdict(list)

        self.decision_feature = None
        self.decision_threshold = None

        # the two subtrees induced by the above decision function
        self.subtrees = [None, None]

    def add_example(self, example):
        self.examples.append(example)
        self.example_output_stats.add(float(example["_OUTPUT"]))

    def _examples_features(self):
        """Return the set of useable features from the examples.
        (Internally assumes examples[0] has all features.)"""
        return set(f for f in self.examples[0].keys() if not f.startswith("_"))

    def _sort_by_features(self, feature=None):
        """Called after final example is added. Only needs to be called once,
        for the root node, since sorting is preserved when splitting."""
        if feature is None:
            features = self._examples_features()
        else:
            assert type(feature) is str
            features = [feature]

        for feature in features:
            if self.examples_sorted_by_feature[feature]: continue  #already done
            self.examples_sorted_by_feature[feature] = list(self.examples)
            self.examples_sorted_by_feature[feature].sort(key=lambda e: e[feature])

    def avg_squared_error(self, examples=None):
        "returns avg squared error of output"
        examples = examples or self.examples  # by default, use training examples
        output_stats = SummaryStats()
        for example in examples:
            prediction = self.predict(example)
            output_stats.add((prediction - float(example["_OUTPUT"])) ** 2)
        return output_stats.avg()

    def predict(self, example):
        "recursively goes down decision tree, and returns avg output value at leaf."
        subtree = self._decision_subtree(example)
        if subtree is None:
            return self.example_output_stats.avg()
        else:
            return subtree.predict(example)

    def decision_str(self):
        return "%s < %s" % (self.decision_feature, self.decision_threshold)

    def print_tree(self, prefix=""):
        if self.decision_feature is None:
            print "Leaf(", len(self.examples), "examples,",
            print "avg_output=", self.example_output_stats.avg(),
            print "var_output=", self.example_output_stats.var(),
            print ")"
        else:
            print "Node(", len(self.examples), "examples,",
            print "decision = [", self.decision_str(),
            print "] )"

            prefix += "    "
            print prefix, "false =>",
            self.subtrees[0].print_tree(prefix)

            print prefix, "true  =>",
            self.subtrees[1].print_tree(prefix)

    def _find_best_split(self, feature):
        """Find the best threshold value to split this node on, using @feature.
        Returns (less_than_threshold, split_err).
        The @less_than_threshold is what you use to "decide", i.e.:
            if example[feature] < less_than_threshold:
                decide_left ...
            else:
                decide_right ...
        Note that this method doesn't actually split anything: it just figures out
        which threshold value would be best to split at.
        """
        self._sort_by_features(feature)
        left_output_stats = SummaryStats()
        right_output_stats = SummaryStats()
        assert len(self.examples) == len(self.examples_sorted_by_feature[feature])

        # To begin, let's assume we push all examples into the right child node.
        for example in self.examples:
            right_output_stats.add(float(example["_OUTPUT"]))

        # Now, move the examples one by one to the left child node.
        # (Note the examples sorted by value -- it's as if we're adjusting the
        # less_than_threshold.)
        # After each example, calculate the goodness-of-split, and track the best.
        best_threshold = None
        best_err = None
        last_feature_value = None
        for example in self.examples_sorted_by_feature[feature]:
            feature_value = example[feature]
            output_value = float(example["_OUTPUT"])

            # Speed optimization: skip over examples with same feature value.
            if feature_value == last_feature_value:
                left_output_stats.add(output_value)
                right_output_stats.remove(output_value)
                continue

            last_feature_value = feature_value  # remember for next iteration

            left_count = left_output_stats.count()
            right_count = right_output_stats.count()

            # Edge-case: left or right child is empty
            if left_count == 0 or right_count == 0:
                left_output_stats.add(output_value)
                right_output_stats.remove(output_value)
                continue  # not a true split

            # Compute goodness-of-split: weighted average of the 2 output variances.
            if left_count <= 1: left_err = 0
            else: left_err = (left_count - 1) * left_output_stats.var()

            if right_count <= 1: right_err = 0
            else: right_err = (right_count - 1) * right_output_stats.var()

            err = left_err + right_err
            if best_err is None or err < best_err:
                best_threshold = feature_value
                best_err = err

            left_output_stats.add(output_value)
            right_output_stats.remove(output_value)

        # to save memory, delete this sorted array (we'll never use it again anyway)
        del self.examples_sorted_by_feature[feature]
        return (best_threshold, best_err)

    def _split_subtrees(self, feature, threshold):
        """Reset the two subtrees based on the current decision function."""
        assert feature is not None
        assert threshold is not None
        assert len(self.examples) >= 2

        self.decision_feature = feature
        self.decision_threshold = threshold

        self.subtrees = [DecisionTree(), DecisionTree()]
        for example in self.examples:
            decision = int(example[self.decision_feature] < self.decision_threshold)
            if decision == 0:
                self.subtrees[0].add_example(example)
            else:
                self.subtrees[1].add_example(example)

    def _decision_subtree(self, example):
        """returns one of the two child subtrees, or None if we are a leaf."""
        if self.decision_feature is None: return None
        decision = example[self.decision_feature] < self.decision_threshold
        return self.subtrees[int(decision)]

    def grow_tree(self, features_considered_per_node=2):
        # Stop growing based on termination criteria:
        if len(self.examples) <= 1: return
        if self.example_output_stats.var() < 0.001: return

        # assume all examples have all features
        feature_names = self._examples_features()
        feature_subset = random.sample(feature_names, features_considered_per_node)

        best_feature = None
        best_threshold = None
        best_avg_error = None
        for feature in feature_subset:
            (threshold, avg_error) = self._find_best_split(feature)
            if avg_error is None: continue  # no split possible (all same value?)

            if best_avg_error is None or avg_error < best_avg_error:
                best_feature = feature
                best_threshold = threshold
                best_avg_error = avg_error

        # Did we fail to find a good decision?
        if best_feature is None: return

        # Accept the best decision
        self._split_subtrees(best_feature, best_threshold)

        # Grow recursively at each branch.
        self.subtrees[0].grow_tree(features_considered_per_node)
        self.subtrees[1].grow_tree(features_considered_per_node)

def test_DecisionTree_simple():
    te = TrainingExamples()
    te.add_example({ 'fat': 1}, 1)
    te.add_example({ 'fat': 1}, 1)
    te.add_example({ 'fat': 1}, 1)
    te.add_example({ 'fat': 1}, 1)
    te.add_example({ 'fat': 0}, 1)
    te.add_example({ 'fat': 0}, 0)
    te.add_example({ 'fat': 0}, 0)
    te.add_example({ 'fat': 0}, 0)

    tree = DecisionTree()
    for example in te.examples:
        tree.add_example(example)
    tree.grow_tree(features_considered_per_node=1)
    tree.print_tree()

    assert tree.predict({'fat': 1}) >= 1.0
    assert tree.predict({'fat': 0}) <= 0.3

def test_DecisionTree_medium():
    # come up with some input->output formula to see if we can learn it.
    def formula(x,y,z):
        return x + y + z

    # helper function to generate input->output examples
    def random_example():
        x = random.random() + 0.1
        y = random.random() + 0.1
        z = random.random() + 0.1
        output = formula(x,y,z)
        return (x,y,z,output)

    # generate a training set
    te = TrainingExamples()
    for i in xrange(20):
        (x,y,z,output) = random_example()
        te.add_example({'x':x, 'y':y, 'z':z}, output)

    # learn a decision tree based on training examples
    tree = DecisionTree()
    for example in te.examples:
        tree.add_example(example)
    tree.grow_tree(features_considered_per_node=1)

    # check that we learned the training set with enough accuracy
    tree.print_tree()
    assert tree.avg_squared_error(te.examples) < 0.001

    # TODO: train with a much larger training set, and test how it performs on non-training examples

if __name__ == "__main__":
    test_DecisionTree_simple()
    test_DecisionTree_medium()

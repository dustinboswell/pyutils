import random

from stats import SummaryStats
from classifier import BinaryClassifier
from decision_tree import DecisionTree
from training_examples import TrainingExamples

class DecisionForest(BinaryClassifier):
    """http://stat-www.berkeley.edu/users/breiman/RandomForests/cc_home.htm
    Holds a set of DecisionTrees and averages their outputs.
    DecisionForest can be used as a Classifier if all training example outputs
    are 0/1, but can also be used for ordinary "regression"."""

    def prob_output1(self, example):
        """If all the training examples had _OUTPUT that were 0/1,
        then this should just work as a classifier as-is."""
        return self.predict(example)

    def predict(self, example):
        """Returns the average predicted output over all our trees."""
        output_stats = SummaryStats()
        for tree in self.trees:
            output_stats.add(tree.predict(example))
        return output_stats.avg()

    def avg_squared_error(self, examples):
        """Returns average squared error of the predicted output.
        This is useful to determine if the DecisionForest was able to learn the
        training set completely.
        If the training_error is very high, then the forest was not powerful enough
        to learn the training set (and perhaps features_considered_per_node should be
        increased.)
        In theory, a DecisionForest should never overfit the training data, because
        each tree was trained on a random bootstrap sample of the training examples.
        """
        errors = SummaryStats()
        for example in examples:
            prediction = self.predict(example)
            squared_error = (prediction - float(example["_OUTPUT"])) ** 2
            errors.add(squared_error)
        return errors.avg()

    def train(self, training_examples, train_on_subset=True, num_trees=100, features_considered_per_node=2, **kwds):
        print "Training a decision forest of %d trees, using %d examples, and %d features considered per node." % (
                num_trees, len(training_examples), features_considered_per_node)
        self.trees = []
        total_test_output_stats = SummaryStats()

        binary_classification = all(example["_OUTPUT"] in [0,1] for example in training_examples)
        #binary_classification = True
        #for example in training_examples:
        #    output = example["_OUTPUT"]
        #    if output not in [0,1]:
        #        binary_classification = False
        #        break

        for tree_i in xrange(1, num_trees+1):
            tree = DecisionTree()
            self.trees.append(tree)

            test_set_ids = set(xrange(len(training_examples)))
            for i in xrange(len(training_examples)):
                if train_on_subset:  # N samples with replacement ("bootstrap")
                    index = random.randint(0, len(training_examples)-1)
                else:
                    index = i

                tree.add_example(training_examples[index])
                test_set_ids.discard(index)

            print "Growing tree %d/%d ..." % (tree_i, num_trees),
            tree.grow_tree(features_considered_per_node=features_considered_per_node)

            # Report the in-sample training error
            if binary_classification:
                print "area-under-curve for %d training examples is %2.2f" % (
                        len(tree.examples), tree.test(tree.examples, print_level=0))
            else:
                print "%2.2f avg err^2 on %d training examples" % (
                        tree.avg_squared_error(), len(tree.examples)),


            # Report the out-of-sample testing error, if we have any out-of-sample
            # examples to test on.
            if train_on_subset:
                print "; ",
                test_set = [training_examples[i] for i in test_set_ids]

                if binary_classification:
                    # Do a true out-of-sample test just on this one tree
                    # Temporarily make this a forest-of-one-tree...
                    save_trees = self.trees
                    self.trees = [tree]
                    self.test(test_set)
                    self.trees = save_trees
                else:
                    avg_squared_error = tree.avg_squared_error(test_set)
                    total_test_output_stats.add(avg_squared_error)

                    print "out-of-sample avg err^2 on %d test cases: %.2f [%.2f avg. for all %d trees so far]" % (len(test_set), avg_squared_error, total_test_output_stats.avg(), tree_i),

            print


def test_DecisionForest():
    """Train a decision forest against an arbitrary formula, to see if it can
    approximate it to an arbitrary low error, given enough examples."""
    def formula(x,y,z):
        return (x ** 2) + (x * y * z) + (10 * z) + (y / z) + 25

    def random_input_output():
        x = random.random() + 0.1
        y = random.random() + 0.1
        z = random.random() + 0.1
        output = formula(x,y,z)
        return ({'x':x, 'y':y, 'z':z}, output)


    te = TrainingExamples()
    for i in xrange(1, 5000):
        (input, output) = random_input_output()
        te.add_example(input, output)

        if i % 500: continue

        print "Testing after", i, "training examples"
        forest = DecisionForest()
        forest.train(te, train_on_subset=True, num_trees=10, features_considered_per_node=3)

        # Measure the true out-of-sample error rate for the entire forest.
        predict_err = SummaryStats()
        for j in xrange(10000):
            (input, output) = random_input_output()
            predicted_output = forest.predict(input)
            predict_err.add((output - predicted_output) ** 2)
        print "avg squared error = ", predict_err.avg()


if __name__ == "__main__":
    test_DecisionForest()

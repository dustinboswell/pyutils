import collections
import json
import cPickle

class TrainingExamples:
    """Used to store a set of input/output "training" examples for supervised learning.
    Each example's input is a dictionary of feature_name => feature_value,
    where feature_name is a string, and feature_value is a float/int.
    Each output value is also a float/int."""

    def __init__(self):
        self.examples = []
        self.feature_names = set()  # union of all feature names in examples
        self.counts = collections.defaultdict(int)

    def add_example(self, features, output_value):
        """Add the example to our list, and update various counts about it."""
        self.counts[self._key('*', '*', output_value)] += 1

        for (name, value) in features.iteritems():
            if name.startswith("_"): continue
            self.feature_names.add(name)

            self.counts[self._key(name, '*', '*')] += 1
            self.counts[self._key(name, '*', output_value)] += 1
            self.counts[self._key(name, value, '*')] += 1
            self.counts[self._key(name, value, output_value)] += 1

        features["_OUTPUT"] = output_value  # sneak into the dict
        self.examples.append(features)

    def count(self, feature_name='*', feature_value='*', output_value='*'):
        """Returns how many of our examples match the given criteria.
        The default of '*' means 'matches any'."""
        key = self._key(feature_name, feature_value, output_value)
        return self.counts.get(key, 0)

    @staticmethod
    def _key(feature_name, feature_value, output_value):
        return "%s--%s--%s" % (feature_name, feature_value, output_value)

    def normalize_one_example(self, example):
        """Normalize the feature values of this example to the range [0, 1], which is
        the percentile of that feature values, given the distribution seen in the training
        set.  Note that @example doesn't need to be in the training set.

        Assumes normalize_feature_values() has been called."""
        if example.get("_NORMALIZED", False): return  # already normalized
        for (name, value) in example.items():
            if name.startswith("_"):
                continue

            # get all the (sorted) values we've seen for this feature name.
            values = self.unnormalized_feature_values[name]

            if value <= values[0]:  # smaller than anything we've ever seen
                norm_value = 0.0
            elif value >= values[-1]:  # larger than anything we've ever seen
                norm_value = 1.0
            else:
                for quant_i in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
                    quant_value = values[int(quant_i*len(values))]
                    if value >= quant_value:
                        norm_value = quant_i
                        break

            # replace the feature value with the normalized value
            example[name] = norm_value
            example["_ORIG_" + name] = value  # save the old value

        example["_NORMALIZED"] = True

    def normalize_feature_values(self):
        """1) Calculate the feature value distribution for each feature (over all
        example's values)
        2) Replace each feature value with a value in [0,1] which represents it's
        "percentile" -- 0.0 means the value is amoung the lowest seen."""
        num_already_normalized = 0
        self.unnormalized_feature_values = {}  # name -> [values]
        for ex in self.examples:
            if ex.get("_NORMALIZED", False):
                num_already_normalized += 1
            for (name, value) in ex.iteritems():
                if name.startswith("_"): continue
                self.unnormalized_feature_values.setdefault(name, []).append(value)
        for array in self.unnormalized_feature_values.itervalues():
            array.sort()

        if num_already_normalized:  # can't mix/match normalize/unnormalized
            assert num_already_normalized == len(self.examples)

        for ex in self.examples:
            self.normalize_one_example(ex)

    def save_to_file(self, filename, num_examples=999999999):
        with open(filename, 'w') as fout:
            # the entire output is a json list of dictionaries.
            fout.write(json.dumps(list(self.examples[0:num_examples])))

    def load_from_file(self, filename):
        examples = json.loads(open(filename).read())
        for ex in examples:
            # probably not necessary to remote '_OUTPUT', but add_example() will add it
            output = ex["_OUTPUT"]
            del ex["_OUTPUT"]
            self.add_example(ex, output)

    def pickle_load(self, filename):
        f = open(filename,'rb')
        tmp_dict = cPickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def pickle_save(self, filename):
        f = open(filename,'wb')
        cPickle.dump(self.__dict__, f, 2)
        f.close()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

# TODO: define a number of "toy-problem" TrainingExamples of varying degrees
# of complexity, so that each classifier can be tested against it.

# Example usage.
if __name__ == "__main__":
    te = TrainingExamples()
    te.add_example({ 'fat': 300}, 1)
    te.add_example({ 'fat': 300}, 1)
    te.add_example({ 'fat': 100}, 0)

    for example in te:
        print example
    te.pickle_save("/tmp/training_examples.tmp")
    te.pickle_load("/tmp/training_examples.tmp")

    for example in te:
        print example
    te.pickle_save("/tmp/training_examples2.tmp")


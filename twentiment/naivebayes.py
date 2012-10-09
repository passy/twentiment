#!/usr/bin/env python3

"""
A classifier based on the Naive Bayes algorithm.  In order to find the
probability for a label, this algorithm first uses the Bayes rule to
express P(label|features) in terms of P(label) and P(features|label):

|                       P(label) * P(features|label)
|  P(label|features) = ------------------------------
|                              P(features)

The algorithm then makes the 'naive' assumption that all features are
independent, given the label:

|                       P(label) * P(f1|label) * ... * P(fn|label)
|  P(label|features) = --------------------------------------------
|                                         P(features)

Rather than computing P(featues) explicitly, the algorithm just
calculates the denominator for each label, and normalizes them so they
sum to one:

|                       P(label) * P(f1|label) * ... * P(fn|label)
|  P(label|features) = --------------------------------------------
|                        SUM[l]( P(l) * P(f1|l) * ... * P(fn|l) )


:author: Edward Loper <edloper@gradient.cis.upenn.edu>
:author: Pascal Hartig <phartig@weluse.de>
:url: <http://www.nltk.org/>
:license: Apache 2.0
"""

import logging
from collections import defaultdict
from twentiment.thirdparty.probability import (FreqDist, DictionaryProbDist,
                                               ELEProbDist, sum_logs)


class NaiveBayesClassifier(object):
    """
    A Naive Bayes classifier.  Naive Bayes classifiers are
    paramaterized by two probability distributions:

      - P(label) gives the probability that an input will receive each
        label, given no information about the input's features.

      - P(fname=fval|label) gives the probability that a given feature
        (fname) will receive a given value (fval), given that the
        label (label).

    If the classifier encounters an input with a feature that has
    never been seen with any label, then rather than assigning a
    probability of 0 to all labels, it will ignore that feature.

    The feature value 'None' is reserved for unseen feature values;
    you generally should not use 'None' as a feature value for one of
    your own features.
    """

    LOG = logging.getLogger('NaiveBayesClassifier')

    def __init__(self, label_probdist, feature_probdist):
        self._label_probdist = label_probdist
        self._feature_probdist = feature_probdist
        self._labels = list(label_probdist.samples())

    @staticmethod
    def train(labeled_featuresets, estimator=ELEProbDist):
        """
        :param labeled_featureset: A set of classified featuresets,
            i.e., a list of tuples ``[(featureset, label)]``.
        :param estimator: An estimator probability distribution. Defaults to an
            expected likelyhood estimation probability distribution.
        """

        label_freqdist = FreqDist()
        #: Features and values are stored in dictionaries defaulting to
        #: empty frequency distributions or sets, respectively.
        feature_freqdist = defaultdict(FreqDist)
        feature_values = defaultdict(set)
        #: Set of all feature names used, across labels.
        fnames = set()

        # Count how many times a particular feature value occured given the
        # label and feature name.
        for featureset, label in labeled_featuresets:
            # Track every label occurence.
            label_freqdist.inc(label)

            for fname, fval in featureset.items():
                # Increment the freq(fval|label, fname)
                feature_freqdist[label, fname].inc(fval)
                # Record that the fname can take the value fval.
                feature_values[fname].add(fval)
                # Keep a set of all used feature names.
                fnames.add(fname)

        for label in label_freqdist:
            num_samples = label_freqdist[label]
            for fname in fnames:
                # The count of the feature given the label.
                count = feature_freqdist[label, fname].N()
                # Create a balance between the labels, such that for each
                # 'missing' occasion of this feature in the current label,
                # there is a 'None' element incremented. So in the end all
                # freqdists have the same count, but with different fvalues.
                feature_freqdist[label, fname].inc(None, num_samples - count)
                # Make sure the values are aware of the None value.
                if (num_samples - count > 0):
                    feature_values[fname].add(None)

        #: The distribution P(label)
        label_probdist = estimator(label_freqdist)

        #: The distribution P(fval|label, fname)
        feature_probdist = {}
        for ((label, fname), freqdist) in feature_freqdist.items():
            # Create the estimator with as many bins as there are values of the
            # current feature name.
            probdist = estimator(freqdist, bins=len(feature_values[fname]))
            feature_probdist[label, fname] = probdist

        return NaiveBayesClassifier(label_probdist, feature_probdist)

    def prob_classify(self, featureset):
        """Calculate the probabilities the given featureset classifications
        and return a DictionaryProbDist instance.

        Works in O(nm) with n = # of labels, m = # of featureset elements.
        """

        # Work on a copy of the feature set, because we mutate it.
        fset = featureset.copy()
        for fname in featureset:
            for label in self._labels:
                if (label, fname) in self._feature_probdist:
                    break
            else:
                # Discard feature name we haven't been trained on from the
                # input set.
                del fset[fname]

        # Now we're working with a feature set that only includes known
        # features.

        # Instead of working with the product of the separate probabilities,
        # we use the sum of the logarithms to prevent underflows and make the
        # result more stable.

        #: The probability of each label, given the features. Starting with
        #: the probability of the label itself.
        logprob = {}
        for label in self._labels:
            logprob[label] = self._label_probdist.logprob(label)

        # Add the logarithmic probability of the features given the labels.
        for label in self._labels:
            for (fname, fval) in fset.items():
                feature_probs = self._feature_probdist.get((label, fname))

                if feature_probs is not None:
                    logprob[label] += feature_probs.logprob(fval)
                else:
                    # This should not occur if the classifier was created with
                    # the train() method.
                    logprob[label] += sum_logs([])  # = -INF.

        return DictionaryProbDist(logprob, normalize=True, log=True)

    def classify(self, featureset):
        """Return the most likely label for a given featureset."""

        return self.prob_classify(featureset).max()

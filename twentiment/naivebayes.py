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

    def __init__(self, label_probdist, feature_probdist):
        self._label_probdist = label_probdist
        self._feature_probdist = feature_probdist
        self._labels = label_probdist.samples()

    @staticmethod
    def train(labeled_featuresets, estimator=ELEProbDist):
        """
        :param labeled_featureset: A set of classified featuresets,
            i.e., a list of tuples ``[(featureset, label)]``.
        :param estimator: An estimator probability distribution. Defaults to an
            expected likelyhood estimation probability distribution.
        """

        label_freqdist = FreqDist()
        # Features and values are stored in dictionaries defaulting to
        # empty frequency distributions or sets, respectively.
        feature_freqdist = defaultdict(FreqDist)
        feature_values = defaultdict(set)

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

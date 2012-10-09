#!/usr/bin/env python3
"""
Supervised learning phase for the twentiment analyzer.

:author: 2012, Pascal Hartig <phartig@rdrei.net>
:license: BSD
"""

from __future__ import division
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.corpus import movie_reviews

import nltk.classify.util


def word_features(words):
    """Turns a list of words into a feature dictionary.

    The word is used as feature name, whereas the feature value is always
    True.
    """

    return {word: True for word in words}


def word_feature_lists(ids, sentiment):
    return [(word_features(movie_reviews.words(fileids=[f])), sentiment)
            for f in ids]


def learn_main():
    neg_ids = movie_reviews.fileids('neg')
    pos_ids = movie_reviews.fileids('pos')

    neg_features = word_feature_lists(neg_ids, 'neg')
    pos_features = word_feature_lists(pos_ids, 'pos')

    # Using 3/4 of the features as training set, the rest as test set
    neg_cutoff = int(len(neg_features) * (3 / 4))
    pos_cutoff = int(len(pos_features) * (3 / 4))

    train_features = neg_features[:neg_cutoff] + pos_features[:pos_cutoff]
    test_features = neg_features[neg_cutoff:] + pos_features[pos_cutoff:]

    print("Training on {} instances. Testing on {} instances.".format(
        len(train_features), len(test_features)
    ))

    classifier = NaiveBayesClassifier.train(train_features)
    print("Accuracy: ", nltk.classify.util.accuracy(classifier,
                                                    test_features))
    classifier.show_most_informative_features()
    import debug


if __name__ == "__main__":
    learn_main()

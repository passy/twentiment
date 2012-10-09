#!/usr/bin/env python3

from twentiment.thirdparty.probability import FreqDist
from twentiment.naivebayes import NaiveBayesClassifier
from twentiment.text import normalize_text

import zmq


def learn_main():
    pos_tweets = [('I love this car', 'positive'),
                  ('This view is amazing', 'positive'),
                  ('I feel great this morning', 'positive'),
                  ('I am so excited about the concert', 'positive'),
                  ('He is my best friend', 'positive')]

    neg_tweets = [('I do not like this car', 'negative'),
                  ('This view is horrible', 'negative'),
                  ('I feel tired this morning', 'negative'),
                  ('I am not looking forward to the concert', 'negative'),
                  ('He is my enemy', 'negative')]

    tweets = []
    for (words, sentiment) in pos_tweets + neg_tweets:
        tweets.append((normalize_text(words), sentiment))

    features = list(get_word_features(get_words(tweets)))
    training_set = [(extract_features(doc), label) for (doc, label)
                    in tweets]

    classifier = NaiveBayesClassifier.train(training_set)
    start_server(classifier)


def start_server(classifier):
    context = zmq.Context()
    # Create a new reply server.
    socket = context.socket(zmq.REP)

    socket.bind("tcp://127.0.0.1:10001")

    while True:
        message = socket.recv()
        response = handle_message(classifier, str(message, "utf-8"))
        socket.send_unicode(response)


def handle_message(classifier, message):
    twfeat = extract_features(normalize_text(message))
    prob_result = classifier.prob_classify(twfeat)
    return "{}".format(prob_result.prob('positive') -
                       prob_result.prob('negative'))


def get_words(tweets):
    all_words = []
    for (words, _) in tweets:
        all_words.extend(words)

    return all_words


def get_word_features(wordlist):
    return FreqDist(wordlist).keys()


def extract_features(document):
    return {word: True for word in document}


if __name__ == "__main__":
    learn_main()
